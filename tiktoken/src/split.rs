//! Encoding a long text as independent chunks, and where to cut it.
//!
//! Chunk boundaries are cuts (see `cut`): encoding the chunks on their own
//! gives the ids of the whole text.

use std::cell::RefCell;
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize};

use crate::cut::{self, Rules};
use crate::encode_loop::Engine;
use crate::pool;

/// Nominal chunk length in bytes.
pub(crate) const CHUNK: usize = 8 * 1024;

/// Encode `text` as chunks on the pool, appending to `result`; falls back to
/// this thread when parallelism is 1, the text has no cut points, or the pool
/// is busy. At most `max_chunks` chunks, by default a multiple of the thread
/// count up to two per thread.
///
/// One job does it all: each thread takes chunks and appends their ids to a
/// buffer of its own that lives across calls; the thread that finishes the
/// last chunk makes room in `result`, and each thread then copies its own
/// chunks there. A fresh buffer per chunk, read back by whichever thread did
/// the copy, cost about half again the encoding time of the chunk: its lines
/// were last owned by other cores. A second job for the copy cost a round of
/// wake-ups.
#[inline(never)]
pub(crate) fn encode(
    engine: &Engine<'_>,
    text: &str,
    chunk: usize,
    max_chunks: Option<usize>,
    result: &mut Vec<u32>,
) {
    if let Some((threads, cuts)) = plan(engine, text, chunk, max_chunks)
        && let Some(session) = pool::session()
    {
        let n = cuts.len() - 1;
        let base = result.len();
        let job = Fused {
            next: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            lens: (0..n).map(|_| AtomicUsize::new(0)).collect(),
            dst: AtomicPtr::default(),
            failed: AtomicBool::new(false),
            result: SendPtr(std::ptr::from_mut(result)),
        };
        let workers = threads.min(n);
        session.run_affine(workers, workers, |_, _| job.part(engine, text, &cuts));
        let total: usize = job.lens.iter().map(|l| l.load(Relaxed)).sum();
        // SAFETY: the job wrote every element in `base..base + total`
        unsafe { result.set_len(base + total) };
        return;
    }
    result.reserve(text.len() / 3);
    engine.encode_into(text, result);
}

/// The shared state of one fused encode job.
struct Fused {
    next: AtomicUsize,
    /// Chunks encoded so far; `lens[i]` is stored before chunk `i` counts.
    done: AtomicUsize,
    lens: Box<[AtomicUsize]>,
    /// Where the job's ids go in the result, set once every chunk is done.
    dst: AtomicPtr<u32>,
    /// Set when a thread unwinds, so the others stop waiting.
    failed: AtomicBool,
    result: SendPtr<Vec<u32>>,
}

impl Fused {
    /// One thread's share: chunks until none is left, then, once all are
    /// done, the copy of its own.
    fn part(&self, engine: &Engine<'_>, text: &str, cuts: &[usize]) {
        let n = self.lens.len();
        let guard = FailGuard(&self.failed);
        SCRATCH.with_borrow_mut(|s| {
            s.ids.clear();
            s.parts.clear();
            loop {
                let i = self.next.fetch_add(1, Relaxed);
                if i >= n {
                    break;
                }
                let part = &text[cuts[i]..cuts[i + 1]];
                let start = s.ids.len();
                s.ids.reserve(part.len() / 3);
                engine.encode_into(part, &mut s.ids);
                let len = s.ids.len() - start;
                s.parts.push((i, start, len));
                self.lens[i].store(len, Relaxed);
                if self.done.fetch_add(1, AcqRel) + 1 == n {
                    self.publish();
                }
            }
            if s.parts.is_empty() {
                return;
            }
            let Some(dst) = self.wait_dst() else { return };
            let src = s.ids.as_ptr();
            let (mut at, mut k) = (0, 0);
            for &(i, start, len) in &s.parts {
                // a thread takes chunks in increasing order, so the sum of
                // the lengths below `i` carries over from its previous one
                while k < i {
                    at += self.lens[k].load(Relaxed);
                    k += 1;
                }
                // SAFETY: `publish` made room for every chunk's ids at `dst`,
                // and chunk `i` owns `at..at + len`; the source is this
                // thread's buffer
                unsafe { std::ptr::copy_nonoverlapping(src.add(start), dst.add(at), len) };
            }
            if s.ids.capacity() > SCRATCH_KEEP {
                s.ids = Vec::new();
            }
        });
        std::mem::forget(guard);
    }

    /// Run by the thread that finished the last chunk: room for every id.
    fn publish(&self) {
        let total: usize = self.lens.iter().map(|l| l.load(Relaxed)).sum();
        // SAFETY: the caller is blocked in the job until every thread has
        // returned, and only this thread touches the vector meanwhile; the
        // Acquire half of the last `done` increment made every length visible
        let result = unsafe { &mut *self.result.0 };
        result.reserve(total);
        let dst = result.as_mut_ptr().wrapping_add(result.len());
        self.dst.store(dst, Release);
    }

    fn wait_dst(&self) -> Option<*mut u32> {
        let mut k = 0u32;
        loop {
            let p = self.dst.load(Acquire);
            if !p.is_null() {
                return Some(p);
            }
            if self.failed.load(Relaxed) {
                return None;
            }
            k = k.wrapping_add(1);
            if k < 1 << 14 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// Marks the job failed if its thread unwinds, so that no thread waits for a
/// chunk that will never be done.
struct FailGuard<'a>(&'a AtomicBool);

impl Drop for FailGuard<'_> {
    fn drop(&mut self) {
        self.0.store(true, Relaxed);
    }
}

/// Largest scratch buffer a thread keeps after copying out its ids.
const SCRATCH_KEEP: usize = 1 << 18;

struct Scratch {
    ids: Vec<u32>,
    /// `(chunk, start, len)` of each chunk whose ids are in `ids`.
    parts: Vec<(usize, usize, usize)>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = const {
        RefCell::new(Scratch { ids: Vec::new(), parts: Vec::new() })
    };
}

struct SendPtr<T>(*mut T);

// SAFETY: the threads behind it write disjoint ranges, or one at a time
unsafe impl<T> Sync for SendPtr<T> {}

/// The counting twin of [`encode`].
#[inline(never)]
pub(crate) fn count(
    engine: &Engine<'_>,
    text: &str,
    chunk: usize,
    max_chunks: Option<usize>,
) -> usize {
    if let Some((threads, cuts)) = plan(engine, text, chunk, max_chunks) {
        let parts = pool::run(cuts.len() - 1, threads, |i| {
            engine.count(&text[cuts[i]..cuts[i + 1]])
        });
        if let Some(parts) = parts {
            return parts.iter().sum();
        }
    }
    engine.count(text)
}

/// Thread count and chunk boundaries, or `None` to stay on this thread.
fn plan(
    engine: &Engine<'_>,
    text: &str,
    chunk: usize,
    max_chunks: Option<usize>,
) -> Option<(usize, Vec<usize>)> {
    let rules = engine.pre.cut_rules();
    let threads = pool::threads();
    if threads <= 1 {
        return None;
    }
    let cuts = match max_chunks {
        Some(max) => cuts(text.as_bytes(), chunk, max, rules),
        None => cuts(
            text.as_bytes(),
            1,
            chunk_count(text.len(), chunk, threads),
            rules,
        ),
    };
    (cuts.len() > 2).then_some((threads, cuts))
}

/// Chunks for a text of `len` bytes: one per `chunk` bytes, rounded up to a
/// multiple of `threads` once there are more than `threads`, so that every
/// thread gets the same number; at most two per thread, which measured
/// faster than three or four.
fn chunk_count(len: usize, chunk: usize, threads: usize) -> usize {
    let c = len.div_ceil(chunk.max(1));
    if c <= threads {
        return c;
    }
    (c.div_ceil(threads) * threads).min(2 * threads)
}

/// Chunk boundaries of `b`: `0`, the cuts in increasing order, `b.len()`.
/// Aims for `min(len / chunk, max_chunks)` chunks; a nominal cut moves
/// forward to the first cut within half a chunk, and is dropped if there is
/// none.
pub(crate) fn cuts(b: &[u8], chunk: usize, max_chunks: usize, rules: Rules) -> Vec<usize> {
    let n = b.len();
    let chunks = n.div_ceil(chunk.max(1)).min(max_chunks).max(1);
    let step = n / chunks;
    let mut out = Vec::with_capacity(chunks + 1);
    out.push(0);
    for k in 1..chunks {
        let from = (k * step).max(out[out.len() - 1] + 1);
        let to = (from + step / 2).min(n);
        if let Some(c) = cut::next(&b[..to], from, rules) {
            out.push(c);
        }
    }
    out.push(n);
    out
}

#[cfg(test)]
#[path = "split_tests.rs"]
mod tests;
