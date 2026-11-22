//! Process-wide worker pool for encoding one long text in parallel.
//!
//! The pool serves one caller at a time, for one job or a [`Session`] of
//! several. A caller that finds it busy (another thread holds it, or the
//! caller is itself inside a job) gets `None` and does the work on its own
//! thread, so work never queues and the pool never runs more threads than it
//! was sized for. The caller takes chunks too, so a job uses `threads - 1`
//! pool workers.
//!
//! After a job a worker spins for twice the job's duration (at most
//! [`SPIN_MAX`]) so back-to-back calls find it awake, then parks; the caller
//! of the next job unparks it.

use std::any::Any;
use std::cell::UnsafeCell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// Upper bound on threads per job, the caller included.
const MAX_THREADS: usize = 256;
const SPIN_MAX: Duration = Duration::from_micros(50);

/// The configured thread count; 0 means "use [`default_threads`]".
static THREADS: AtomicUsize = AtomicUsize::new(0);
static DEFAULT: OnceLock<usize> = OnceLock::new();

pub(crate) fn set_parallelism(n: usize) {
    THREADS.store(n.min(MAX_THREADS), Relaxed);
}

/// Threads a parallel job may use, the caller included. Never creates the
/// pool.
#[inline]
pub(crate) fn threads() -> usize {
    match THREADS.load(Relaxed) {
        0 => *DEFAULT.get_or_init(default_threads),
        n => n,
    }
}

fn default_threads() -> usize {
    let from_env = std::env::var("TIKTOKEN_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0);
    from_env.unwrap_or_else(performance_cores).min(MAX_THREADS)
}

fn available() -> usize {
    thread::available_parallelism().map_or(1, |n| n.get())
}

/// Logical CPUs of the fastest core type. On Apple silicon the efficiency
/// cores are several times slower, and a chunk that lands on one holds up the
/// whole job, so only the performance cores count.
#[cfg(target_os = "macos")]
fn performance_cores() -> usize {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let mut n: c_int = 0;
    let mut len = size_of::<c_int>();
    // SAFETY: `oldp` points at a c_int and `oldlenp` holds its size
    let r = unsafe {
        sysctlbyname(
            c"hw.perflevel0.logicalcpu".as_ptr(),
            (&raw mut n).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    // intel macs have no perflevel0
    if r == 0 && n > 0 {
        n as usize
    } else {
        available()
    }
}

#[cfg(not(target_os = "macos"))]
fn performance_cores() -> usize {
    available()
}

/// Run `f(0..n)` on the pool and return the results in index order, or
/// `None` if the pool is busy. A panic in any `f(i)` is re-raised here after
/// every other index has finished.
pub(crate) fn run<R: Send>(
    n: usize,
    threads: usize,
    f: impl Fn(usize) -> R + Sync,
) -> Option<Vec<R>> {
    Some(session()?.run(n, threads, |i, _| f(i)))
}

/// Hold the pool for several jobs in a row, or `None` if it is busy.
pub(crate) fn session() -> Option<Session> {
    let pool = POOL.get_or_init(Pool::new);
    pool.busy
        .compare_exchange(false, true, Acquire, Relaxed)
        .is_ok()
        // lazily: a `Session` built and dropped on failure would release
        // the holder's claim
        .then(|| Session { pool })
}

/// The pool, held until dropped; no other thread can start a job meanwhile.
pub(crate) struct Session {
    pool: &'static Pool,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.pool.busy.store(false, Release);
    }
}

impl Session {
    /// Run `f(i, me)` for `i` in `0..n` and return the results in index
    /// order. `me` names the thread that runs the call: 0 for the caller,
    /// `1..threads` for pool workers, the same in every job of the session.
    /// A panic in any call is re-raised here after every other index has
    /// finished.
    pub(crate) fn run<R: Send>(
        &self,
        n: usize,
        threads: usize,
        f: impl Fn(usize, usize) -> R + Sync,
    ) -> Vec<R> {
        let slots: Vec<Slot<R>> = (0..n).map(|_| Slot(UnsafeCell::new(None))).collect();
        let call = |i: usize, me: usize| {
            let r = f(i, me);
            // SAFETY: index `i` is claimed by exactly one thread
            unsafe { *slots[i].0.get() = Some(r) };
        };
        self.exec(n, threads, false, &call);
        slots
            .into_iter()
            .map(|s| s.0.into_inner().unwrap_or_else(|| unreachable!()))
            .collect()
    }

    /// Run `f(i, me)` for `i` in `0..n`, where each thread first takes the index
    /// equal to its `me` (see [`run`](Self::run)), then any left over: work
    /// that reads what thread `i` wrote in an earlier job of the session
    /// finds it in that thread's cache.
    pub(crate) fn run_affine(&self, n: usize, threads: usize, f: impl Fn(usize, usize) + Sync) {
        self.exec(n, threads, true, &f);
    }

    fn exec(&self, n: usize, threads: usize, affine: bool, run: &(dyn Fn(usize, usize) + Sync)) {
        let job = Job {
            next: AtomicUsize::new(0),
            claimed: if affine {
                (0..n).map(|_| AtomicBool::new(false)).collect()
            } else {
                Box::default()
            },
            n,
            pending: AtomicUsize::new(n),
            workers: threads.min(n).min(MAX_THREADS).saturating_sub(1),
            start: Instant::now(),
            run,
            panic: Mutex::new(None),
        };
        self.pool.dispatch(&job);
        if let Some(p) = job
            .panic
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
        {
            panic::resume_unwind(p);
        }
    }
}

struct Slot<R>(UnsafeCell<Option<R>>);

// SAFETY: each slot is written by one thread and read after the job is done
unsafe impl<R: Send> Sync for Slot<R> {}

struct Job<'a> {
    next: AtomicUsize,
    /// Per index, whether a thread has taken it; empty unless the job is
    /// affine.
    claimed: Box<[AtomicBool]>,
    n: usize,
    pending: AtomicUsize,
    /// Pool workers allowed to join: those with a smaller index.
    workers: usize,
    start: Instant,
    run: &'a (dyn Fn(usize, usize) + Sync),
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Job<'_> {
    /// Take and run indices until none is left; `me` is the calling
    /// thread's number.
    fn work(&self, me: usize) {
        if self.claimed.is_empty() {
            loop {
                let i = self.next.fetch_add(1, Relaxed);
                if i >= self.n {
                    return;
                }
                self.call(i, me);
            }
        }
        if me < self.n && !self.claimed[me].swap(true, Relaxed) {
            self.call(me, me);
        }
        for i in 0..self.n {
            if !self.claimed[i].load(Relaxed) && !self.claimed[i].swap(true, Relaxed) {
                self.call(i, me);
            }
        }
    }

    fn call(&self, i: usize, me: usize) {
        if let Err(p) = panic::catch_unwind(AssertUnwindSafe(|| (self.run)(i, me))) {
            let mut first = self.panic.lock().unwrap_or_else(PoisonError::into_inner);
            first.get_or_insert(p);
        }
        self.pending.fetch_sub(1, Release);
    }

    fn wait_done(&self) {
        let mut k = 0u32;
        while self.pending.load(Acquire) != 0 {
            k = k.wrapping_add(1);
            if k < 1 << 14 {
                std::hint::spin_loop();
            } else {
                thread::yield_now();
            }
        }
    }
}

#[repr(align(128))]
struct Padded<T>(T);

#[derive(Default)]
struct Worker {
    thread: OnceLock<Thread>,
    sleeping: AtomicBool,
}

impl Worker {
    fn wake(&self) {
        // a load first: a spinning worker's flag stays false, and swapping
        // it anyway pulled its line over from another core for every job
        if self.sleeping.load(SeqCst)
            && self.sleeping.swap(false, SeqCst)
            && let Some(t) = self.thread.get()
        {
            t.unpark();
        }
    }
}

struct Pool {
    busy: AtomicBool,
    epoch: Padded<AtomicU64>,
    /// The running job; null between jobs. Points into the caller's frame,
    /// which stays alive until `inside` drops back to zero.
    job: AtomicPtr<Job<'static>>,
    /// Workers that may be holding a pointer to the current job.
    inside: Padded<AtomicUsize>,
    /// Workers started so far; grows only while `busy` is held.
    spawned: AtomicUsize,
    workers: Box<[Worker]>,
}

static POOL: OnceLock<Pool> = OnceLock::new();

impl Pool {
    fn new() -> Pool {
        Pool {
            busy: AtomicBool::new(false),
            epoch: Padded(AtomicU64::new(0)),
            job: AtomicPtr::new(std::ptr::null_mut()),
            inside: Padded(AtomicUsize::new(0)),
            spawned: AtomicUsize::new(0),
            workers: (1..MAX_THREADS).map(|_| Worker::default()).collect(),
        }
    }

    #[cfg(not(miri))]
    fn dispatch(&'static self, job: &Job<'_>) {
        let workers = self.spawn_up_to(job.workers);
        if workers == 0 {
            job.work(0);
            return;
        }
        let ptr = (job as *const Job<'_>).cast_mut().cast::<Job<'static>>();
        self.job.store(ptr, SeqCst);
        self.epoch.0.fetch_add(1, SeqCst);
        // a wake-up tree (each woken worker waking two more), and one
        // wake-all call on the epoch word (os_sync_wake_by_address_all),
        // both measured slower than waking everyone from here
        for w in &self.workers[..workers] {
            w.wake();
        }
        job.work(0);
        job.wait_done();
        self.job.store(std::ptr::null_mut(), SeqCst);
        while self.inside.0.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
    }

    // miri reports threads still alive at exit, so it gets scoped threads
    #[cfg(miri)]
    fn dispatch(&'static self, job: &Job<'_>) {
        thread::scope(|s| {
            for k in 0..job.workers {
                s.spawn(move || job.work(k + 1));
            }
            job.work(0);
        });
    }

    /// Start workers until `want` exist (or spawning fails); returns how many
    /// can join.
    #[cfg(not(miri))]
    fn spawn_up_to(&'static self, want: usize) -> usize {
        let mut have = self.spawned.load(Relaxed);
        while have < want {
            let epoch = self.epoch.0.load(SeqCst);
            let idx = have;
            let spawned = thread::Builder::new()
                .name(format!("tiktoken-{idx}"))
                .spawn(move || self.worker_main(idx, epoch));
            let Ok(handle) = spawned else { break };
            let _ = self.workers[idx].thread.set(handle.thread().clone());
            have += 1;
            self.spawned.store(have, Relaxed);
        }
        have.min(want)
    }

    fn worker_main(&'static self, idx: usize, mut seen: u64) {
        let mut spin = Duration::ZERO;
        loop {
            seen = self.wait(idx, seen, spin);
            self.inside.0.fetch_add(1, SeqCst);
            let ptr = self.job.load(SeqCst);
            // SAFETY: non-null means the caller has not passed its `inside`
            // check yet, so the job outlives this use
            if let Some(job) = unsafe { ptr.as_ref() }
                && idx < job.workers
            {
                job.work(idx + 1);
                spin = (job.start.elapsed() * 2).min(SPIN_MAX);
            }
            self.inside.0.fetch_sub(1, SeqCst);
        }
    }

    /// Wait for the epoch to move past `seen`: spin for `spin`, then park.
    fn wait(&self, idx: usize, seen: u64, spin: Duration) -> u64 {
        let t0 = Instant::now();
        let mut k = 0u32;
        loop {
            let e = self.epoch.0.load(Acquire);
            if e != seen {
                return e;
            }
            k = k.wrapping_add(1);
            if k.is_multiple_of(64) && t0.elapsed() >= spin {
                break;
            }
            std::hint::spin_loop();
        }
        let me = &self.workers[idx];
        loop {
            me.sleeping.store(true, SeqCst);
            let e = self.epoch.0.load(SeqCst);
            if e != seen {
                me.sleeping.store(false, Relaxed);
                return e;
            }
            thread::park();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `run`, waiting while other tests in this process hold the pool.
    fn run_now<R: Send>(n: usize, threads: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
        loop {
            if let Some(r) = run(n, threads, &f) {
                return r;
            }
            thread::yield_now();
        }
    }

    #[test]
    fn results_come_back_in_order() {
        for n in [1, 2, 3, 7, 64] {
            let got = run_now(n, 4, |i| i * 10);
            assert_eq!(got, (0..n).map(|i| i * 10).collect::<Vec<_>>());
        }
    }

    #[test]
    fn panic_in_a_chunk_reaches_the_caller() {
        let r = panic::catch_unwind(|| {
            run_now(16, 4, |i| {
                assert_ne!(i, 11, "chunk eleven");
                i
            })
        });
        let msg = r.unwrap_err();
        let msg = msg.downcast_ref::<String>().unwrap();
        assert!(msg.contains("chunk eleven"), "{msg}");
        // the pool is usable afterwards
        assert_eq!(run_now(4, 4, |i| i), vec![0, 1, 2, 3]);
    }

    #[test]
    fn nested_job_is_refused() {
        let inner = run_now(4, 4, |_| run(2, 2, |i| i).is_none());
        // the pool is busy for the outer job's whole duration
        assert!(inner.iter().all(|&refused| refused));
    }

    #[test]
    fn a_session_holds_the_pool_across_jobs() {
        let s = loop {
            if let Some(s) = session() {
                break s;
            }
            thread::yield_now();
        };
        let owners = s.run(16, 4, |_, me| me);
        assert!(owners.iter().all(|&me| me < 4));
        // refused attempts leave the holder's claim in place
        assert!(run(2, 2, |i| i).is_none());
        assert!(session().is_none());
        let hits: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
        s.run_affine(4, 4, |i, _| {
            hits[i].fetch_add(1, Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Relaxed) == 1));
        drop(s);
        assert_eq!(run_now(3, 3, |i| i), vec![0, 1, 2]);
    }
}
