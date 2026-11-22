//! Moving text into and token ids out of wasm memory.
//!
//! The generated bindings copy a string with a per-char loop up to its first
//! non-ASCII char, allocate and free a buffer on every call, and hand ids
//! back through a fresh allocation plus a view and a copy. Here the text goes
//! into buffers that live across calls: by `TextEncoder.encodeInto`, or, when
//! most of it is outside ASCII, as UTF-16 code units that [`utf16`] converts.
//! Ids come back as a single `slice` of a cached view.

use std::cell::RefCell;

use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;

use crate::utf16;

#[wasm_bindgen(inline_js = r#"
const te = new TextEncoder();
let mem = null, u8 = null, u16 = null, u32 = null;
function views() {
  u8 = new Uint8Array(mem.buffer);
  u16 = new Uint16Array(mem.buffer);
  u32 = new Uint32Array(mem.buffer);
}
export function tk_bind(m) {
  mem = m;
  views();
}
// whether TextEncoder.encode plus a copy beats encodeInto, timed once on
// 1 KB of ASCII: JavaScriptCore's encodeInto runs about 20 times slower
// than its encode, V8's about 1.6 times faster
let copyEncoded = null;
function pick() {
  if (typeof performance === 'undefined') return false;
  const s = 'The quick brown fox jumps over the lazy dog. '.repeat(24);
  const buf = new Uint8Array(4096);
  const time = (f) => {
    const t0 = performance.now();
    let k = 0, t = 0;
    do {
      for (let i = 0; i < 16; i++) f();
      k += 16;
      t = performance.now() - t0;
    } while (t < 1);
    return t / k;
  };
  const into = time(() => te.encodeInto(s, buf));
  const copy = time(() => buf.set(te.encode(s)));
  return copy < into * 0.8;
}
// mostly chars of two or more utf-8 bytes, judged from 16 samples
function wide(s, n) {
  const step = n >>> 4;
  let w = 0;
  for (let i = 0; i < 16; i++) if (s.charCodeAt(i * step) > 0x7f) w++;
  return w >= 10;
}
// 2 * length of the utf-8 written at p8, or 2 * units + 1 of the utf-16
// written at p16; -1 - units when the buffers are too small
export function tk_read(s, p8, c8, p16, c16) {
  if (typeof s !== 'string') throw new TypeError('text must be a string');
  const n = s.length;
  // 16: the slack the utf-16 conversion may write past its output
  if (n * 3 + 16 > c8 || n > c16) return -1 - n;
  // memory.grow detaches the old buffer
  if (u8.byteLength === 0) views();
  if (n < 32) {
    let i = 0;
    for (; i < n; i++) {
      const c = s.charCodeAt(i);
      if (c > 0x7f) break;
      u8[p8 + i] = c;
    }
    if (i === n) return 2 * n;
    return 2 * (i + te.encodeInto(s.slice(i), u8.subarray(p8 + i, p8 + c8)).written);
  }
  if (n >= 64 && wide(s, n)) {
    const v = u16, at = p16 >>> 1;
    let i = 0;
    for (; i + 4 <= n; i += 4) {
      v[at + i] = s.charCodeAt(i);
      v[at + i + 1] = s.charCodeAt(i + 1);
      v[at + i + 2] = s.charCodeAt(i + 2);
      v[at + i + 3] = s.charCodeAt(i + 3);
    }
    for (; i < n; i++) v[at + i] = s.charCodeAt(i);
    return 2 * n + 1;
  }
  if (copyEncoded === null) copyEncoded = pick();
  if (copyEncoded) {
    const b = te.encode(s);
    u8.set(b, p8);
    return 2 * b.length;
  }
  return 2 * te.encodeInto(s, u8.subarray(p8, p8 + c8)).written;
}
export function tk_ids(ptr, len) {
  if (u32.byteLength === 0) views();
  const at = ptr >>> 2;
  return u32.slice(at, at + len);
}
"#)]
extern "C" {
    fn tk_bind(memory: JsValue);
    fn tk_read(text: &JsValue, p8: *mut u8, c8: usize, p16: *mut u16, c16: usize) -> f64;
    fn tk_ids(ptr: *const u32, len: usize) -> JsValue;
}

struct Buffers {
    utf8: Vec<u8>,
    utf16: Vec<u16>,
}

thread_local! {
    static BUF: RefCell<Buffers> = const {
        RefCell::new(Buffers { utf8: Vec::new(), utf16: Vec::new() })
    };
}

pub(crate) fn bind() {
    tk_bind(wasm_bindgen::memory());
}

/// Run `f` on the utf-8 of the JS string `text`.
pub(crate) fn with_text<R>(text: &JsValue, f: impl FnOnce(&str) -> R) -> R {
    BUF.with(|b| {
        let Buffers { utf8, utf16 } = &mut *b.borrow_mut();
        let r = loop {
            let r = tk_read(
                text,
                utf8.as_mut_ptr(),
                utf8.capacity(),
                utf16.as_mut_ptr(),
                utf16.capacity(),
            );
            if r >= 0.0 {
                break r as usize;
            }
            let units = (-r) as usize - 1;
            utf8.clear();
            utf8.reserve(units * 3 + utf16::SLACK);
            utf16.clear();
            utf16.reserve(units);
        };
        let len = if r & 1 == 0 {
            r >> 1
        } else {
            // SAFETY: the code units were written into the capacity
            unsafe { utf16.set_len(r >> 1) };
            // SAFETY: tk_read checked the room for 3 bytes a unit plus slack
            unsafe { utf16::transcode(utf16, utf8.as_mut_ptr()) }
        };
        // SAFETY: `len` bytes were written into the capacity
        unsafe { utf8.set_len(len) };
        // SAFETY: both paths write utf-8, lone surrogates as U+FFFD
        f(unsafe { std::str::from_utf8_unchecked(utf8) })
    })
}

/// A `Uint32Array` copy of `ids`.
pub(crate) fn ids(ids: &[u32]) -> JsValue {
    tk_ids(ids.as_ptr(), ids.len())
}
