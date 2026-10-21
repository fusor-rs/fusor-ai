#![expect(
    unsafe_code,
    reason = "byte and f32 views of one 8-byte aligned allocation"
)]

/// An 8-byte aligned byte buffer, so f32 views of it are always aligned.
pub(crate) struct Blob {
    words: Vec<u64>,
    len: usize,
}

impl Blob {
    pub(crate) fn zeroed(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(8)],
            len,
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: the Vec<u64> owns at least `len` initialized bytes; u8 has no alignment needs.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast(), self.len) }
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `bytes`, with exclusive access through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast(), self.len) }
    }

    /// The f32 values from byte offset `start` (a multiple of 4) to the end of the buffer.
    pub(crate) fn f32s(&self, start: usize) -> &[f32] {
        assert!(
            start.is_multiple_of(4),
            "f32 views start on a 4-byte boundary"
        );
        let count = (self.len - start) / 4;
        // SAFETY: the buffer is 8-byte aligned and `start` is a multiple of 4, so the pointer is
        // aligned for f32; `count` f32s lie within the `len` initialized bytes; every bit
        // pattern is a valid f32.
        unsafe { std::slice::from_raw_parts(self.bytes().as_ptr().add(start).cast(), count) }
    }
}
