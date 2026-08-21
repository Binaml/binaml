//! Shared bit/score helpers.

use super::dgp::BITS_PER_U8;

#[inline]
pub fn sample<'a>(x_flat: &'a [u8], n_inputs: usize, idx: usize) -> &'a [u8] {
    let off = idx * n_inputs;
    &x_flat[off..off + n_inputs]
}

#[inline]
pub fn score_i8(xs: &[u8], w0: &[i8], w1: &[i8]) -> i32 {
    debug_assert_eq!(xs.len() * BITS_PER_U8, w0.len());
    let mut s = 0i32;
    for (j, &byte) in xs.iter().enumerate() {
        let base = j * BITS_PER_U8;
        s += score_byte(byte, &w0[base..base + BITS_PER_U8], &w1[base..base + BITS_PER_U8]);
    }
    s
}

#[inline]
pub fn score_byte(byte: u8, w0: &[i8], w1: &[i8]) -> i32 {
    debug_assert_eq!(w0.len(), BITS_PER_U8);
    let mut s = 0i32;
    let mut b = byte;
    for i in 0..BITS_PER_U8 {
        if b & 1 != 0 {
            s += w1[i] as i32;
        } else {
            s += w0[i] as i32;
        }
        b >>= 1;
    }
    s
}

#[inline]
pub fn score_mem_u8(xs: &[u8], w0: &[u8], w1: &[u8], midpoint: i32) -> i32 {
    debug_assert_eq!(xs.len(), w0.len());
    debug_assert_eq!(xs.len(), w1.len());
    let mut s = 0i32;
    for ((&byte, &w0b), &w1b) in xs.iter().zip(w0.iter()).zip(w1.iter()) {
        s += score_byte_mem_packed(byte, w0b, w1b);
    }
    s - midpoint
}

/// Sum active memory bits: `w1` where input is 1, `w0` where input is 0.
#[inline]
pub fn score_byte_mem_packed(byte: u8, w0: u8, w1: u8) -> i32 {
    ((w0 & !byte) | (w1 & byte)).count_ones() as i32
}

#[inline]
pub fn update_byte_mem(w0: &mut u8, w1: &mut u8, byte: u8, target: u8) {
    let mask = 0u8.wrapping_sub(target); // 0xFF if target != 0, else 0
    *w1 = (*w1 & !byte) | (byte & mask);
    *w0 = (*w0 & byte) | (!byte & mask);
}

#[inline]
#[cfg(test)]
pub fn score_byte_mem(byte: u8, w0: &[u8], w1: &[u8]) -> i32 {
    debug_assert_eq!(w0.len(), BITS_PER_U8);
    let mut s = 0i32;
    let mut b = byte;
    for i in 0..BITS_PER_U8 {
        if b & 1 != 0 {
            s += w1[i] as i32;
        } else {
            s += w0[i] as i32;
        }
        b >>= 1;
    }
    s
}

/// Symmetric memory score: matching bits minus midpoint.
///
/// Per bit: stored `m`, input `b` → contribution is `(m == b)`; over a byte that
/// is `8 - popcount(m ^ x)`.
#[inline]
pub fn score_sym_u8(xs: &[u8], mem: &[u8], midpoint: i32) -> i32 {
    debug_assert_eq!(xs.len(), mem.len());
    let mut s = 0i32;
    for (&byte, &m) in xs.iter().zip(mem.iter()) {
        s += score_byte_sym(byte, m);
    }
    s - midpoint
}

#[inline]
pub fn score_byte_sym(byte: u8, mem: u8) -> i32 {
    (BITS_PER_U8 - (byte ^ mem).count_ones() as usize) as i32
}

pub fn count_errors<F>(x_flat: &[u8], y: &[bool], n_inputs: usize, mut pred: F) -> usize
where
    F: FnMut(&[u8]) -> bool,
{
    y.iter()
        .enumerate()
        .filter(|(i, &yi)| pred(sample(x_flat, n_inputs, *i)) != yi)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unpack(w: u8) -> [u8; BITS_PER_U8] {
        std::array::from_fn(|i| (w >> i) & 1)
    }

    #[test]
    fn packed_mem_score_matches_unpacked() {
        for byte in 0u8..=255 {
            for w0 in 0u8..=255 {
                for w1 in 0u8..=255 {
                    let unpacked_w0 = unpack(w0);
                    let unpacked_w1 = unpack(w1);
                    assert_eq!(
                        score_byte_mem_packed(byte, w0, w1),
                        score_byte_mem(byte, &unpacked_w0, &unpacked_w1),
                    );
                }
            }
        }
    }

    #[test]
    fn packed_mem_update_matches_scalar() {
        for byte in 0u8..=255 {
            for target in [0u8, 1] {
                let mut w0 = 0xA5u8;
                let mut w1 = 0x3Cu8;
                let w0_before = w0;
                let w1_before = w1;

                update_byte_mem(&mut w0, &mut w1, byte, target);

                let mut uw0 = unpack(w0_before);
                let mut uw1 = unpack(w1_before);
                for i in 0..BITS_PER_U8 {
                    if byte & (1 << i) != 0 {
                        uw1[i] = target;
                    } else {
                        uw0[i] = target;
                    }
                }
                assert_eq!(w0, uw0.iter().enumerate().fold(0u8, |acc, (i, &b)| acc | (b << i)));
                assert_eq!(w1, uw1.iter().enumerate().fold(0u8, |acc, (i, &b)| acc | (b << i)));
            }
        }
    }
}
