//! Per-wire gate helpers: independent parent weights, summed output.

use crate::gate::{get_weight, sign};

/// Stream boolean → pole target (`+127` / `−128`).
#[inline]
pub fn bool_target_pole(y: bool) -> i8 {
    crate::gate::pole(u8::from(y))
}

/// Lane for parent `a` given sign bit `sa ∈ {0,1}`.
#[inline]
pub fn lane_a(sa: u8) -> u8 {
    sa
}

/// Lane for parent `b` given sign bit `sb ∈ {0,1}`.
#[inline]
pub fn lane_b(sb: u8) -> u8 {
    2 + sb
}

/// Pack `(sa, sb)` into one byte for observe scratch.
#[inline]
pub fn pack_signs(sa: u8, sb: u8) -> u8 {
    sa | (sb << 1)
}

#[inline]
pub fn unpack_sa(packed: u8) -> u8 {
    packed & 1
}

#[inline]
pub fn unpack_sb(packed: u8) -> u8 {
    (packed >> 1) & 1
}

#[inline]
pub fn clamp_i8(x: i16) -> i8 {
    x.clamp(i8::MIN as i16, i8::MAX as i16) as i8
}

/// Forward sum and boolean polarity from parent activations.
pub fn forward_sum(packed: u32, act_a: i8, act_b: i8) -> (i16, u8) {
    let sa = sign(act_a);
    let sb = sign(act_b);
    let wa = get_weight(packed, lane_a(sa)) as i16;
    let wb = get_weight(packed, lane_b(sb)) as i16;
    (wa + wb, pack_signs(sa, sb))
}

/// Gate boolean readout matches pole target.
#[inline]
pub fn activation_matches_target(act: i8, target: i8) -> bool {
    (act >= 0) == (target > 0)
}

/// Nudge weight toward pole target `T`; no-op at `i8` bounds.
#[inline]
pub fn nudge_weight(w: i8, target: i8) -> i8 {
    if target > 0 {
        if w < 127 {
            w + 1
        } else {
            w
        }
    } else if w > -128 {
        w - 1
    } else {
        w
    }
}

/// Parent wire weights at gate (`w0`, `w1`).
#[inline]
pub fn parent_weights(packed: u32, is_parent_b: bool) -> (i8, i8) {
    let base = if is_parent_b { 2u8 } else { 0 };
    (get_weight(packed, base), get_weight(packed, base + 1))
}

/// Desired parent sign from weight pair and gate target (argmax / argmin).
/// Tie on `w0 == w1`: keep current sign, no upstream write.
pub fn want_sign(target: i8, w0: i8, w1: i8, _act_p: i8) -> Option<u8> {
    if w0 == w1 {
        return None;
    }
    Some(if target > 0 {
        u8::from(w1 > w0)
    } else {
        u8::from(w1 < w0)
    })
}

/// Why backprop did or did not fire for an internal parent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpropOutcome {
    Fired { want: u8 },
    AlreadyAligned,
    WeightTie,
}

/// Whether to propagate to parent when gate output mismatches target.
pub fn backprop_outcome(target: i8, w0: i8, w1: i8, act_p: i8) -> BackpropOutcome {
    let Some(ws) = want_sign(target, w0, w1, act_p) else {
        return BackpropOutcome::WeightTie;
    };
    if sign(act_p) == ws {
        return BackpropOutcome::AlreadyAligned;
    }
    BackpropOutcome::Fired { want: ws }
}

/// Parent sign to request upstream, if any.
pub fn backprop_sign(target: i8, w0: i8, w1: i8, act_p: i8) -> Option<u8> {
    match backprop_outcome(target, w0, w1, act_p) {
        BackpropOutcome::Fired { want } => Some(want),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::set_weight;

    #[test]
    fn bool_target_pole_values() {
        assert_eq!(bool_target_pole(true), 127);
        assert_eq!(bool_target_pole(false), -128);
    }

    #[test]
    fn forward_sum_selects_per_parent_sign() {
        let mut packed = 0u32;
        set_weight(&mut packed, 0, 10); // w_a0
        set_weight(&mut packed, 1, 20); // w_a1
        set_weight(&mut packed, 2, -5); // w_b0
        set_weight(&mut packed, 3, 30); // w_b1
        let (sum, signs) = forward_sum(packed, -1, 1);
        assert_eq!(sum, 10 + 30);
        assert_eq!(signs, pack_signs(0, 1));
        let (sum2, _) = forward_sum(packed, 1, -1);
        assert_eq!(sum2, 20 + (-5));
    }

    #[test]
    fn nudge_respects_bounds() {
        assert_eq!(nudge_weight(127, 127), 127);
        assert_eq!(nudge_weight(126, 127), 127);
        assert_eq!(nudge_weight(-128, -128), -128);
        assert_eq!(nudge_weight(-127, -128), -128);
    }

    #[test]
    fn want_sign_tie_keeps_none() {
        assert_eq!(want_sign(127, 5, 5, -1), None);
        assert_eq!(want_sign(127, 5, 5, 1), None);
    }

    #[test]
    fn want_sign_max_for_target_one() {
        assert_eq!(want_sign(127, 3, 7, 0), Some(1));
        assert_eq!(want_sign(127, 7, 3, 1), Some(0));
    }

    #[test]
    fn want_sign_min_for_target_zero() {
        assert_eq!(want_sign(-128, 3, 7, 1), Some(0));
        assert_eq!(want_sign(-128, 7, 3, 0), Some(1));
    }

    #[test]
    fn backprop_fires_when_misaligned() {
        assert_eq!(backprop_sign(127, 3, 7, -1), Some(1));
        assert_eq!(backprop_sign(-128, 3, 7, 1), Some(0));
    }

    #[test]
    fn backprop_skips_when_aligned() {
        assert_eq!(backprop_sign(127, 3, 7, 1), None);
        assert_eq!(backprop_sign(-128, 3, 7, -1), None);
    }
}
