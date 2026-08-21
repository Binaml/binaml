//! Gate weight packing and i8 activation helpers.

/// Boolean sign: 1 if `x >= 0`, else 0.
#[inline]
pub fn sign(x: i8) -> u8 {
    u8::from(x >= 0)
}

/// Table lane from parent activations.
#[inline]
pub fn lane(act_a: i8, act_b: i8) -> u8 {
    2 * sign(act_a) + sign(act_b)
}

/// Pole target for a definite parent sign.
#[inline]
pub fn pole(sign_bit: u8) -> i8 {
    if sign_bit != 0 { 127 } else { -128 }
}

/// Stream boolean → sink target (`±1`).
#[inline]
pub fn bool_target(y: bool) -> i8 {
    if y { 1 } else { -1 }
}

/// Cost to flip activation `v` to sign `want` (`0` or `1`).
#[inline]
pub fn flip_cost(v: i8, want: u8) -> u8 {
    if sign(v) == want {
        0
    } else if want != 0 {
        v.wrapping_neg() as u8
    } else {
        (v as u8).saturating_add(1)
    }
}

/// Total cost to realize row `s` for target `T`.
pub fn row_total(s: u8, act_a: i8, act_b: i8, weights: [i8; 4], target: i8) -> u8 {
    let s_a = s >> 1;
    let s_b = s & 1;
    let input = flip_cost(act_a, s_a) + flip_cost(act_b, s_b);
    let weight = weights[s as usize].abs_diff(target);
    input + weight
}

/// Minimum total cost over all four rows.
pub fn min_total(act_a: i8, act_b: i8, weights: [i8; 4], target: i8) -> u8 {
    (0..4u8)
        .map(|s| row_total(s, act_a, act_b, weights, target))
        .min()
        .unwrap_or(0)
}

/// Number of parent signs that must flip to realize row `s`.
#[inline]
pub fn sign_mismatches(s: u8, act_a: i8, act_b: i8) -> u8 {
    let s_a = s >> 1;
    let s_b = s & 1;
    u8::from(sign(act_a) != s_a) + u8::from(sign(act_b) != s_b)
}

/// Row minimizing `total(s)` among rows reachable with at most one parent sign flip.
pub fn best_row_at_most_one_flip(act_a: i8, act_b: i8, weights: [i8; 4], target: i8) -> u8 {
    (0..4u8)
        .filter(|&s| sign_mismatches(s, act_a, act_b) <= 1)
        .min_by_key(|&s| row_total(s, act_a, act_b, weights, target))
        .unwrap_or(0)
}

#[inline]
pub fn get_weight(packed: u32, row: u8) -> i8 {
    let shift = 8 * (row as u32);
    (((packed >> shift) & 0xFF) as u8) as i8
}

#[inline]
pub fn set_weight(packed: &mut u32, row: u8, w: i8) {
    let shift = 8 * (row as u32);
    let mask = !(0xFFu32 << shift);
    *packed = (*packed & mask) | (((w as u8) as u32) << shift);
}

/// Nudge weight by 1 toward target `T`.
#[inline]
pub fn nudge_weight(w: i8, target: i8) -> i8 {
    if w < target {
        w.saturating_add(1)
    } else if w > target {
        w.saturating_add(-1)
    } else {
        w
    }
}

#[inline]
pub fn row_weights(packed: u32) -> [i8; 4] {
    [
        get_weight(packed, 0),
        get_weight(packed, 1),
        get_weight(packed, 2),
        get_weight(packed, 3),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_threshold() {
        assert_eq!(sign(-1), 0);
        assert_eq!(sign(0), 1);
        assert_eq!(sign(127), 1);
    }

    #[test]
    fn weight_pack_roundtrip() {
        let mut packed = 0u32;
        set_weight(&mut packed, 0, 10);
        set_weight(&mut packed, 3, -128);
        assert_eq!(get_weight(packed, 0), 10);
        assert_eq!(get_weight(packed, 3), -128);
    }

    #[test]
    fn nudge_toward_target() {
        assert_eq!(nudge_weight(0, -1), -1);
        assert_eq!(nudge_weight(-5, 1), -4);
    }
}
