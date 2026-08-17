/// Signed association between a boolean stream and residual-sign labels.
pub fn association_score(values: &[bool], signs: &[bool]) -> i64 {
    let n = signs.len() as i64;
    let nx = values.iter().filter(|value| **value).count() as i64;
    let ny = signs.iter().filter(|sign| **sign).count() as i64;
    let nxy = values
        .iter()
        .zip(signs)
        .filter(|(value, sign)| **value && **sign)
        .count() as i64;
    n * nxy - nx * ny
}

#[cfg(test)]
mod tests {
    use super::association_score;

    #[test]
    fn association_score_is_zero_for_independent_pattern() {
        let values = [true, true, false, false];
        let signs = [true, false, true, false];
        assert_eq!(association_score(&values, &signs), 0);
    }
}
