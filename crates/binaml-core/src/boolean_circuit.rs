/// Shared truth-table lookup for two-input boolean gates.
#[inline]
pub(crate) fn evaluate_truth_table(truth_table: u8, first: bool, second: bool) -> bool {
    truth_table & (1_u8 << ((u8::from(first) << 1) | u8::from(second))) != 0
}
