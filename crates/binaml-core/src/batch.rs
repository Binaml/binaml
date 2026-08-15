#[derive(Clone, Copy)]
pub struct SignBatch<'a> {
    pub feature_columns: &'a [&'a [bool]],
    pub signs: &'a [bool],
}
