#[derive(Clone, Copy)]
struct FlatBatch<'a> {
    features: &'a [bool],
    batch_size: usize,
    feature_count: usize,
}

#[derive(Clone, Copy)]
pub struct SignBatch<'a> {
    pub feature_columns: &'a [&'a [bool]],
    pub signs: &'a [bool],
    flat: Option<FlatBatch<'a>>,
}

impl<'a> SignBatch<'a> {
    pub fn from_columns(feature_columns: &'a [&'a [bool]], signs: &'a [bool]) -> Self {
        Self {
            feature_columns,
            signs,
            flat: None,
        }
    }

    pub fn from_flat(
        features: &'a [bool],
        batch_size: usize,
        feature_count: usize,
        signs: &'a [bool],
    ) -> Self {
        Self {
            feature_columns: &[],
            signs,
            flat: Some(FlatBatch {
                features,
                batch_size,
                feature_count,
            }),
        }
    }

    pub fn feature_count(&self) -> usize {
        self.flat
            .map(|flat| flat.feature_count)
            .unwrap_or(self.feature_columns.len())
    }

    pub fn column(&self, index: usize) -> Option<&[bool]> {
        if let Some(flat) = self.flat {
            if index >= flat.feature_count {
                return None;
            }
            let start = index * flat.batch_size;
            Some(&flat.features[start..start + flat.batch_size])
        } else {
            self.feature_columns.get(index).copied()
        }
    }
}
