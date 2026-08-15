use crate::boolean_circuit::evaluate_truth_table;

/// Immutable compact boolean function for ensemble evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionGraph {
    pub(crate) source_indices: Vec<usize>,
    pub(crate) nodes: Vec<CompactNode>,
    pub(crate) output: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CompactNode {
    Source(usize),
    Constant(bool),
    Composed {
        first: usize,
        second: usize,
        truth_table: u8,
    },
}

impl FunctionGraph {
    #[must_use]
    pub fn evaluate(&self, features: &[bool]) -> bool {
        let mut values = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let value = match *node {
                CompactNode::Source(source_index) => *features
                    .get(self.source_indices[source_index])
                    .unwrap_or(&false),
                CompactNode::Constant(value) => value,
                CompactNode::Composed {
                    first,
                    second,
                    truth_table,
                } => evaluate_truth_table(truth_table, values[first], values[second]),
            };
            values.push(value);
        }
        values[self.output]
    }

    #[cfg(test)]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[cfg(test)]
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.source_indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{CompactNode, FunctionGraph};

    const AND: u8 = 0b1000;
    const OR: u8 = 0b1110;
    const XOR: u8 = 0b0110;

    fn graph(source_indices: Vec<usize>, nodes: Vec<CompactNode>, output: usize) -> FunctionGraph {
        FunctionGraph {
            source_indices,
            nodes,
            output,
        }
    }

    #[test]
    fn evaluates_source_node() {
        let function = graph(vec![1], vec![CompactNode::Source(0)], 0);
        assert!(function.evaluate(&[false, true]));
        assert!(!function.evaluate(&[true, false]));
    }

    #[test]
    fn evaluates_constant_nodes() {
        let function = graph(
            vec![],
            vec![CompactNode::Constant(true), CompactNode::Constant(false)],
            0,
        );
        assert!(function.evaluate(&[]));
        assert!(!graph(vec![], vec![CompactNode::Constant(false)], 0).evaluate(&[]));
    }

    #[test]
    fn evaluates_composed_gates() {
        let function = graph(
            vec![0, 1],
            vec![
                CompactNode::Source(0),
                CompactNode::Source(1),
                CompactNode::Composed {
                    first: 0,
                    second: 1,
                    truth_table: AND,
                },
            ],
            2,
        );
        assert!(function.evaluate(&[true, true]));
        assert!(!function.evaluate(&[true, false]));
        assert!(!function.evaluate(&[false, true]));
        assert!(!function.evaluate(&[false, false]));

        let or_gate = graph(
            vec![0, 1],
            vec![
                CompactNode::Source(0),
                CompactNode::Source(1),
                CompactNode::Composed {
                    first: 0,
                    second: 1,
                    truth_table: OR,
                },
            ],
            2,
        );
        assert!(or_gate.evaluate(&[false, true]));
        assert!(or_gate.evaluate(&[true, false]));
        assert!(or_gate.evaluate(&[true, true]));
        assert!(!or_gate.evaluate(&[false, false]));

        let xor_gate = graph(
            vec![0, 1],
            vec![
                CompactNode::Source(0),
                CompactNode::Source(1),
                CompactNode::Composed {
                    first: 0,
                    second: 1,
                    truth_table: XOR,
                },
            ],
            2,
        );
        assert!(xor_gate.evaluate(&[false, true]));
        assert!(xor_gate.evaluate(&[true, false]));
        assert!(!xor_gate.evaluate(&[true, true]));
        assert!(!xor_gate.evaluate(&[false, false]));
    }

    #[test]
    fn missing_feature_defaults_to_false() {
        let function = graph(vec![3], vec![CompactNode::Source(0)], 0);
        assert!(!function.evaluate(&[true, false]));
    }

    #[test]
    fn output_selects_requested_node() {
        let function = graph(
            vec![],
            vec![CompactNode::Constant(true), CompactNode::Constant(false)],
            1,
        );
        assert!(!function.evaluate(&[]));
    }

    #[test]
    fn counts_nodes_and_sources() {
        let function = graph(
            vec![0, 2],
            vec![
                CompactNode::Source(0),
                CompactNode::Source(1),
                CompactNode::Composed {
                    first: 0,
                    second: 1,
                    truth_table: AND,
                },
            ],
            2,
        );
        assert_eq!(function.node_count(), 3);
        assert_eq!(function.source_count(), 2);
    }
}
