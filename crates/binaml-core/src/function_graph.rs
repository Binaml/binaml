use crate::boolean_circuit::evaluate_truth_table;

/// Immutable compact boolean function for ensemble evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionGraph {
    pub(crate) source_indices: Box<[usize]>,
    pub(crate) nodes: Box<[CompactNode]>,
    pub(crate) n_sources: u16,
    pub(crate) n_nodes: u16,
    pub(crate) output: u16,
    pub(crate) invert_output: bool,
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
    pub(crate) fn empty(source_capacity: usize, node_capacity: usize) -> Self {
        Self {
            source_indices: vec![0; source_capacity].into_boxed_slice(),
            nodes: vec![CompactNode::Constant(false); node_capacity].into_boxed_slice(),
            n_sources: 0,
            n_nodes: 0,
            output: 0,
            invert_output: false,
        }
    }

    pub(crate) fn reset_from_parts(
        &mut self,
        source_indices: &[usize],
        nodes: &[CompactNode],
        output: usize,
        invert_output: bool,
    ) {
        self.source_indices[..source_indices.len()].copy_from_slice(source_indices);
        self.nodes[..nodes.len()].clone_from_slice(nodes);
        self.n_sources = u16::try_from(source_indices.len()).expect("source count fits in u16");
        self.n_nodes = u16::try_from(nodes.len()).expect("node count fits in u16");
        self.output = u16::try_from(output).expect("output index fits in u16");
        self.invert_output = invert_output;
    }

    #[must_use]
    pub fn evaluate(&self, features: &[bool]) -> bool {
        self.evaluate_with_scratch(features, &mut vec![false; self.nodes.len()])
    }

    pub(crate) fn evaluate_with_scratch(&self, features: &[bool], scratch: &mut [bool]) -> bool {
        let n_nodes = self.n_nodes as usize;
        for index in 0..n_nodes {
            let value = match self.nodes[index] {
                CompactNode::Source(source_index) => *features
                    .get(self.source_indices[source_index])
                    .unwrap_or(&false),
                CompactNode::Constant(value) => value,
                CompactNode::Composed {
                    first,
                    second,
                    truth_table,
                } => evaluate_truth_table(truth_table, scratch[first], scratch[second]),
            };
            scratch[index] = value;
        }
        let value = scratch[self.output as usize];
        if self.invert_output {
            !value
        } else {
            value
        }
    }

    #[cfg(test)]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.n_nodes as usize
    }

    #[cfg(test)]
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.n_sources as usize
    }
}

#[cfg(test)]
mod tests {
    use super::{CompactNode, FunctionGraph};

    const AND: u8 = 0b1000;
    const OR: u8 = 0b1110;
    const XOR: u8 = 0b0110;

    fn graph(source_indices: Vec<usize>, nodes: Vec<CompactNode>, output: usize) -> FunctionGraph {
        let mut function = FunctionGraph::empty(source_indices.len().max(1), nodes.len().max(1));
        function.reset_from_parts(&source_indices, &nodes, output, false);
        function
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
