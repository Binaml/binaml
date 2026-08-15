use crate::function_builder::{BuildNodeId, EphemeralGraph, EphemeralNode};
use crate::function_graph::{CompactNode, FunctionGraph};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactError {
    InvalidOutput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Slot {
    Source(usize),
    Constant(bool),
    Composed {
        first: usize,
        second: usize,
        truth_table: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SimplifyResult {
    Keep,
    Constant(bool),
    Alias(usize),
    Gate {
        first: usize,
        second: usize,
        truth_table: u8,
    },
}

pub fn compact(graph: EphemeralGraph, output: BuildNodeId) -> Result<FunctionGraph, CompactError> {
    if output.0 >= graph.nodes.len() {
        return Err(CompactError::InvalidOutput);
    }

    let mut slots: Vec<Slot> = graph
        .nodes
        .iter()
        .map(|node| match node {
            EphemeralNode::Source { input_index } => Slot::Source(*input_index),
            EphemeralNode::Composed {
                first,
                second,
                truth_table,
            } => Slot::Composed {
                first: first.0,
                second: second.0,
                truth_table: *truth_table,
            },
        })
        .collect();
    let mut aliases = vec![None; slots.len()];

    loop {
        let mut changed = false;
        for index in (0..slots.len()).rev() {
            if aliases[index].is_some() {
                continue;
            }
            if !matches!(slots[index], Slot::Composed { .. }) {
                continue;
            }
            let Slot::Composed {
                first,
                second,
                truth_table,
            } = slots[index].clone()
            else {
                continue;
            };
            match simplify(
                resolve_id(first, &aliases),
                resolve_id(second, &aliases),
                truth_table,
            ) {
                SimplifyResult::Keep => {}
                SimplifyResult::Constant(value) => {
                    slots[index] = Slot::Constant(value);
                    changed = true;
                }
                SimplifyResult::Alias(target) => {
                    let target = resolve_id(target, &aliases);
                    if aliases[index] != Some(target) {
                        aliases[index] = Some(target);
                        changed = true;
                    }
                }
                SimplifyResult::Gate {
                    first,
                    second,
                    truth_table,
                } => {
                    let first = resolve_id(first, &aliases);
                    let second = resolve_id(second, &aliases);
                    let next = Slot::Composed {
                        first,
                        second,
                        truth_table,
                    };
                    if slots[index] != next {
                        slots[index] = next;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let output = resolve_id(output.0, &aliases);
    let reachable = backward_reach(&slots, &aliases, output);
    let order: Vec<usize> = (0..slots.len())
        .filter(|index| reachable.contains(index))
        .collect();
    let mut old_to_new = HashMap::new();
    let mut nodes = Vec::new();
    let mut source_indices = Vec::new();
    let mut source_map = HashMap::new();

    for &old_index in &order {
        let resolved = resolve_id(old_index, &aliases);
        let new_index = nodes.len();
        old_to_new.insert(old_index, new_index);
        match &slots[resolved] {
            Slot::Source(input_index) => {
                let compact_source = *source_map.entry(*input_index).or_insert_with(|| {
                    let index = source_indices.len();
                    source_indices.push(*input_index);
                    index
                });
                nodes.push(CompactNode::Source(compact_source));
            }
            Slot::Constant(value) => nodes.push(CompactNode::Constant(*value)),
            Slot::Composed {
                first,
                second,
                truth_table,
            } => {
                let first = *old_to_new
                    .get(&resolve_id(*first, &aliases))
                    .ok_or(CompactError::InvalidOutput)?;
                let second = *old_to_new
                    .get(&resolve_id(*second, &aliases))
                    .ok_or(CompactError::InvalidOutput)?;
                nodes.push(CompactNode::Composed {
                    first,
                    second,
                    truth_table: *truth_table,
                });
            }
        }
    }

    let output = *old_to_new.get(&output).ok_or(CompactError::InvalidOutput)?;

    Ok(FunctionGraph {
        source_indices,
        nodes,
        output,
    })
}

fn resolve_id(index: usize, aliases: &[Option<usize>]) -> usize {
    let mut current = index;
    while let Some(next) = aliases[current] {
        current = next;
    }
    current
}

fn simplify(first: usize, second: usize, truth_table: u8) -> SimplifyResult {
    match (first, second, truth_table) {
        (_, _, 0b0000) => SimplifyResult::Constant(false),
        (_, _, 0b1111) => SimplifyResult::Constant(true),
        (a, b, 0b1010) if a == b => SimplifyResult::Alias(a),
        (a, b, 0b1100) if a == b => SimplifyResult::Alias(a),
        (a, b, 0b0011) if a == b => SimplifyResult::Gate {
            first: a,
            second: a,
            truth_table: 0b0011,
        },
        (a, b, 0b0101) if a == b => SimplifyResult::Constant(false),
        (a, _, 0b1010) => SimplifyResult::Alias(a),
        (_, b, 0b1100) => SimplifyResult::Alias(b),
        (a, _, 0b0011) => SimplifyResult::Gate {
            first: a,
            second: a,
            truth_table: 0b0011,
        },
        (_, b, 0b0101) => SimplifyResult::Gate {
            first: b,
            second: b,
            truth_table: 0b0011,
        },
        _ => SimplifyResult::Keep,
    }
}

fn backward_reach(slots: &[Slot], aliases: &[Option<usize>], output: usize) -> HashSet<usize> {
    let mut reachable = HashSet::new();
    let mut stack = vec![output];
    while let Some(index) = stack.pop() {
        let resolved = resolve_id(index, aliases);
        if !reachable.insert(resolved) {
            continue;
        }
        match &slots[resolved] {
            Slot::Source(_) | Slot::Constant(_) => {}
            Slot::Composed { first, second, .. } => {
                stack.push(*first);
                stack.push(*second);
            }
        }
    }
    reachable
}

#[cfg(test)]
mod tests {
    use super::compact;
    use crate::function_builder::{
        BuildNodeId, EphemeralGraph, EphemeralNode, FunctionBuildConfig, FunctionBuilder,
    };
    use crate::SignBatch;

    #[test]
    fn compaction_drops_unused_sources() {
        let first = [false, true, false, true];
        let second = [true, true, true, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, true, false, true];
        let config = FunctionBuildConfig {
            batch_size: 4,
            parent_top_k: 2,
            max_layers: 1,
        };
        let (graph, output) = FunctionBuilder::build(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config,
        )
        .unwrap();
        let compacted = compact(graph, output).unwrap();
        assert!(compacted.source_count() <= 2);
    }

    #[test]
    fn compaction_folds_truth_tables() {
        let graph = EphemeralGraph {
            nodes: vec![
                EphemeralNode::Source { input_index: 0 },
                EphemeralNode::Composed {
                    first: BuildNodeId(0),
                    second: BuildNodeId(0),
                    truth_table: 0b1010,
                },
            ],
            layers: vec![vec![BuildNodeId(0)], vec![BuildNodeId(1)]],
        };
        let compacted = compact(graph, BuildNodeId(1)).unwrap();
        assert_eq!(compacted.node_count(), 1);
    }

    #[test]
    fn topological_order_valid() {
        let first = [false, true, false, true];
        let second = [false, false, true, true];
        let third = [true, false, true, false];
        let columns = [&first[..], &second[..], &third[..]];
        let signs = [false, true, true, false];
        let config = FunctionBuildConfig {
            batch_size: 4,
            parent_top_k: 3,
            max_layers: 2,
        };
        let (graph, output) = FunctionBuilder::build(
            SignBatch {
                feature_columns: &columns,
                signs: &signs,
            },
            config,
        )
        .unwrap();
        let compacted = compact(graph, output).unwrap();
        for (index, node) in compacted.nodes.iter().enumerate() {
            if let crate::function_graph::CompactNode::Composed { first, second, .. } = node {
                assert!(first < &index);
                assert!(second < &index);
            }
        }
    }
}
