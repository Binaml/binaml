use crate::function_build_common::{BuildNodeId, EphemeralNode};
use crate::function_graph::{CompactNode, FunctionGraph};
use crate::workspace::BuildWorkspace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactError {
    InvalidOutput,
    ExpertTooLarge,
}

pub(crate) const NO_ALIAS: u32 = u32::MAX;
pub(crate) const NO_MAP: u16 = u16::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompactSlotKind {
    Source,
    Constant,
    Composed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompactSlot {
    pub kind: CompactSlotKind,
    pub a: usize,
    pub b: usize,
    pub truth_table: u8,
}

impl Default for CompactSlot {
    fn default() -> Self {
        Self {
            kind: CompactSlotKind::Source,
            a: 0,
            b: 0,
            truth_table: 0,
        }
    }
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

pub(crate) fn compact_build_workspace_into(
    workspace: &mut BuildWorkspace,
    output: BuildNodeId,
    invert_output: bool,
    max_expert_nodes: usize,
    graph: &mut FunctionGraph,
) -> Result<(), CompactError> {
    if output.0 >= workspace.node_len {
        return Err(CompactError::InvalidOutput);
    }

    for index in 0..workspace.node_len {
        workspace.compact_aliases[index] = NO_ALIAS;
        workspace.compact_slots[index] = match workspace.nodes[index] {
            EphemeralNode::Source { input_index } => CompactSlot {
                kind: CompactSlotKind::Source,
                a: input_index,
                b: 0,
                truth_table: 0,
            },
            EphemeralNode::Composed {
                first,
                second,
                truth_table,
            } => CompactSlot {
                kind: CompactSlotKind::Composed,
                a: first.0,
                b: second.0,
                truth_table,
            },
        };
    }

    loop {
        let mut changed = false;
        for index in (0..workspace.node_len).rev() {
            if workspace.compact_aliases[index] != NO_ALIAS {
                continue;
            }
            let slot = workspace.compact_slots[index];
            if slot.kind != CompactSlotKind::Composed {
                continue;
            }
            match simplify(
                resolve_alias_u32(slot.a, &workspace.compact_aliases),
                resolve_alias_u32(slot.b, &workspace.compact_aliases),
                slot.truth_table,
            ) {
                SimplifyResult::Keep => {}
                SimplifyResult::Constant(value) => {
                    workspace.compact_slots[index] = CompactSlot {
                        kind: CompactSlotKind::Constant,
                        a: usize::from(value),
                        b: 0,
                        truth_table: 0,
                    };
                    changed = true;
                }
                SimplifyResult::Alias(target) => {
                    let target = resolve_alias_u32(target, &workspace.compact_aliases);
                    if workspace.compact_aliases[index] != target as u32 {
                        workspace.compact_aliases[index] = target as u32;
                        changed = true;
                    }
                }
                SimplifyResult::Gate {
                    first,
                    second,
                    truth_table,
                } => {
                    let first = resolve_alias_u32(first, &workspace.compact_aliases);
                    let second = resolve_alias_u32(second, &workspace.compact_aliases);
                    let next = CompactSlot {
                        kind: CompactSlotKind::Composed,
                        a: first,
                        b: second,
                        truth_table,
                    };
                    if workspace.compact_slots[index] != next {
                        workspace.compact_slots[index] = next;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    let output = resolve_alias_u32(output.0, &workspace.compact_aliases);
    workspace.compact_reachable.fill(false);
    let mut stack_len = 0usize;
    workspace.compact_order[stack_len] = output as u16;
    stack_len += 1;
    while stack_len > 0 {
        stack_len -= 1;
        let index = workspace.compact_order[stack_len] as usize;
        let resolved = resolve_alias_u32(index, &workspace.compact_aliases);
        if workspace.compact_reachable[resolved] {
            continue;
        }
        workspace.compact_reachable[resolved] = true;
        if let CompactSlot {
            kind: CompactSlotKind::Composed,
            a: first,
            b: second,
            ..
        } = workspace.compact_slots[resolved]
        {
            workspace.compact_order[stack_len] = first as u16;
            stack_len += 1;
            workspace.compact_order[stack_len] = second as u16;
            stack_len += 1;
        }
    }

    let mut order_len = 0usize;
    for index in 0..workspace.node_len {
        if workspace.compact_reachable[index] {
            workspace.compact_order[order_len] = index as u16;
            order_len += 1;
        }
    }

    workspace.compact_old_to_new.fill(NO_MAP);
    let mut source_count = 0usize;
    let mut node_count = 0usize;

    for order_index in 0..order_len {
        let old_index = workspace.compact_order[order_index] as usize;
        let resolved = resolve_alias_u32(old_index, &workspace.compact_aliases);
        let new_index = node_count;
        workspace.compact_old_to_new[old_index] = new_index as u16;
        match workspace.compact_slots[resolved] {
            CompactSlot {
                kind: CompactSlotKind::Source,
                a: input_index,
                ..
            } => {
                let compact_source = find_or_insert_source(
                    &mut workspace.compact_sources,
                    &mut source_count,
                    input_index,
                );
                workspace.compact_nodes[new_index] = CompactNode::Source(compact_source);
            }
            CompactSlot {
                kind: CompactSlotKind::Constant,
                a: value,
                ..
            } => {
                workspace.compact_nodes[new_index] = CompactNode::Constant(value != 0);
            }
            CompactSlot {
                kind: CompactSlotKind::Composed,
                a: first,
                b: second,
                truth_table,
            } => {
                let first = workspace
                    .compact_old_to_new
                    .get(resolve_alias_u32(first, &workspace.compact_aliases))
                    .copied()
                    .ok_or(CompactError::InvalidOutput)? as usize;
                let second = workspace
                    .compact_old_to_new
                    .get(resolve_alias_u32(second, &workspace.compact_aliases))
                    .copied()
                    .ok_or(CompactError::InvalidOutput)? as usize;
                workspace.compact_nodes[new_index] = CompactNode::Composed {
                    first,
                    second,
                    truth_table,
                };
            }
        }
        node_count += 1;
    }

    let output = workspace
        .compact_old_to_new
        .get(output)
        .copied()
        .ok_or(CompactError::InvalidOutput)? as usize;

    if node_count > max_expert_nodes {
        return Err(CompactError::ExpertTooLarge);
    }

    graph.reset_from_parts(
        &workspace.compact_sources[..source_count],
        &workspace.compact_nodes[..node_count],
        output,
        invert_output,
    );
    Ok(())
}

fn find_or_insert_source(sources: &mut [usize], source_count: &mut usize, input_index: usize) -> usize {
    for (index, &existing) in sources.iter().enumerate().take(*source_count) {
        if existing == input_index {
            return index;
        }
    }
    let index = *source_count;
    sources[index] = input_index;
    *source_count += 1;
    index
}

fn resolve_alias_u32(mut index: usize, aliases: &[u32]) -> usize {
    while aliases[index] != NO_ALIAS {
        index = aliases[index] as usize;
    }
    index
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

#[cfg(test)]
mod tests {
    use super::compact_build_workspace_into;
    use crate::function_build_common::{
        BuildNodeId, EphemeralNode, DEFAULT_L_PAT, DEFAULT_MAX_EXPERT_NODES,
    };
    use crate::function_builder::{FunctionBuildConfig, FunctionBuilder};
    use crate::function_graph::{CompactNode, FunctionGraph};
    use crate::workspace::{BuildWorkspace, ModelCapacity};
    use crate::SignBatch;

    #[test]
    fn compaction_drops_unused_sources() {
        let first = [false, true, false, true];
        let second = [true, true, true, true];
        let columns = [&first[..], &second[..]];
        let signs = [false, true, false, true];
        let config = FunctionBuildConfig::new(4, 2, 2, DEFAULT_MAX_EXPERT_NODES, DEFAULT_L_PAT);
        let model = FunctionBuilder::build(
            SignBatch::from_columns(&columns, &signs),
            config,
        )
        .unwrap();
        assert!(model.graph.source_count() <= 2);
    }

    #[test]
    fn compaction_folds_truth_tables() {
        let capacity = ModelCapacity::new(1, 4, 2, 1, DEFAULT_MAX_EXPERT_NODES, 0);
        let mut workspace = BuildWorkspace::new(capacity);
        workspace.nodes[0] = EphemeralNode::Source { input_index: 0 };
        workspace.nodes[1] = EphemeralNode::Composed {
            first: BuildNodeId(0),
            second: BuildNodeId(0),
            truth_table: 0b1010,
        };
        workspace.node_len = 2;
        let mut graph = FunctionGraph::empty(1, DEFAULT_MAX_EXPERT_NODES);
        compact_build_workspace_into(
            &mut workspace,
            BuildNodeId(1),
            false,
            DEFAULT_MAX_EXPERT_NODES,
            &mut graph,
        )
        .unwrap();
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn topological_order_valid() {
        let first = [false, true, false, true];
        let second = [false, false, true, true];
        let third = [true, false, true, false];
        let columns = [&first[..], &second[..], &third[..]];
        let signs = [false, true, true, false];
        let config = FunctionBuildConfig::new(4, 3, 3, DEFAULT_MAX_EXPERT_NODES, DEFAULT_L_PAT);
        let model = FunctionBuilder::build(
            SignBatch::from_columns(&columns, &signs),
            config,
        )
        .unwrap();
        for (index, node) in model.graph.nodes().iter().enumerate() {
            if let CompactNode::Composed { first, second, .. } = node {
                assert!(first < &index);
                assert!(second < &index);
            }
        }
    }
}
