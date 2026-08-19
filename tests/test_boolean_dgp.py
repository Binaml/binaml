from __future__ import annotations

import numpy as np
import pytest
from binaml.environments.boolean_dgp import (
    BooleanDgpConfig,
    ConjunctionTerm,
    FunctionSpec,
    sample_function,
)


def test_conjunction_term_rejects_duplicate_indices() -> None:
    with pytest.raises(ValueError, match="distinct"):
        ConjunctionTerm((0, 0), (False, False))


def test_function_spec_rejects_legacy_family_dict() -> None:
    with pytest.raises(ValueError, match="legacy"):
        FunctionSpec.from_dict({"constant": False, "terms": [], "family": "truth_table"})


def test_function_spec_rejects_legacy_flat_formula_dict() -> None:
    with pytest.raises(ValueError, match="legacy flat"):
        FunctionSpec.from_dict({"feature_indices": [0], "negated": [False]})


def test_sample_function_respects_bounds() -> None:
    rng = np.random.default_rng(0)
    config = BooleanDgpConfig(
        n_features=8,
        min_n_terms=1,
        max_n_terms=10,
        min_term_degree=1,
        max_term_degree=7,
    )
    for _ in range(20):
        function = sample_function(rng, config)
        assert 1 <= len(function.terms) <= 10
        for term in function.terms:
            assert 1 <= len(term.feature_indices) <= 7
            assert len(set(term.feature_indices)) == len(term.feature_indices)


def test_negated_literal_flips_output() -> None:
    term = ConjunctionTerm((1,), (True,))
    assert term.evaluate(np.array([0, 0], dtype=np.uint8)) == 1
    assert term.evaluate(np.array([0, 1], dtype=np.uint8)) == 0


def test_anf_xor_and_constant() -> None:
    function = FunctionSpec(
        True,
        (
            ConjunctionTerm((0, 1), (False, False)),
            ConjunctionTerm((0,), (False,)),
        ),
    )
    assert function.evaluate(np.array([1, 1, 0], dtype=np.uint8)) == 1
    assert function.evaluate(np.array([1, 0, 0], dtype=np.uint8)) == 0
    assert function.evaluate(np.array([0, 0, 0], dtype=np.uint8)) == 1
