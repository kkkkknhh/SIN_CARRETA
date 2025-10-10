"""Determinism guards for critical modules."""

import random

import numpy as np

from dag_validation import DAGValidator


def test_validator_is_deterministic() -> None:
    random.seed(0)
    np.random.seed(0)
    v1 = DAGValidator().validate(sample="S")
    random.seed(0)
    np.random.seed(0)
    v2 = DAGValidator().validate(sample="S")
    assert v1 == v2, "Non-deterministic DAG validation result"
