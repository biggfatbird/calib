"""Thresholds are chosen without test labels."""

import inspect

import numpy as np
import pytest

from wdbc_calib.thresholds import select_cost_threshold


def test_signature_has_no_test_labels():
    names = list(inspect.signature(select_cost_threshold).parameters)
    assert names == ["y_oof", "p_oof", "cost_fn", "cost_fp"]
    joined = " ".join(names)
    assert "test" not in joined


def test_known_cost_minimum():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.2, 0.8, 0.3, 0.6, 0.7])
    # t = 0.3 predicts one false positive and no false negatives. Cost = 1.
    # t = 0.5 predicts one false positive and one false negative. Cost = 6 when FN costs 5.
    threshold, cost = select_cost_threshold(y, p, cost_fn=5, cost_fp=1)
    assert threshold == pytest.approx(0.3)
    assert cost == pytest.approx(1)


def test_tie_breaks_toward_one_half_then_lower():
    y = np.array([0, 1])
    p = np.array([0.2, 0.8])
    # Both t=0.2 (FP+TP, cost 1) and a threshold that predicts nothing are worse or equal.
    # t=0.8: FN=0? p>=0.8 -> second positive, first negative: FP=0 FN=0 cost 0.
    # p>=0.8 is perfect. Unique.
    threshold, cost = select_cost_threshold(y, p, cost_fn=1, cost_fp=1)
    assert cost == 0
    # t=0.5 and t=0.8 both have zero cost; the rule keeps the one closer to 0.5.
    assert threshold == pytest.approx(0.5)

    # Two thresholds with the same predictions are not both candidates unless they are stored values.
    # Construct equal cost at 0.2 and 0.8 by costs that do not distinguish, then check the rule
    # on a case with two different candidate thresholds and identical cost.
    y2 = np.array([1, 1])
    p2 = np.array([0.2, 0.9])
    # Every threshold in (0, 0.2] has FN=0. Candidates 0.2 and... 0.0 predicts both positive FN=0.
    # cost is 0 for t=0 and t=0.2. Closer to 0.5 is 0.2.
    threshold, cost = select_cost_threshold(y2, p2, cost_fn=1, cost_fp=1)
    assert cost == 0
    assert threshold == pytest.approx(0.2)
