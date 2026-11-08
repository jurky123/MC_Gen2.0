import json

import pytest

from mcgen.stability_guard import StabilityGuard, StabilityPolicy


def policy():
    return StabilityPolicy(warmup=0, calibration=4, window=4, patience=2)


def feed(guard, loss, norms):
    result = None
    for norm in norms:
        result = guard.observe(guard.last_step+1, loss, norm)
    return result


def test_isolated_spikes_and_improving_loss_do_not_stop_training():
    guard = StabilityGuard(policy())
    feed(guard, .1, [.2]*4)
    feed(guard, .09, [.2]*4)
    for _ in range(20):
        assert not feed(guard, .08, [.2, .2, .2, 1000])['stop']
    # Large gradients alone do not diagnose deterioration.
    assert not feed(guard, .07, [3.]*4)['stop']


def test_sustained_gradient_and_loss_growth_stops_after_patience():
    guard = StabilityGuard(policy())
    feed(guard, .1, [.2]*4)
    feed(guard, .08, [.2]*4)
    assert not feed(guard, .12, [2.]*4)['stop']
    decision = feed(guard, .13, [2.]*4)
    assert decision['stop'] and decision['reason'] == 'persistent_loss_and_gradient_growth'
    assert decision['clipped_fraction'] == 1.


def test_resume_preserves_calibration_and_partial_window():
    original = StabilityGuard(policy())
    feed(original, .1, [.2]*4)
    feed(original, .08, [.2]*4)
    feed(original, .12, [2.]*4)
    feed(original, .13, [2.]*2)
    restored = StabilityGuard.from_state(json.loads(json.dumps(original.state_dict())))
    assert feed(original, .13, [2.]*2) == feed(restored, .13, [2.]*2)
    assert restored.bad_windows == 2


def test_recovery_resets_patience_and_nonfinite_values_stop_immediately():
    guard = StabilityGuard(policy())
    feed(guard, .1, [.2]*4)
    feed(guard, .08, [.2]*4)
    feed(guard, .12, [2.]*4)
    assert not feed(guard, .08, [.2]*4)['stop']
    assert not feed(guard, .12, [2.]*4)['stop']
    assert guard.observe(guard.last_step+1, float('nan'), 1.)['stop']
    with pytest.raises(ValueError):
        guard.observe(guard.last_step, .1, .2)
