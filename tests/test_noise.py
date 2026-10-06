import math

import pytest
import torch

from noise import (ConstantSchedule, CosineSchedule, ExponentialSchedule,
                   LinearSchedule, PIController, js_div_per_example,
                   normalized_attention_entropy)


def make_controller(**kwargs):
    defaults = dict(signal = 'jsd', target = 0.05, kp = 1.0, ki = 0.001, ema = 0.0, p_min = 0.05, p_max = 0.5, p_init = 0.2)
    return PIController(**{**defaults, **kwargs})


def test_constant_schedule():
    schedule = ConstantSchedule(p = 0.2)
    assert schedule.value(0) == schedule.value(10**6) == 0.2


@pytest.mark.parametrize("schedule", [
    ExponentialSchedule(p0 = 0.5, p_final = 0.2, gamma = 1.0, steps_per_epoch = 100),
    CosineSchedule(p0 = 0.5, p_final = 0.2, total_steps = 500),
    LinearSchedule(p0 = 0.5, p_final = 0.2, total_steps = 500),
])
def test_open_loop_schedules_decay_from_p0_towards_p_final(schedule):
    values = [schedule.value(step) for step in range(0, 501, 50)]
    assert values[0] == pytest.approx(0.5)
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert all(0.2 - 1e-9 <= value <= 0.5 + 1e-9 for value in values)


def test_cosine_and_linear_reach_p_final_and_stay_there():
    for schedule in (CosineSchedule(0.5, 0.2, 500), LinearSchedule(0.5, 0.2, 500)):
        assert schedule.value(500) == pytest.approx(0.2)
        assert schedule.value(10**6) == pytest.approx(0.2)
        assert schedule.value(250) == pytest.approx(0.35)


def test_exponential_decay_rate_is_per_epoch():
    # same realized noise at the end of an epoch regardless of its length
    short = ExponentialSchedule(p0 = 0.5, p_final = 0.2, gamma = 1.0, steps_per_epoch = 10)
    long = ExponentialSchedule(p0 = 0.5, p_final = 0.2, gamma = 1.0, steps_per_epoch = 2500)
    assert short.value(10) == pytest.approx(long.value(2500)) == pytest.approx(0.2 + 0.3 * math.exp(-1))
    # after 5 epochs the noise is still measurably above its final value
    assert long.value(5 * 2500) - 0.2 > 1e-3


def test_controller_starts_at_p_init_and_holds_it_at_zero_error():
    controller = make_controller()
    assert controller.value(0) == pytest.approx(0.2)
    for _ in range(100):
        controller.update(0.05)
    assert controller.value(0) == pytest.approx(0.2)


def test_controller_raises_noise_when_predictors_disagree():
    controller = make_controller()
    for _ in range(50):
        controller.update(0.3)
    assert controller.value(0) > 0.2
    high = controller.value(0)
    for _ in range(500):
        controller.update(0.0)
    assert controller.value(0) < high


def test_controller_raises_noise_when_attention_collapses():
    collapsed = make_controller(signal = 'entropy', target = 0.8)
    spread = make_controller(signal = 'entropy', target = 0.8)
    for _ in range(50):
        collapsed.update(0.3)
        spread.update(0.95)
    assert collapsed.value(0) > 0.2 > spread.value(0)


def test_controller_stays_in_range_and_does_not_wind_up():
    controller = make_controller()
    for _ in range(100000):
        controller.update(1.0)
    assert controller.value(0) == pytest.approx(0.5)
    # after a long saturation the noise leaves the ceiling within the time
    # the integral term needs to cover the range, not 100000 steps later
    steps = 0
    while controller.value(0) > 0.05 + 1e-9:
        controller.update(0.0)
        steps += 1
        assert steps < 20000
    for _ in range(2000):
        controller.update(0.0)
    assert controller.value(0) == pytest.approx(0.05)
    assert controller.integral == 0.0


def test_controller_smooths_the_signal():
    controller = make_controller(ema = 0.9, kp = 1.0, ki = 0.0, p_init = 0.05)
    controller.update(0.05)
    controller.update(1.0)
    # one outlier moves the average by 10% of its size
    assert controller.signal_ema == pytest.approx(0.145)
    assert controller.value(0) == pytest.approx(0.05 + 0.095)


def test_probe_only_needed_for_jsd():
    assert make_controller(probe_every = 10).needs_probe(20)
    assert not make_controller(probe_every = 10).needs_probe(21)
    assert not make_controller(signal = 'entropy', target = 0.8).needs_probe(20)


def test_js_div_is_normalized():
    same = torch.tensor([[0.3, 0.7]])
    assert js_div_per_example(same, same).item() == pytest.approx(0.0, abs = 1e-6)
    opposite = js_div_per_example(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
    assert opposite.item() == pytest.approx(1.0, abs = 1e-4)
    mixed = js_div_per_example(torch.tensor([[0.9, 0.1], [0.5, 0.5]]), torch.tensor([[0.6, 0.4], [0.5, 0.5]]))
    assert mixed.shape == (2,) and mixed[0] > mixed[1]


def test_attention_entropy_is_normalized_over_real_tokens():
    uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25, 0.0, 0.0]]).unsqueeze(-1)
    peaked = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).unsqueeze(-1)
    selectable = torch.tensor([[True, True, True, True, False, False]])
    assert normalized_attention_entropy(uniform, selectable).item() == pytest.approx(1.0)
    assert normalized_attention_entropy(peaked, selectable).item() == pytest.approx(0.0)
    # without the mask, padding counts as positions the attention avoids
    assert normalized_attention_entropy(uniform).item() < 1.0
