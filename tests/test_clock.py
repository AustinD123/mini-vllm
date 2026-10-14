import pytest

from minivllm.clock import FakeClock


def test_fake_clock_moves_only_when_told():
    clock = FakeClock(start=1.0)
    assert clock.now() == 1.0
    clock.charge(0.25)
    assert clock.now() == pytest.approx(1.25)
    clock.advance_to(5.0)
    assert clock.now() == 5.0
    clock.advance_to(2.0)  # already past: no-op
    assert clock.now() == 5.0


def test_fake_clock_rejects_negative_charge():
    with pytest.raises(ValueError):
        FakeClock().charge(-0.1)
