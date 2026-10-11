import pytest

from minivllm.runner.fake import OVERHEAD_MS, step_latency


def test_zero_work_is_overhead_only():
    assert step_latency(0, 0, 0) == pytest.approx(OVERHEAD_MS)


def test_monotonic_in_prefill_tokens():
    assert step_latency(10, 0, 0) < step_latency(11, 0, 0)


def test_monotonic_in_decode_seqs():
    assert step_latency(0, 10, 0) < step_latency(0, 11, 0)


def test_monotonic_in_total_context():
    assert step_latency(0, 0, 1000) < step_latency(0, 0, 1001)


# extra edge case
def test_prefill_term_is_linear():
    base = step_latency(0, 0, 0)
    one = step_latency(100, 0, 0) - base
    two = step_latency(200, 0, 0) - base
    assert two == pytest.approx(2 * one)


# extra edge case
@pytest.mark.parametrize("args", [(-1, 0, 0), (0, -1, 0), (0, 0, -1)])
def test_negative_argument_raises(args):
    with pytest.raises(ValueError):
        step_latency(*args)
