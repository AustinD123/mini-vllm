import numpy as np
import pytest

from minivllm.bench.fit_lengths import fit_lognormal


def test_recovers_parameters_from_a_known_lognormal():
    rng = np.random.default_rng(0)
    sample = rng.lognormal(mean=4.2, sigma=0.9, size=20000)
    mu, sigma = fit_lognormal(np.round(sample))
    assert mu == pytest.approx(4.2, abs=0.02)
    assert sigma == pytest.approx(0.9, abs=0.02)


def test_constant_lengths_give_zero_sigma():
    mu, sigma = fit_lognormal([8, 8, 8])
    assert mu == pytest.approx(np.log(8))
    assert sigma == pytest.approx(0.0)


# extra edge case
def test_empty_input_raises():
    with pytest.raises(ValueError):
        fit_lognormal([])


# extra edge case
def test_zero_length_raises():
    with pytest.raises(ValueError):
        fit_lognormal([3, 0, 5])
