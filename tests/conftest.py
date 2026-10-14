import pytest
import torch


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
        ),
    ]
)
def device(request) -> str:
    """Run an engine test on CPU, and again on CUDA when a GPU build of torch is installed."""
    return request.param
