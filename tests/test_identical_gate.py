"""identical_outputs_gate: the JSON-ready form of the identical-outputs gate, run on the tiny float64 model."""

import json

from engine_helpers import make_models
from minivllm.bench.gpu_checks import identical_outputs_gate
from minivllm.bench.jobs import build_jobs
from minivllm.bench.workload import make_workload


def test_gate_record_passes_and_serialises_to_json():
    ref, paged = make_models()
    requests = make_workload(rate=5.0, num_requests=6, seed=4, prompt_mu=2.0, prompt_sigma=0.5,
                             output_mu=1.5, output_sigma=0.4, shared_prefix_len=8, max_len=20)
    jobs = build_jobs(requests, vocab_size=50, seed=4)
    record = identical_outputs_gate(ref, paged, jobs, max_batch=2, token_budget=24, num_blocks=32)
    assert record["passed"], record
    assert record["requests"] == 6 and record["output_tokens"] > 0
    assert set(record["checks"]) == {"lengths_match_output_len", "static_equals_sequential",
                                     "engine_equals_sequential", "engine_per_sequence_equals_sequential"}
    json.dumps(record)
