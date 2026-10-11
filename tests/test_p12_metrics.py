import pytest

from minivllm.bench.metrics import RequestLog, e2e, itl, percentile, summarize, tpot, ttft


def log_a() -> RequestLog:
    return RequestLog(request_id=0, arrival_time=0.0, dequeue_time=1.0, token_times=[2.0, 3.0, 5.0], finish_time=5.0)


def test_hand_built_log_known_answers():
    a = log_a()
    assert ttft(a) == pytest.approx(2.0)
    assert tpot(a) == pytest.approx(1.5)
    assert itl(a) == pytest.approx([1.0, 2.0])
    assert e2e(a) == pytest.approx(5.0)


def test_ttft_measured_from_scheduled_arrival_not_dequeue():
    a = log_a()
    assert ttft(a) != pytest.approx(a.token_times[0] - a.dequeue_time)


def test_nearest_rank_percentiles():
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 90) == 90.0
    assert percentile(values, 99) == 99.0


def test_tpot_with_one_token_raises():
    one = RequestLog(request_id=1, arrival_time=0.0, dequeue_time=0.0, token_times=[1.0], finish_time=1.0)
    with pytest.raises(ValueError):
        tpot(one)


def test_summarize_throughput_and_goodput():
    # log A misses the TPOT SLO; B misses the TTFT SLO; C meets both
    b = RequestLog(request_id=1, arrival_time=0.0, dequeue_time=0.0, token_times=[3.0, 3.05], finish_time=3.05)
    c = RequestLog(request_id=2, arrival_time=0.0, dequeue_time=0.0, token_times=[1.0, 1.05, 1.1], finish_time=1.1)
    s = summarize([log_a(), b, c])
    # tokens 3 + 2 + 3 = 8 over a 5.0 s run
    assert s["throughput"] == pytest.approx(8 / 5.0)
    # only C meets both SLOs: 1 request over 5.0 s
    assert s["goodput"] == pytest.approx(1 / 5.0)


# extra edge case
def test_itl_has_one_fewer_entry_than_tokens():
    assert len(itl(log_a())) == len(log_a().token_times) - 1


# extra edge case
def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


# extra edge case
def test_single_token_request_counts_toward_ttft_and_e2e():
    one = RequestLog(request_id=1, arrival_time=0.0, dequeue_time=0.0, token_times=[0.5], finish_time=0.5)
    assert ttft(one) == pytest.approx(0.5)
    assert e2e(one) == pytest.approx(0.5)


# review: nearest-rank p99 of 10 values is the 10th value, not the 9th
def test_nearest_rank_p99_of_ten_values_is_the_max():
    assert percentile([float(v) for v in range(1, 11)], 99) == 10.0


# review: summarize must not crash when no request has a second output token
def test_summarize_with_only_single_token_requests_does_not_raise():
    one = RequestLog(request_id=1, arrival_time=0.0, dequeue_time=0.0, token_times=[0.5], finish_time=0.5)
    summarize([one])
