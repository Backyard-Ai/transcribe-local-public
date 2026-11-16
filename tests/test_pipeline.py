import socket
import urllib.request

import pytest

from transcribe_local import safety
from transcribe_local.pipeline import TimeLeft, free_stem, load_job, save_job


@pytest.fixture
def blocked_network(monkeypatch):
    # Registering the originals makes monkeypatch put them back after each test.
    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    monkeypatch.setattr(socket.socket, "connect_ex", socket.socket.connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", socket.getaddrinfo)
    safety.block_network()


def test_free_stem_keeps_name_when_unused(tmp_path):
    assert free_stem(tmp_path, "interview") == "interview"


def test_free_stem_never_overwrites_an_earlier_transcript(tmp_path):
    (tmp_path / "interview.docx").touch()
    assert free_stem(tmp_path, "interview") == "interview (2)"
    (tmp_path / "interview (2)_review.csv").touch()
    assert free_stem(tmp_path, "interview") == "interview (3)"


def test_job_record_round_trip(tmp_path):
    path = tmp_path / "job.json"
    save_job(path, {"complete": True})
    assert load_job(path, {"complete": False}) == {"complete": True}
    assert not (tmp_path / "job.part").exists()


@pytest.mark.parametrize("damage", ['{"complete": tr', "", "�"])
def test_damaged_job_record_starts_fresh_instead_of_failing(tmp_path, damage):
    path = tmp_path / "job.json"
    path.write_text(damage, encoding="utf-8")
    assert load_job(path, {"complete": False}) == {"complete": False}


def test_missing_job_record_starts_fresh(tmp_path):
    assert load_job(tmp_path / "none.json", {"complete": False}) == {"complete": False}


def make_time_left(*times):
    clock = iter(times)
    return TimeLeft(clock=lambda: next(clock))


def test_time_left_ignores_setup_before_the_first_segment():
    # Setup took 500 seconds, then 100 seconds of audio took 10 seconds, so 10x speed.
    time_left = make_time_left(500.0, 510.0)
    assert time_left.minutes(100.0, 1000.0) is None
    assert time_left.minutes(200.0, 1000.0) == pytest.approx(800 / 10 / 60)


def test_time_left_does_not_count_resumed_audio_as_done_this_run():
    time_left = make_time_left(0.0, 10.0)
    assert time_left.minutes(7200.0, 21600.0) is None
    assert time_left.minutes(7300.0, 21600.0) == pytest.approx(14300 / 10 / 60)


def test_time_left_never_goes_negative():
    time_left = make_time_left(0.0, 10.0)
    time_left.minutes(50.0, 100.0)
    assert time_left.minutes(120.0, 100.0) == 0.0


def test_blocked_network_refuses_connections(blocked_network):
    with pytest.raises(safety.NetworkBlockedError):
        socket.socket().connect(("8.8.8.8", 53))
    with pytest.raises(safety.NetworkBlockedError):
        socket.socket().connect_ex(("8.8.8.8", 53))


def test_blocked_network_refuses_lookups(blocked_network):
    with pytest.raises(safety.NetworkBlockedError):
        socket.getaddrinfo("example.com", 443)


def test_blocked_network_stops_a_real_web_request(blocked_network):
    with pytest.raises(OSError):
        urllib.request.urlopen("https://example.com", timeout=5)


def test_blocked_network_still_allows_this_machine(blocked_network):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.socket()
    client.connect(server.getsockname())
    client.close()
    server.close()


def test_blocking_twice_is_harmless(blocked_network):
    safety.block_network()
    with pytest.raises(safety.NetworkBlockedError):
        socket.socket().connect(("8.8.8.8", 53))


def test_time_left_starts_measuring_again_after_a_speed_change():
    # Fast on the card at first, then the processor takes over at a tenth of the speed.
    time_left = make_time_left(0.0, 10.0, 20.0, 30.0)
    time_left.minutes(0.0, 1000.0)
    assert time_left.minutes(100.0, 1000.0) == pytest.approx(900 / 10 / 60)
    time_left.restart()
    assert time_left.minutes(100.0, 1000.0) is None
    assert time_left.minutes(110.0, 1000.0) == pytest.approx(890 / 1 / 60)
