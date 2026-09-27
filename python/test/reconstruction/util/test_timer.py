"""Port of upstream pycolmap util/timer_test.py (Open3D style)."""
import time


def test_timer_init(recon):
    assert recon.util.Timer() is not None


def test_timer_start(recon):
    timer = recon.util.Timer()
    timer.start()


def test_timer_pause(recon):
    timer = recon.util.Timer()
    timer.start()
    timer.pause()


def test_timer_resume(recon):
    timer = recon.util.Timer()
    timer.start()
    timer.pause()
    timer.resume()


def test_timer_reset(recon):
    timer = recon.util.Timer()
    timer.start()
    timer.reset()


def test_timer_elapsed_seconds(recon):
    timer = recon.util.Timer()
    timer.start()
    time.sleep(0.01)
    assert timer.elapsed_seconds() > 0.0
