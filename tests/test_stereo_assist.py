"""Synthetic audio checks; no capture device, game, or audible playback needed."""
import os
import sys
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest

if sys.platform != "win32":
    pytest.skip("Windows capture integration", allow_module_level=True)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from main import AudioRadarApp
from audio_capture import AudioCaptureThread
from mono_output import MonoMixThread
from overlay import OverlayRadar
from PyQt6.QtWidgets import QApplication


def test_stereo_detection_display_and_right_ear_mix():
    app = QApplication.instance() or QApplication([])
    capture = AudioCaptureThread(freq_low=150, freq_high=4000)
    capture.noise_ratio = 0
    events = []
    capture.audio_data_signal.connect(lambda a, i: events.append((a, i)))
    t = np.arange(capture.chunk_frames) / capture.samplerate
    left = np.sin(2 * np.pi * 240 * t)
    right = np.sin(2 * np.pi * 3000 * t)
    mixed = np.column_stack((left, right)) * 0.02
    capture._process_chunk(mixed, False)
    assert any(a < 0 for a, _ in events) and any(a > 0 for a, _ in events)

    # Boost cannot make accepted sounds disappear above the loud cutoff.
    capture.max_amplitude = 0.03
    capture.gain = 50
    events.clear()
    capture._process_chunk(mixed, False)
    assert len(events) == 2
    events.clear()
    capture._process_chunk(mixed * 10, False)
    assert events == []
    capture.max_amplitude = 1
    capture._process_chunk(mixed * 100, False)
    assert events  # 1 means disabled, even for float audio above full scale.

    capture.gain = 1
    capture.noise_ratio = 1.5
    for _ in range(300):
        capture._process_chunk(mixed, False)
    events.clear()
    capture._process_chunk(mixed, False)
    assert not events  # Steady ambience settles below the adaptive threshold.
    capture._process_chunk(mixed * 3, False)
    assert events  # A new transient still registers.
    capture.noise_ratio = 0
    events.clear()
    capture._process_chunk(mixed, False)
    assert events  # Suppression can be disabled for sustained sounds.

    radar = OverlayRadar()
    with patch("overlay.time.monotonic", return_value=10):
        radar.update_audio_data(-30, 0.0051)
        radar.update_audio_data(30, 0.01)
        radar.update_audio_data(0, 0.01)
    assert {b["angle"] for b in radar.blips} == {-90, 0, 90}
    with patch("overlay.time.monotonic", return_value=10.19):
        radar.decay_signal()
        assert len(radar.blips) == 3
    with patch("overlay.time.monotonic", return_value=10.36):
        radar.decay_signal()
        assert not radar.blips

    mono = MonoMixThread()
    output = mono._to_mono(mixed)
    np.testing.assert_allclose(output[:, 1], mixed.mean(axis=1), atol=1e-8)
    np.testing.assert_array_equal(output[:, 0], output[:, 1])
    for n in range(5):
        mono.feed(np.full((1200, 2), n))
    assert mono._q.qsize() == 2
    assert mono._q.get_nowait()[0, 0] == 3  # Old playback is dropped.
    assert AudioRadarApp._validated_audio_param("deadzone", float("nan")) is None
    assert AudioRadarApp._validated_audio_param("deadzone", 9) == 0.4

    # Mono failures reach the UI across real worker threads.
    statuses = []
    capture.status_signal.connect(statuses.append)
    capture.set_mono(True)
    with patch.object(MonoMixThread, "_resolve_speaker", side_effect=RuntimeError("headphones disconnected")), \
         patch.object(AudioCaptureThread, "_run_system_loopback", lambda self: self._mono.wait(2000)):
        capture.start()
        assert capture.wait(3000)
        app.processEvents()
    assert any("headphones disconnected" in s for s in statuses)

    # System fallback must capture the cable, never replayed headphone audio.
    cable = SimpleNamespace(name="CABLE Input", isloopback=True)
    phones = SimpleNamespace(name="Headphones", isloopback=True)
    with patch("audio_capture.sc.all_microphones", return_value=[phones, cable]), \
         patch("audio_capture.sc.default_speaker", return_value=phones), \
         patch("mono_output.detect_virtual_cable", return_value=cable.name), \
         patch.object(capture, "_capture_loop") as record:
        capture._run_system_loopback()
        record.assert_called_once_with(cable, [])
    radar.close()
