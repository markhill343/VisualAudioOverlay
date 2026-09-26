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
        assert [b["angle"] for b in radar.blips] == [-90]
    with patch("overlay.time.monotonic", return_value=10.51):
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
    assert AudioRadarApp._validated_audio_param("left_size", 9) == 3
    radar.left_hold_ms = 0
    with patch("overlay.time.monotonic", return_value=20):
        radar.update_audio_data(-90, 0.01)
        radar.update_audio_data(90, 0.01)
    assert radar.blips[0]['expires'] == radar.blips[1]['expires']

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


def test_hotkey_hides_only_visuals_and_ignores_key_repeat():
    import ctypes
    from ctypes import wintypes
    from unittest.mock import Mock
    app = QApplication.instance() or QApplication([])
    radar = OverlayRadar()
    radar.show()
    host = SimpleNamespace(radar_active=True, overlay=radar,
                           bridge=SimpleNamespace(statusChanged=Mock()),
                           emit_overlay_position=Mock(), winId=lambda: 123)
    host.toggle_overlay = lambda: AudioRadarApp.toggle_overlay(host)
    with patch("main.ctypes.windll.user32.RegisterHotKey", return_value=1) as register:
        AudioRadarApp._register_hotkey(host)
        assert register.call_args.args[2:] == (0x4003, ord('H'))
    msg = wintypes.MSG()
    msg.message, msg.wParam = 0x0312, 1
    assert AudioRadarApp.nativeEvent(host, b'windows_generic_MSG', ctypes.addressof(msg)) == (True, 0)
    assert not radar.isVisible() and host.radar_active
    msg.message = 0x000F  # Ordinary paint messages must pass through untouched.
    assert AudioRadarApp.nativeEvent(host, b'windows_generic_MSG', ctypes.addressof(msg)) == (False, 0)
    host.toggle_overlay()
    assert radar.isVisible()
    host.radar_active = False
    radar.hide()
    host.toggle_overlay()
    assert not radar.isVisible()
    with patch("main.ctypes.windll.user32.RegisterHotKey", return_value=0):
        AudioRadarApp._register_hotkey(host)
        assert host._hotkey_hwnd is None and "already in use" in host.hotkey_status
    radar.close()


def test_stereo_switch_surround_and_silent_start():
    from unittest.mock import Mock, MagicMock
    capture = AudioCaptureThread(freq_low=20, freq_high=20000)
    capture.noise_ratio = 0
    events, devices = [], []
    capture.audio_data_signal.connect(lambda a, i: events.append((a, i)))
    capture.device_info_signal.connect(lambda name, channels: devices.append(channels))
    t = np.arange(1200) / 48000
    data = np.zeros((1200, 8))
    data[:, 4] = np.sin(2 * np.pi * 240 * t) * 0.1
    capture._process_chunk(data, True)
    assert events[-1][0] == pytest.approx(-135)
    events.clear()
    capture._process_chunk(data, False)
    assert events and all(-90 <= a < 0 for a, _ in events)
    data[:, 4] = 0
    data[:, 6] = np.sin(2 * np.pi * 240 * t) * 0.1
    events.clear()
    capture._process_chunk(data, True)
    assert events[-1][0] == pytest.approx(-90)
    data[:, 6] = 0
    data[:, 3] = 0.5
    events.clear()
    capture._process_chunk(data, True)
    assert not events  # Subwoofer-only audio has no bearing.
    for forced, raw_channels, expected in [(False, 8, 8), (True, 8, 2), (False, 2, 2)]:
        capture.force_stereo = forced
        capture.running = False
        device = SimpleNamespace(name='Test output', recorder=MagicMock())
        device.recorder.return_value.__enter__.return_value.record.return_value = np.zeros((1200, raw_channels))
        from audio_devices import channel_labels
        mask = 0x63f if raw_channels == 8 else 3
        with patch('audio_capture.endpoint_format', return_value={'mask': mask, 'labels': channel_labels(raw_channels, mask)}):
            capture._capture_loop(device, [])
        assert devices[-1] == expected  # Silence must not permanently select stereo.
    host = SimpleNamespace(audio_settings={'stereo_mode': 1},
                           _validated_audio_param=AudioRadarApp._validated_audio_param,
                           _apply_audio_settings_to_thread=Mock(), _queue_settings_save=Mock(),
                           _restart_capture_if_active=Mock())
    AudioRadarApp.set_audio_param(host, 'stereo_mode', 0)
    host._restart_capture_if_active.assert_called_once()

    AudioRadarApp.set_audio_param(host, 'stereo_mode', 0)
    host._restart_capture_if_active.assert_called_once()


def test_explicit_device_routing_and_raw_meters():
    import json
    from audio_devices import channel_labels
    capture = AudioCaptureThread(sensitivity=1, freq_low=10000, freq_high=20000)
    selected = SimpleNamespace(id='cable-id', name='Cable', isloopback=True)
    phones = SimpleNamespace(id='phones-id', name='Headphones', isloopback=True)
    capture.capture_device_id = selected.id
    statuses, meters, cues = [], [], []
    capture.status_signal.connect(statuses.append)
    capture.channel_levels_signal.connect(lambda s: meters.append(json.loads(s)))
    capture.audio_data_signal.connect(lambda a, i: cues.append(a))
    with patch('audio_capture.sc.all_microphones', return_value=[phones, selected]), \
         patch('audio_capture.sc.default_speaker', return_value=phones), \
         patch.object(capture, '_capture_loop') as record:
        capture._run_system_loopback()
        record.assert_called_once_with(selected, [])
        record.reset_mock()
        capture.capture_device_id = 'missing'
        capture._run_system_loopback()
        record.assert_not_called()
        assert 'unavailable' in statuses[-1]
        capture.capture_device_id = phones.id
        capture.set_mono(True)
        capture._run_system_loopback()
        record.assert_not_called()
        assert 'feedback' in statuses[-1]
    capture.channel_labels = channel_labels(8, 0x63f)
    for channel in range(8):
        data = np.zeros((1200, 8))
        data[:, channel] = 0.1
        capture._next_meter = 0
        capture._process_chunk(data, True)
        assert meters[-1]['db'][channel] == pytest.approx(-20)
        assert sum(v > -60 for v in meters[-1]['db']) == 1
    assert meters[-1]['seen'] == list(range(8))
    assert not cues  # Meters ignore frequency and sensitivity filtering.
    capture.channel_labels = channel_labels(6, 0x60f)
    capture.sensitivity = 0.005
    capture.noise_ratio = 0
    capture.set_freq_range(20, 20000)
    data = np.zeros((1200, 6))
    data[:, 4] = 0.1
    capture._process_chunk(data, True)
    assert cues[-1] == pytest.approx(-90)  # A side-layout 5.1 stream is not rear-left.
