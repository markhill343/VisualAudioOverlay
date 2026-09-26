import numpy as np
import soundcard as sc
from PyQt6.QtCore import QThread, pyqtSignal
import time
import json
from audio_devices import endpoint_format, SURROUND_MASKS

from direction import band_rms, stereo_angle, surround_angle

class AudioCaptureThread(QThread):
    audio_data_signal = pyqtSignal(float, float)
    device_info_signal = pyqtSignal(str, int)
    channel_levels_signal = pyqtSignal(str)
    # Human-readable capture problems (device gone, no loopback, fallbacks).
    # The app forwards these to the dashboard status line so failures are
    # visible to the user, not just printed to a console nobody sees.
    status_signal = pyqtSignal(str)

    def __init__(self, sensitivity=0.005, gain=1.0, freq_low=20, freq_high=20000, max_amplitude=1.0,
                 target_pid=None, target_name=None):
        super().__init__()
        self.sensitivity = sensitivity
        self.gain = gain
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.max_amplitude = max_amplitude  # ignore sounds louder than this (1.0 = no limit)
        self.running = True
        self.samplerate = 48000
        self.chunk_frames = 1200  # 25 ms, shared by both capture paths.
        self.force_stereo = True
        self.deadzone = 0.08
        self.noise_ratio = 1.5
        self._noise = {}
        self.capture_device_id = None
        self.channel_labels = None
        self._next_meter = 0.0
        self._meter_peak = None
        self._seen_channels = set()
        # When target_pid is set, capture only that program (and its children)
        # via WASAPI process loopback. None = whole-system loopback (soundcard).
        self.target_pid = target_pid
        self.target_name = target_name
        # Mono output: when enabled, the raw (unfiltered) captured audio is also
        # summed to mono and played to `mono_device` for single-sided listeners.
        # Read at thread start (like target); toggle via the app before Start.
        self.mono_enabled = False
        self.mono_device = None
        self._mono = None

    def set_target(self, pid, name=None):
        """Choose the capture source. Only takes effect before the thread starts."""
        self.target_pid = pid
        self.target_name = name

    def set_mono(self, enabled, device=None):
        """Enable/disable the mono down-mix output and pick its playback device.
        Only takes effect before the thread starts (the thread is recreated on
        each Start, so the app re-applies this in start_radar)."""
        self.mono_enabled = bool(enabled)
        self.mono_device = device or None

    def set_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity
    
    def set_gain(self, gain):
        self.gain = gain
    
    def set_freq_range(self, low, high):
        low, high = sorted((low, high))
        if (low, high) != (self.freq_low, self.freq_high):
            self._noise.clear()
        self.freq_low, self.freq_high = low, high
    
    def set_max_amplitude(self, max_amp):
        self.max_amplitude = max_amp

    def run(self):
        # Per-app capture takes the process-loopback path; otherwise capture the
        # whole system mix the way we always have.
        self._start_mono()
        try:
            if self.target_pid and not self.capture_device_id:
                self._run_process_loopback()
            else:
                self._run_system_loopback()
        finally:
            self._stop_mono()

    # ── Mono down-mix output ───────────────────────────────────────────
    def _start_mono(self):
        if not self.mono_enabled:
            return
        try:
            from mono_output import MonoMixThread
            self._mono = MonoMixThread(device_name=self.mono_device,
                                       samplerate=self.samplerate)
            self._mono.failed.connect(self.status_signal)
            self._mono.start()
            print(f"Mono output on -> {self.mono_device or 'default device'}")
        except Exception as e:
            print(f"Mono output unavailable ({e}); continuing without it.")
            self.status_signal.emit(f"Mono output unavailable: {e}")
            self._mono = None

    def _feed_mono(self, data):
        """Send the RAW (pre-bandpass) chunk to the mono player so the user hears
        the full game audio, not just the filtered footstep band."""
        if self._mono is not None:
            self._mono.feed(data)

    def _stop_mono(self):
        if self._mono is not None:
            try:
                self._mono.stop()
            except Exception:
                pass
            self._mono = None

    def _run_process_loopback(self):
        """Capture a single program's audio. Falls back to system audio on failure."""
        try:
            from process_loopback import ProcessLoopbackCapture
        except Exception as e:
            print(f"Process loopback unavailable ({e}); using system audio.")
            self.status_signal.emit("Per-app capture unavailable - using system audio")
            self._run_system_loopback()
            return

        cap = ProcessLoopbackCapture(self.target_pid, samplerate=self.samplerate, channels=2)
        try:
            cap.start()
        except Exception as e:
            print(f"Process loopback failed ({e}); using system audio.")
            cap.close()
            self.status_signal.emit("Per-app capture failed - using system audio")
            self._run_system_loopback()
            return

        label = self.target_name or f"PID {self.target_pid}"
        print(f"Capturing app audio: {label} (per-app, Stereo L/R)")
        self.device_info_signal.emit(f"{label} (per-app)", 2)
        self.channel_labels = ['FL', 'FR']
        try:
            while self.running:
                data = cap.read(self.chunk_frames)
                self._feed_mono(data)
                self._process_chunk(data, use_surround=False)
        except Exception as e:
            print(f"Process loopback capture error: {e}")
            if self.running:
                self.status_signal.emit(
                    f"Capture of {label} stopped unexpectedly - restart the radar")
        finally:
            cap.close()

    def _run_system_loopback(self):
        try:
            mics = sc.all_microphones(include_loopback=True)
            loopbacks = [m for m in mics if m.isloopback]
            
            if not loopbacks:
                print("No loopback device found.")
                self.status_signal.emit(
                    "No audio output device found - the radar can't capture anything")
                return
            
            # Pick device by priority
            device = None
            if self.capture_device_id:
                device = next((lb for lb in loopbacks if lb.id == self.capture_device_id), None)
                if device is None:
                    self.status_signal.emit("Selected capture device is unavailable; select it again. No fallback used.")
                    return
                if self.mono_enabled:
                    output = sc.get_speaker(self.mono_device) if self.mono_device else sc.default_speaker()
                    if output is None or output.id == device.id:
                        self.status_signal.emit("Capture and Mono Output must be different devices to prevent feedback")
                        return
            elif self.mono_enabled:
                # Capture the game's cable, never the headphones replaying our mix.
                from mono_output import detect_virtual_cable
                cable = detect_virtual_cable()
                device = next((lb for lb in loopbacks if cable and cable in lb.name), None)
                if device is None:
                    self.status_signal.emit("Mono needs game audio routed to a virtual cable; choose your game under Program")
                    return
            try:
                default_name = sc.default_speaker().name
                for lb in loopbacks:
                    if device is None and default_name in lb.name:
                        device = lb
                        break
            except:
                pass
            
            if not device:
                for lb in loopbacks:
                    if "Microphone" not in lb.name:
                        device = lb
                        break
            
            if not device:
                device = loopbacks[0]
            
            print(f"Using loopback device: {device.name}")
            self._capture_loop(device, [] if self.mono_enabled or self.capture_device_id else loopbacks)

        except Exception as e:
            print(f"Error in audio capture: {e}")
            import traceback
            traceback.print_exc()
            if self.running:
                self.status_signal.emit(f"Audio capture error: {e}")

    def _capture_loop(self, device, all_loopbacks):
        try:
            try:
                fmt = endpoint_format(device)
            except Exception:
                fmt = {'mask': 0, 'labels': []}
            with device.recorder(samplerate=self.samplerate) as mic:
                first_data = mic.record(numframes=self.chunk_frames)
                raw_channels = first_data.shape[1]
                self.channel_labels = fmt['labels'] if len(fmt['labels']) == raw_channels else [f'Ch {i + 1}' for i in range(raw_channels)]
                
                use_surround = fmt['mask'] in SURROUND_MASKS and raw_channels in (6, 8) and not self.force_stereo
                if raw_channels > 2 and fmt['mask'] not in SURROUND_MASKS:
                    self.status_signal.emit("Unrecognised channel layout: meters available, surround direction disabled")

                effective = raw_channels if use_surround else min(raw_channels, 2)
                mode = "360° Surround" if use_surround else "Stereo L/R"
                print(f"Channels: {raw_channels} | Mode: {mode}")
                self.device_info_signal.emit(device.name, effective)
                
                self._feed_mono(first_data)
                self._process_chunk(first_data, use_surround)

                while self.running:
                    data = mic.record(numframes=self.chunk_frames)
                    self._feed_mono(data)
                    self._process_chunk(data, use_surround)
                    
        except RuntimeError as e:
            print(f"Device '{device.name}' failed: {e}")
            if self.running:
                self.status_signal.emit(
                    f"Audio device '{device.name}' failed - " +
                    ("trying another output" if all_loopbacks else "stop and restart capture; no fallback used"))
            for lb in all_loopbacks:
                if lb.name == device.name or "Microphone" in lb.name:
                    continue
                try:
                    time.sleep(0.5)
                    self._capture_loop(lb, [])
                    return
                except Exception:
                    continue
            # Fallbacks exhausted (or none to try): tell the user instead of
            # leaving a radar that silently never blips again.
            if self.running and all_loopbacks:
                self.status_signal.emit(
                    "All audio devices failed - stop and restart the radar")
    
    def _process_chunk(self, data, use_surround):
        if data.ndim != 2 or not data.size or not np.isfinite(data).all():
            return
        # Raw meters precede downmix, frequency filtering, gain and noise gates.
        peak = np.max(np.abs(data), axis=0)
        self._meter_peak = peak if self._meter_peak is None or len(peak) != len(self._meter_peak) else np.maximum(peak, self._meter_peak)
        self._seen_channels.update(int(i) for i in np.flatnonzero(peak > 0.001))
        now = time.monotonic()
        if now >= self._next_meter:
            labels = self.channel_labels or [f'Ch {i + 1}' for i in range(data.shape[1])]
            self.channel_levels_signal.emit(json.dumps({'labels': labels,
                'db': (20 * np.log10(np.maximum(self._meter_peak, 1e-5))).tolist(),
                'seen': sorted(self._seen_channels), 'rate': self.samplerate}))
            self._meter_peak = None
            self._next_meter = now + 0.1
        if self.channel_labels and data.shape[1] >= 6:
            canonical = ['FL', 'FR', 'FC', 'LFE', 'BL', 'BR', 'SL', 'SR']
            if not set(self.channel_labels).issubset(canonical):
                return  # Meters stay useful without guessing unknown positions.
            data = np.column_stack([data[:, self.channel_labels.index(c)] if c in self.channel_labels
                                    else np.zeros(len(data)) for c in canonical])
        if not use_surround and data.shape[1] >= 6:
            # Windows 5.1/7.1 order: FL FR C LFE BL BR [SL SR]. Omit LFE.
            left = data[:, 0] + 0.707 * (data[:, 2] + data[:, 4])
            right = data[:, 1] + 0.707 * (data[:, 2] + data[:, 5])
            if data.shape[1] >= 8:
                left = left + 0.707 * data[:, 6]
                right = right + 0.707 * data[:, 7]
            data = np.column_stack((left, right))
        # ponytail: three broad bands can reveal opposite-side sounds at different
        # frequencies; overlapping sources in the same band still cannot be separated.
        edges = sorted({self.freq_low, self.freq_high} |
                       {f for f in (500, 2000) if self.freq_low < f < self.freq_high})
        bands = [(self.freq_low, self.freq_high)] if use_surround else list(zip(edges, edges[1:]))
        dt = len(data) / self.samplerate
        for low, high in bands:
            rms = band_rms(data, self.samplerate, low, high)
            if use_surround and len(rms) >= 6:
                rms = rms.copy()
                rms[3] = 0
            intensity = float(max(rms))
            key = (low, high)
            floor = self._noise.get(key, 0.0)
            tau = 2.0 if intensity > floor else 0.2
            self._noise[key] = floor + (intensity - floor) * (1 - np.exp(-dt / tau))
            threshold = max(self.sensitivity, floor * self.noise_ratio) if self.noise_ratio else self.sensitivity
            if intensity <= threshold or (self.max_amplitude < 1 and intensity >= self.max_amplitude):
                continue
            if use_surround and len(rms) >= 6:
                sl, sr = (rms[6] / 2, rms[7] / 2) if len(rms) >= 8 else (0, 0)
                angle = surround_angle(rms[0] + sl, rms[1] + sr, rms[2], rms[4] + sl, rms[5] + sr)
            elif len(rms) >= 2:
                angle = stereo_angle(rms[0], rms[1], self.deadzone)
            else:
                angle = 0.0
            # Gain affects visibility only, never detection or loud-sound rejection.
            self.audio_data_signal.emit(float(angle), intensity * self.gain)

    def stop(self):
        self.running = False
        self.wait()
