import ctypes
import time
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen

from direction import angle_diff

class OverlayRadar(QWidget):
    positionChanged = pyqtSignal(int, int)   # committed position (persist to disk)
    positionPreview = pyqtSignal(int, int)   # live drag frames (UI readout only)

    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Visual Audio Overlay")
        self.base_window_flags = (
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.drag_enabled = False
        self.drag_start_global = None
        self.drag_start_window = None
        self._apply_window_flags()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        self.resize(300, 300)
        
        # Default accent must match the dashboard's default swatch (#9751F2) so the
        # first Start looks the same as the UI shows, even before the user touches
        # the colour picker.
        self.accent_color = QColor("#9751F2")
        self.stroke_width = 6
        self.blips = []
        self.stereo = True
        self.hold_ms = 200
        self.left_size = 1.5
        self.left_hold_ms = 150
        
        # Started/stopped with visibility (show/hideEvent) so the 30ms repaint
        # tick doesn't keep running while the overlay is hidden.
        self.decay_timer = QTimer(self)
        self.decay_timer.timeout.connect(self.decay_signal)

    def _apply_window_flags(self):
        flags = self.base_window_flags
        if not self.drag_enabled:
            flags |= Qt.WindowType.WindowTransparentForInput
        self.setWindowFlags(flags)

    def set_drag_enabled(self, enabled):
        enabled = bool(enabled)
        if self.drag_enabled == enabled:
            return

        was_visible = self.isVisible()
        pos = self.pos()
        self.drag_enabled = enabled
        self.drag_start_global = None
        self.drag_start_window = None
        self.setCursor(Qt.CursorShape.OpenHandCursor if enabled else Qt.CursorShape.ArrowCursor)
        self._apply_window_flags()
        self.move(pos)

        if was_visible:
            self.show()
            self.raise_()
            self._remove_win11_border()
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self.decay_timer.start(30)
        self._remove_win11_border()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.decay_timer.stop()
        self.blips = []

    def _remove_win11_border(self):
        """
        Windows 11 applies rounded corners and a border/shadow to ALL windows,
        including frameless ones. This opts out via the DWM API.
        Safe on non-Windows - the try/except swallows it silently.
        """
        try:
            hwnd = int(self.winId())

            # 1. Disable rounded corners
            #    DWMWA_WINDOW_CORNER_PREFERENCE = 33, DWMWCP_DONOTROUND = 1
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 33,
                ctypes.byref(ctypes.c_int(1)),
                ctypes.sizeof(ctypes.c_int)
            )

            # 2. Remove drop shadow / border glow
            class MARGINS(ctypes.Structure):
                _fields_ = [
                    ("cxLeftWidth",    ctypes.c_int),
                    ("cxRightWidth",   ctypes.c_int),
                    ("cyTopHeight",    ctypes.c_int),
                    ("cyBottomHeight", ctypes.c_int),
                ]
            ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(
                hwnd, ctypes.byref(MARGINS(0, 0, 0, 0))
            )
        except Exception:
            pass

    def set_accent_color(self, hex_color):
        self.accent_color = QColor(hex_color)
        self.update()
        
    def set_stroke_width(self, width):
        self.stroke_width = width
        self.update()

    def mousePressEvent(self, event):
        if not self.drag_enabled or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        self.drag_start_global = event.globalPosition().toPoint()
        self.drag_start_window = self.pos()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if not self.drag_enabled or self.drag_start_global is None or self.drag_start_window is None:
            return super().mouseMoveEvent(event)

        delta = event.globalPosition().toPoint() - self.drag_start_global
        new_pos = self.drag_start_window + delta
        self.move(new_pos)
        # Live update only - persisting every frame hammers the disk (issue #3).
        self.positionPreview.emit(new_pos.x(), new_pos.y())
        event.accept()

    def mouseReleaseEvent(self, event):
        if not self.drag_enabled or event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)

        self.drag_start_global = None
        self.drag_start_window = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        pos = self.pos()
        self.positionChanged.emit(pos.x(), pos.y())
        event.accept()
        
    def decay_signal(self):
        now = time.monotonic()
        self.blips = [b for b in self.blips if now < b['expires']]
        self.update()
        
    def update_audio_data(self, angle, intensity):
        visual_gain = 5.0
        clamped_intensity = max(0.45, min(1.0, intensity * visual_gain))
        if self.stereo:
            angle = -90.0 if angle < 0 else 90.0 if angle > 0 else 0.0
        expires = time.monotonic() + (self.hold_ms + (self.left_hold_ms if angle < 0 else 0)) / 1000 + 0.15
        
        found = False
        for blip in self.blips:
            # angle_diff wraps at +-180 so a sound directly behind the player
            # (surround: -179 vs +179) refreshes one blip instead of two.
            if angle_diff(blip['angle'], angle) < 20.0:
                blip['life'] = clamped_intensity
                blip['expires'] = expires
                blip['angle'] = angle
                found = True
                break
                
        if not found:
            self.blips.append({'angle': angle, 'life': clamped_intensity, 'expires': expires})
            
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = painter.font()
        font.setPixelSize(14)
        painter.setFont(font)

        if self.drag_enabled:
            # Layered windows can be hard to hit-test on fully transparent pixels.
            # A nearly invisible fill makes the whole radar box draggable in setup mode.
            painter.fillRect(self.rect(), QColor(255, 255, 255, 8))
        
        width = self.width()
        height = self.height()
        center = QPointF(width / 2, height / 2)
        radius = min(width, height) / 2 * 0.8
        
        base_pen = QPen(QColor(255, 255, 255, 30))
        base_pen.setWidth(2)
        painter.setPen(base_pen)
        if self.stereo:
            painter.setPen(QColor(255, 255, 255, 150))
            painter.drawText(QRectF(0, center.y() + 30, width, 24), Qt.AlignmentFlag.AlignCenter, "LEFT     ?     RIGHT")
        else:
            painter.drawEllipse(center, radius, radius)
        
        for blip in self.blips:
            fade = max(0.0, min(1.0, (blip['expires'] - time.monotonic()) / 0.15))
            opacity = int(blip['life'] * fade * 255)
            arc_color = QColor(
                self.accent_color.red(),
                self.accent_color.green(),
                self.accent_color.blue(),
                opacity
            )
            pen = QPen(arc_color)
            emphasis = self.left_size if blip["angle"] < 0 else 1
            pen.setWidthF(self.stroke_width * emphasis)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            if self.stereo and blip['angle'] == 0:
                font.setPixelSize(26)
                painter.setFont(font)
                painter.drawText(QRectF(center.x() - 35, center.y() - 20, 70, 40),
                                 Qt.AlignmentFlag.AlignCenter, "?")
                continue
            
            center_pyqt_angle = 90 - blip['angle']
            span_degrees = 35 * emphasis
            start_deg = center_pyqt_angle - (span_degrees / 2)
            
            start_angle_16 = int(start_deg * 16)
            span_angle_16  = int(span_degrees * 16)
            
            rect = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
            painter.drawArc(rect, start_angle_16, span_angle_16)

        if self.drag_enabled:
            setup_pen = QPen(QColor(255, 255, 255, 120))
            setup_pen.setWidth(1)
            setup_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(setup_pen)
            painter.drawEllipse(center, radius + 8, radius + 8)
