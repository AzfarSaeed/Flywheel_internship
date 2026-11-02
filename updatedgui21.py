import sys
import time
from pathlib import Path
import numpy as np
import serial
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QFrame, QGridLayout, QPushButton,
                             QSizePolicy, QLineEdit, QCheckBox, QGraphicsDropShadowEffect, QFileDialog, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)

# ============================================
#  THEME CONFIGURATION (Light / Off-White Mode)
# ============================================
BG_MAIN = "#F4F6FA"       # app background (off-white)
BG_CARD = "#FFFFFF"       # card background (white)
BG_SIDEBAR = "#FFFFFF"    # sidebar background
BG_SOFT = "#F1F5F9"       # soft inset background (inputs, chips)
TEXT_MAIN = "#1E293B"     # slate-800
TEXT_SUB = "#64748B"      # slate-500
BORDER_COLOR = "#E2E8F0"  # slate-200

ACCENT_BLUE = "#2563EB"   # UI accent (nav highlight, buttons)
NEEDLE_BLUE = "#0EA5E9"   # current-direction color (blue)
TARGET_RED = "#DC2626"    # target-direction color (red)
ERROR_RED = "#DC2626"
SUCCESS_GREEN = "#16A34A" # grid / success color (green)
GRID_GREEN = "#4ADE80"
GRID_FILL = "#DCFCE7"
NEEDLE_GRAY = "#94A3B8"   # needle shaft / hub color
NEEDLE_GRAY_DARK = "#64748B"

SUCCESS_BG = "#ECFDF5"
ERROR_BG = "#FEF2F2"


def add_card_shadow(widget, blur=24, y_offset=4, alpha=35):
    """Subtle drop shadow so white cards lift off the off-white background."""
    effect = QGraphicsDropShadowEffect(widget)
    effect.setBlurRadius(blur)
    effect.setOffset(0, y_offset)
    effect.setColor(QColor(15, 23, 42, alpha))
    widget.setGraphicsEffect(effect)


# ============================================
#  3D GEOMETRY HELPERS (pure numpy, no STL needed)
# ============================================
def spherical_to_cartesian(pitch_deg, yaw_deg, length=1.0):
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)
    x = length * np.cos(pitch_rad) * np.sin(yaw_rad)
    y = length * np.cos(pitch_rad) * np.cos(yaw_rad)
    z = length * np.sin(pitch_rad)
    return np.array([x, y, z])


def _rotation_from_z(direction_unit):
    """Rotation matrix that maps +Z onto direction_unit (Rodrigues' formula)."""
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, direction_unit)
    c = float(np.dot(z_axis, direction_unit))
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def _tapered_tube_mesh(start, direction_unit, r_start, r_end, height, n=16):
    """A cylinder/cone segment (tapered tube) built from 'start' along
    'direction_unit' for 'height', with radius going r_start -> r_end."""
    theta = np.linspace(0, 2 * np.pi, n)
    z = np.array([0.0, height])
    theta_grid, z_grid = np.meshgrid(theta, z)
    frac = z_grid / height if height != 0 else z_grid
    r_grid = r_start + (r_end - r_start) * frac
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    pts = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1)
    R = _rotation_from_z(direction_unit)
    pts_rot = pts @ R.T + np.asarray(start)
    X = pts_rot[:, 0].reshape(x_grid.shape)
    Y = pts_rot[:, 1].reshape(x_grid.shape)
    Z = pts_rot[:, 2].reshape(x_grid.shape)
    return X, Y, Z


def _direction_arc_points(v1, v2, radius=0.55, samples=40):
    """Return a smooth great-circle arc between two 3D unit vectors."""
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.empty((0, 3))

    a /= na
    b /= nb
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    angle = np.arccos(dot)

    if angle < np.radians(0.5):
        return np.empty((0, 3))

    # If the vectors are nearly opposite, choose a stable perpendicular axis.
    if np.sin(angle) < 1e-6:
        helper = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(a, helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        perpendicular = np.cross(a, helper)
        perpendicular /= np.linalg.norm(perpendicular)
        ts = np.linspace(0.0, angle, samples)
        pts = []
        for t in ts:
            p = np.cos(t) * a + np.sin(t) * perpendicular
            pts.append(radius * p)
        return np.asarray(pts)

    ts = np.linspace(0.0, 1.0, samples)
    sin_angle = np.sin(angle)
    pts = []
    for t in ts:
        p = (
            np.sin((1.0 - t) * angle) / sin_angle * a
            + np.sin(t * angle) / sin_angle * b
        )
        p /= np.linalg.norm(p)
        pts.append(radius * p)
    return np.asarray(pts)


# ============================================
#  3D NEEDLE CANVAS (No STL, pure matplotlib)
# ============================================
class Needle3DCanvas(FigureCanvas):
    """Real-time 3D direction view with a time-based, angle-aware animation.

    The coordinate frame is drawn once. Incoming telemetry updates the desired
    orientation, while a GUI timer eases the displayed needle toward it. This
    keeps the visual movement smooth without modifying raw telemetry, guidance,
    device status, target capture, or CSV recording.
    """

    SMOOTHING_TIME_CONSTANT = 0.075  # seconds; increase for more smoothing
    FRAME_INTERVAL_MS = 16           # nominal 60 Hz animation timer

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6.5, 6.0), dpi=100)
        self.fig.patch.set_facecolor(BG_CARD)
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.ax = self.fig.add_subplot(111, projection="3d")
        super().__init__(self.fig)
        self.setParent(parent)

        self.target_pitch = 30.0
        self.target_yaw = 0.0
        self.show_target = True

        # Most recent raw orientation requested by the serial-data handler.
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self._desired_pitch = 0.0
        self._desired_yaw = 0.0

        # Independently animated orientation used only for the 3D drawing.
        self._display_pitch = 0.0
        self._display_yaw = 0.0
        self._last_frame_time = time.perf_counter()
        self._dynamic_artists = []
        self._force_render = True

        self._setup_scene()
        self._render_needle(0.0, 0.0)

        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(self.FRAME_INTERVAL_MS)
        self._animation_timer.timeout.connect(self._animate_needle)
        self._animation_timer.start()

    def _setup_scene(self):
        """Draw the fixed coordinate frame and floor once, not per sample."""
        self.ax.cla()
        self.ax.set_facecolor("#081725")
        self.ax.set_axis_off()
        self.ax.set_xlim([-1.40, 1.40])
        self.ax.set_ylim([-1.40, 1.40])
        self.ax.set_zlim([-0.05, 1.62])
        self.ax.set_box_aspect([1.10, 1.10, 0.95])
        self.ax.view_init(elev=24, azim=45)

        # Fixed floor grid.
        grid_size = 1.25
        steps = 10
        for value in np.linspace(-grid_size, grid_size, steps):
            self.ax.plot(
                [-grid_size, grid_size], [value, value], [0.015, 0.015],
                color="#24485A", lw=0.8, alpha=0.62
            )
            self.ax.plot(
                [value, value], [-grid_size, grid_size], [0.015, 0.015],
                color="#24485A", lw=0.8, alpha=0.62
            )

        xx, yy = np.meshgrid(
            np.linspace(-grid_size, grid_size, 2),
            np.linspace(-grid_size, grid_size, 2)
        )
        self.ax.plot_surface(
            xx, yy, np.zeros_like(xx),
            color="#0B3A2A", alpha=0.46, shade=False, linewidth=0
        )

        # Fixed world axes and axis arrowheads.
        axis_z = 0.055
        self.ax.plot([0, 1.32], [0, 0], [axis_z, axis_z],
                     color="#EF4444", lw=2.0, alpha=0.98)
        self.ax.plot([0, 0], [0, 1.32], [axis_z, axis_z],
                     color="#22C55E", lw=2.0, alpha=0.98)
        self.ax.plot([0, 0], [0, 0], [axis_z, 1.45],
                     color="#CBD5E1", lw=1.7, linestyle=":", alpha=0.98)
        self.ax.quiver(1.15, 0, axis_z, 0.17, 0, 0,
                       color="#EF4444", linewidth=1.8, arrow_length_ratio=0.65)
        self.ax.quiver(0, 1.15, axis_z, 0, 0.17, 0,
                       color="#22C55E", linewidth=1.8, arrow_length_ratio=0.65)
        self.ax.quiver(0, 0, 1.28, 0, 0, 0.17,
                       color="#94A3B8", linewidth=1.7, arrow_length_ratio=0.65)
        self.ax.text(1.36, -0.02, axis_z + 0.01, "X",
                     color="#EF4444", fontsize=11, fontweight="bold")
        self.ax.text(-0.02, 1.36, axis_z + 0.01, "Y",
                     color="#22C55E", fontsize=11, fontweight="bold")
        self.ax.text(-0.02, -0.02, 1.50, "Z (Up)",
                     color="#CBD5E1", fontsize=10, fontweight="bold")

    def _clear_dynamic_artists(self):
        """Remove just the moving artists, leaving the world frame untouched."""
        for artist in self._dynamic_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._dynamic_artists.clear()

    def _track_plot(self, *args, **kwargs):
        """Plot a line and track its artists for safe replacement next frame."""
        artists = self.ax.plot(*args, **kwargs)
        self._dynamic_artists.extend(artists)
        return artists

    @staticmethod
    def _angle_delta(target, current):
        """Shortest signed angular difference, in degrees, in [-180, 180)."""
        return (float(target) - float(current) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _wrap_angle(angle):
        return (float(angle) + 180.0) % 360.0 - 180.0

    def set_show_target(self, visible):
        self.show_target = bool(visible)
        self._force_render = True
        self._render_needle(self._display_pitch, self._display_yaw)
        self.draw_idle()

    def set_target_direction(self, pitch, yaw):
        """Refresh the displayed target after the target-capture window ends."""
        self.target_pitch = float(pitch)
        self.target_yaw = float(yaw)
        self._force_render = True
        self._render_needle(self._display_pitch, self._display_yaw)
        self.draw_idle()

    def update_needle(self, current_pitch, current_yaw):
        """Accept the newest raw sample; animation renders it smoothly later."""
        try:
            pitch = float(current_pitch)
            yaw = float(current_yaw)
        except (TypeError, ValueError):
            return
        if not (np.isfinite(pitch) and np.isfinite(yaw)):
            return

        self.last_pitch = pitch
        self.last_yaw = yaw
        self._desired_pitch = pitch
        self._desired_yaw = self._wrap_angle(yaw)

    def _animate_needle(self):
        """Advance toward the latest orientation using frame-rate-independent easing."""
        now = time.perf_counter()
        dt = max(0.0, min(now - self._last_frame_time, 0.10))
        self._last_frame_time = now
        if dt <= 0.0:
            return

        previous_pitch = self._display_pitch
        previous_yaw = self._display_yaw

        tau = max(0.001, float(self.SMOOTHING_TIME_CONSTANT))
        alpha = 1.0 - np.exp(-dt / tau)
        self._display_pitch += alpha * (self._desired_pitch - self._display_pitch)
        yaw_delta = self._angle_delta(self._desired_yaw, self._display_yaw)
        self._display_yaw = self._wrap_angle(self._display_yaw + alpha * yaw_delta)

        # Snap tiny residuals to the destination so the needle settles exactly.
        if abs(self._desired_pitch - self._display_pitch) < 0.01:
            self._display_pitch = self._desired_pitch
        if abs(self._angle_delta(self._desired_yaw, self._display_yaw)) < 0.01:
            self._display_yaw = self._desired_yaw

        pitch_moved = abs(self._display_pitch - previous_pitch) >= 0.015
        yaw_moved = abs(self._angle_delta(self._display_yaw, previous_yaw)) >= 0.015
        if self._force_render or pitch_moved or yaw_moved:
            self._render_needle(self._display_pitch, self._display_yaw)
            self.draw_idle()
            self._force_render = False

    def _render_needle(self, current_pitch, current_yaw):
        """Rebuild moving overlays only; static axes/grid are never redrawn."""
        self._clear_dynamic_artists()
        axis_z = 0.055
        base = np.array([0.0, 0.0, 0.075])

        # Origin marker.
        theta = np.linspace(0, 2 * np.pi, 60)
        origin_r = 0.08
        self._track_plot(
            origin_r * np.cos(theta), origin_r * np.sin(theta),
            np.full_like(theta, axis_z), color="#1F4B5E", linewidth=1.2, alpha=0.85
        )

        # Target direction.
        target_dir = spherical_to_cartesian(self.target_pitch, self.target_yaw, 1.0)
        target_norm = np.linalg.norm(target_dir)
        if target_norm > 1e-12:
            target_dir /= target_norm
        target_len = 1.28
        target_end = base + target_dir * target_len

        if self.show_target:
            self._track_plot(
                [base[0], target_end[0]], [base[1], target_end[1]],
                [base[2], target_end[2]], color="#39E85F", linestyle="--",
                linewidth=2.7, alpha=0.96
            )
            self._dynamic_artists.append(self.ax.quiver(
                target_end[0] - target_dir[0] * 0.16,
                target_end[1] - target_dir[1] * 0.16,
                target_end[2] - target_dir[2] * 0.16,
                target_dir[0] * 0.16, target_dir[1] * 0.16, target_dir[2] * 0.16,
                color="#39E85F", linewidth=2.0, arrow_length_ratio=0.80
            ))
            target_label = target_end + np.array([-0.02, 0.00, 0.09])
            self._dynamic_artists.append(self.ax.text(
                target_label[0], target_label[1], target_label[2],
                "Target Direction", color="#39E85F", fontsize=10, fontweight="bold"
            ))

        # Current direction vector.
        current_dir = spherical_to_cartesian(current_pitch, current_yaw, 1.0)
        current_norm = np.linalg.norm(current_dir)
        if current_norm < 1e-12:
            current_dir = np.array([0.0, 1.0, 0.0])
        else:
            current_dir /= current_norm
        current_len = 1.32
        current_end = base + current_dir * current_len

        self._track_plot(
            [base[0], current_end[0]], [base[1], current_end[1]],
            [base[2], current_end[2]], color="#29B6F6", linewidth=2.8, alpha=0.96
        )
        self._dynamic_artists.append(self.ax.quiver(
            current_end[0] - current_dir[0] * 0.18,
            current_end[1] - current_dir[1] * 0.18,
            current_end[2] - current_dir[2] * 0.18,
            current_dir[0] * 0.18, current_dir[1] * 0.18, current_dir[2] * 0.18,
            color="#29B6F6", linewidth=2.1, arrow_length_ratio=0.80
        ))

        # Physical needle geometry.
        tail_len = 0.18
        shaft_len = 0.88
        tip_len = 0.25
        tail_start = base - current_dir * tail_len
        X, Y, Z = _tapered_tube_mesh(
            tail_start, current_dir, r_start=0.072, r_end=0.052,
            height=tail_len, n=16
        )
        self._dynamic_artists.append(self.ax.plot_surface(
            X, Y, Z, color="#64748B", alpha=0.95, linewidth=0,
            shade=True, antialiased=True
        ))

        X, Y, Z = _tapered_tube_mesh(
            base, current_dir, r_start=0.026, r_end=0.023,
            height=shaft_len, n=16
        )
        self._dynamic_artists.append(self.ax.plot_surface(
            X, Y, Z, color="#E2E8F0", alpha=1.0, linewidth=0,
            shade=True, antialiased=True
        ))

        tip_start = base + current_dir * shaft_len
        X, Y, Z = _tapered_tube_mesh(
            tip_start, current_dir, r_start=0.068, r_end=0.0,
            height=tip_len, n=24
        )
        self._dynamic_artists.append(self.ax.plot_surface(
            X, Y, Z, color="#0EA5E9", alpha=1.0, linewidth=0,
            shade=True, antialiased=True
        ))

        lateral = np.array([-current_dir[1], current_dir[0], 0.0])
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm > 1e-8:
            lateral /= lateral_norm
        current_label = current_end + lateral * 0.10 + np.array([0.0, 0.0, 0.07])
        self._dynamic_artists.append(self.ax.text(
            current_label[0], current_label[1], current_label[2],
            "Current Direction", color="#29B6F6", fontsize=10, fontweight="bold"
        ))

        # Angular difference arc.
        if self.show_target:
            arc = _direction_arc_points(current_dir, target_dir, radius=0.38, samples=24)
            if len(arc):
                self._track_plot(
                    arc[:, 0], arc[:, 1], arc[:, 2] + 0.055,
                    color="#37C96B", linewidth=2.2, alpha=0.9
                )


# ============================================
#  MAIN DASHBOARD
# ============================================
class NeedleMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Needle Navigation System")
        self.resize(1500, 860)
        self.setMinimumSize(1180, 720)
        self.setFont(QFont("Segoe UI", 12))
        self.setStyleSheet(f"background-color: {BG_MAIN}; color: {TEXT_MAIN}; font-family: 'Segoe UI', Arial, sans-serif;")

        # Serial config
        self.serial_port = None
        self.is_serial_active = False

        # Target positions
        self.target_pitch = 30.0
        self.target_yaw = 0.0
        self.target_roll = 0.0

        # Last raw telemetry values (always stored in degrees).
        self._last_pitch = 0.0
        self._last_yaw = 0.0
        self._last_roll = 0.0

        # Session data used by Export CSV.
        self.live_data = []

        # Automatic 10-second target capture after connecting.
        self.target_capture_active = False
        self.target_capture_samples = []
        self.target_capture_start = None
        self.target_capture_dialog = None
        self.target_capture_timer = QTimer()
        self.target_capture_timer.setInterval(100)
        self.target_capture_timer.timeout.connect(self._target_capture_tick)

        # Arduino/device status command transmitter.
        # Protocol:
        #   C = target is currently being captured (10-second window)
        #   R = target captured, but needle is outside tolerance
        #   G = target captured, and needle is within tolerance
        #
        # The command is sent periodically so the connected MCU always has the
        # latest state, rather than relying on a single one-time character.
        self.device_status = "R"
        self.device_status_timer = QTimer()
        self.device_status_timer.setInterval(100)  # 10 status messages/second
        self.device_status_timer.timeout.connect(self._send_device_status)

        self.setup_ui()

        # Serial polling. The reader drains all waiting lines each tick so a
        # high-rate stream does not leave stale data sitting in the input buffer.
        self.serial_timer = QTimer()
        self.serial_timer.setInterval(20)  # UI refresh ~50 Hz; all queued samples are still consumed.
        self.serial_timer.timeout.connect(self.read_serial_data)

    # ------------------------------------------------------------
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        # Responsive layout: no fixed right panel width and no horizontal
        # scrolling. The 3D view and the right-side panels share the window
        # width, so the interface works both maximized and in a normal window.
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(12)

        # Header: logo on the left, title/subtitle centered, connection status on the right.
        header_layout = QHBoxLayout()

        # Left logo area.
        logo_box = QFrame()
        logo_box.setFixedWidth(190)
        logo_box.setStyleSheet("background: transparent; border: none;")
        logo_layout = QHBoxLayout(logo_box)
        logo_layout.setContentsMargins(0, 0, 0, 0)
        logo_layout.setSpacing(0)

        self.logo_label = QLabel()
        self.logo_label.setFixedSize(150, 52)
        self.logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.logo_label.setStyleSheet("background: transparent; border: none;")

        logo_path = Path(__file__).resolve().parent / "logo.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                self.logo_label.setPixmap(
                    pixmap.scaled(
                        self.logo_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
        logo_layout.addWidget(self.logo_label)
        header_layout.addWidget(logo_box)

        # Center title block.
        center_box = QFrame()
        center_box.setStyleSheet("background: transparent; border: none;")
        center_layout = QVBoxLayout(center_box)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(2)
        center_layout.setAlignment(Qt.AlignCenter)

        lbl_title = QLabel("Smart Needle Navigation System")
        lbl_title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        lbl_title.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")
        lbl_title.setAlignment(Qt.AlignCenter)

        lbl_sub = QLabel("Precision Guidance for Safer Procedures")
        lbl_sub.setFont(QFont("Segoe UI", 12))
        lbl_sub.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        lbl_sub.setAlignment(Qt.AlignCenter)

        center_layout.addWidget(lbl_title)
        center_layout.addWidget(lbl_sub)
        header_layout.addWidget(center_box, stretch=1)

        # Right status.
        status_box = QFrame()
        status_box.setFixedWidth(230)
        status_box.setStyleSheet("background: transparent; border: none;")
        status_layout = QHBoxLayout(status_box)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setAlignment(Qt.AlignRight | Qt.AlignTop)

        self.lbl_status = QLabel("●  Device Disconnected")
        self.lbl_status.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.lbl_status.setStyleSheet(f"color: {ERROR_RED}; border: none;")
        self.lbl_status.setAlignment(Qt.AlignRight | Qt.AlignTop)
        status_layout.addWidget(self.lbl_status)
        header_layout.addWidget(status_box)

        main_layout.addLayout(header_layout)

        # Main dashboard: responsive left/right split.
        dash_layout = QHBoxLayout()
        dash_layout.setContentsMargins(0, 0, 0, 0)
        dash_layout.setSpacing(14)

        # LEFT: 3D view + horizontal controls
        left_column = QVBoxLayout()
        left_column.setSpacing(10)

        view_frame = QFrame()
        view_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view_frame.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 12px; "
            f"border: 1px solid {BORDER_COLOR};"
        )
        add_card_shadow(view_frame)
        view_layout = QVBoxLayout(view_frame)
        view_layout.setContentsMargins(14, 12, 14, 12)
        view_layout.setSpacing(8)

        view_header = QHBoxLayout()
        lbl_3d = QLabel("⬡ 3D View")
        lbl_3d.setFont(QFont("Segoe UI", 17, QFont.Bold))
        lbl_3d.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")

        self.chk_show_target = QCheckBox("Show Target")
        self.chk_show_target.setChecked(True)
        self.chk_show_target.setStyleSheet(
            f"color: {TEXT_SUB}; font-size: 12px; border: none;"
        )
        self.chk_show_target.setCursor(Qt.PointingHandCursor)
        self.chk_show_target.stateChanged.connect(
            lambda state: self.canvas3d.set_show_target(state == Qt.Checked)
        )

        lbl_legend_target = QLabel("┄ Target Direction")
        lbl_legend_target.setStyleSheet(
            f"color: {SUCCESS_GREEN}; font-weight: bold; font-size: 12px; border: none;"
        )
        lbl_legend_curr = QLabel("→ Current Direction")
        lbl_legend_curr.setStyleSheet(
            f"color: {NEEDLE_BLUE}; font-weight: bold; font-size: 12px; border: none;"
        )

        view_header.addWidget(lbl_3d)
        view_header.addStretch()
        view_header.addWidget(self.chk_show_target)
        view_header.addSpacing(12)
        view_header.addWidget(lbl_legend_target)
        view_header.addSpacing(10)
        view_header.addWidget(lbl_legend_curr)
        view_layout.addLayout(view_header)

        self.canvas3d = Needle3DCanvas()
        self.canvas3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view_layout.addWidget(self.canvas3d, stretch=1)
        left_column.addWidget(view_frame, stretch=1)

        # Bottom control bar
        controls_frame = QFrame()
        controls_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        controls_frame.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 12px; "
            f"border: 1px solid {BORDER_COLOR};"
        )
        add_card_shadow(controls_frame, blur=18, y_offset=3, alpha=28)

        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(10, 8, 10, 8)
        controls_layout.setSpacing(10)

        serial_group = QFrame()
        serial_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        serial_group.setStyleSheet(
            f"background: {BG_SOFT}; border: 1px solid {BORDER_COLOR}; border-radius: 8px;"
        )
        serial_layout = QHBoxLayout(serial_group)
        serial_layout.setContentsMargins(9, 5, 9, 5)
        serial_layout.setSpacing(6)

        serial_title = QLabel("SERIAL")
        serial_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        serial_title.setStyleSheet(f"color: {TEXT_SUB}; border: none;")

        port_label = QLabel("Port")
        port_label.setFont(QFont("Segoe UI", 10))
        port_label.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        self.port_input = QLineEdit("COM3")
        self.port_input.setFixedWidth(72)
        self.port_input.setFixedHeight(34)
        self.port_input.setStyleSheet(
            f"background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; "
            f"color: {TEXT_MAIN}; padding: 5px 7px; border-radius: 5px;"
        )

        baud_label = QLabel("Baud")
        baud_label.setFont(QFont("Segoe UI", 10))
        baud_label.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        self.baud_input = QLineEdit("115200")
        self.baud_input.setFixedWidth(70)
        self.baud_input.setFixedHeight(34)
        self.baud_input.setStyleSheet(
            f"background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; "
            f"color: {TEXT_MAIN}; padding: 5px 7px; border-radius: 5px;"
        )

        self.btn_connect = QPushButton("Connect Device")
        self.btn_connect.setCursor(Qt.PointingHandCursor)
        self.btn_connect.setFixedHeight(34)
        self.btn_connect.setStyleSheet(
            f"background-color: {ACCENT_BLUE}; color: white; font-weight: bold; "
            f"font-size: 11px; padding: 6px 12px; border-radius: 5px; border: none;"
        )
        self.btn_connect.clicked.connect(self.toggle_serial)

        serial_layout.addWidget(serial_title)
        serial_layout.addSpacing(4)
        serial_layout.addWidget(port_label)
        serial_layout.addWidget(self.port_input)
        serial_layout.addWidget(baud_label)
        serial_layout.addWidget(self.baud_input)
        serial_layout.addWidget(self.btn_connect)

        # Session / Export group
        export_group = QFrame()
        export_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        export_group.setStyleSheet(
            f"background: {BG_SOFT}; border: 1px solid {BORDER_COLOR}; border-radius: 8px;"
        )
        export_layout = QHBoxLayout(export_group)
        export_layout.setContentsMargins(9, 5, 9, 5)
        export_layout.setSpacing(8)

        export_title = QLabel("SESSION DATA")
        export_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        export_title.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")

        self.lbl_session_file = QLabel("Live telemetry is recorded automatically")
        self.lbl_session_file.setFont(QFont("Segoe UI", 10))
        self.lbl_session_file.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        self.lbl_session_file.setMinimumWidth(170)

        self.btn_export = QPushButton("↓  Export CSV")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setFixedHeight(34)
        self.btn_export.setStyleSheet(
            f"background-color: {SUCCESS_GREEN}; color: white; font-weight: bold; "
            f"font-size: 11px; padding: 6px 12px; border-radius: 5px; border: none;"
        )
        self.btn_export.clicked.connect(self.export_csv)

        export_layout.addWidget(export_title)
        export_layout.addWidget(self.lbl_session_file, stretch=1)
        export_layout.addWidget(self.btn_export)

        controls_layout.addWidget(serial_group, stretch=1)
        controls_layout.addWidget(export_group, stretch=1)
        left_column.addWidget(controls_frame, stretch=0)

        # RIGHT: orientation + guidance
        right_panel_widget = QWidget()
        right_panel_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel_widget.setMinimumWidth(430)
        right_panels = QVBoxLayout(right_panel_widget)
        right_panels.setContentsMargins(0, 0, 0, 0)
        right_panels.setSpacing(12)

        # Needle Orientation panel
        orient_frame = QFrame()
        orient_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        orient_frame.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 12px; border: 1px solid {BORDER_COLOR};"
        )
        add_card_shadow(orient_frame)
        orient_layout = QVBoxLayout(orient_frame)
        orient_layout.setContentsMargins(16, 13, 16, 13)
        orient_layout.setSpacing(7)

        orient_title = QLabel("Needle Orientation")
        orient_title.setFont(QFont("Segoe UI", 19, QFont.Bold))
        orient_title.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")
        orient_sub = QLabel("Live orientation relative to target")
        orient_sub.setFont(QFont("Segoe UI", 12))
        orient_sub.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        orient_layout.addWidget(orient_title)
        orient_layout.addWidget(orient_sub)

        orient_controls = QHBoxLayout()
        orient_controls.setSpacing(7)
        self.lbl_telemetry_badge = QLabel("●  LIVE TELEMETRY")
        self.lbl_telemetry_badge.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.lbl_telemetry_badge.setAlignment(Qt.AlignCenter)
        self.lbl_telemetry_badge.setFixedHeight(36)
        self.lbl_telemetry_badge.setStyleSheet(
            f"background: {SUCCESS_BG}; color: {SUCCESS_GREEN}; border: 1px solid #BBE7D0; "
            f"padding: 5px 9px; border-radius: 9px;"
        )

        # The needle GUI now uses degrees only.
        degree_badge = QLabel("DEGREES (°)")
        degree_badge.setFont(QFont("Segoe UI", 11, QFont.Bold))
        degree_badge.setAlignment(Qt.AlignCenter)
        degree_badge.setFixedHeight(36)
        degree_badge.setStyleSheet(
            f"background: {ACCENT_BLUE}; color: white; border: 1px solid {ACCENT_BLUE}; "
            f"padding: 5px 12px; border-radius: 8px;"
        )

        orient_controls.addWidget(self.lbl_telemetry_badge)
        orient_controls.addStretch()
        orient_controls.addWidget(degree_badge)
        orient_layout.addLayout(orient_controls)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        header_style = (
            f"color: {TEXT_SUB}; font-size: 12px; font-weight: bold; "
            f"letter-spacing: 1px; border: none;"
        )
        current_header = QLabel("CURRENT")
        target_header = QLabel("TARGET")
        error_header = QLabel("ERROR")
        for h in (current_header, target_header, error_header):
            h.setStyleSheet(header_style)
        grid.addWidget(current_header, 0, 1, alignment=Qt.AlignCenter)
        grid.addWidget(target_header, 0, 2, alignment=Qt.AlignCenter)
        grid.addWidget(error_header, 0, 3, alignment=Qt.AlignCenter)

        # Display order requested for the GUI: Roll, Pitch, Yaw.
        self.lbl_curr_roll, self.lbl_tar_roll, self.lbl_err_roll = self._create_row(
            grid, 1, "Roll", self.target_roll
        )
        self.lbl_curr_pitch, self.lbl_tar_pitch, self.lbl_err_pitch = self._create_row(
            grid, 2, "Pitch", self.target_pitch
        )
        self.lbl_curr_yaw, self.lbl_tar_yaw, self.lbl_err_yaw = self._create_row(
            grid, 3, "Yaw", self.target_yaw
        )
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)
        orient_layout.addLayout(grid)

        orient_hint = QLabel("Tolerance  ±2.0°  •  Target is fixed until changed")
        orient_hint.setFont(QFont("Segoe UI", 11))
        orient_hint.setFixedHeight(32)
        orient_hint.setStyleSheet(
            f"color: {TEXT_SUB}; background: {BG_SOFT}; border: 1px solid {BORDER_COLOR}; "
            f"border-radius: 7px; padding: 5px 10px;"
        )
        orient_layout.addWidget(orient_hint)

        state_frame = QFrame()
        state_frame.setFixedHeight(36)
        state_frame.setStyleSheet(
            f"background: {BG_SOFT}; border: 1px solid {BORDER_COLOR}; border-radius: 8px;"
        )
        state_layout = QHBoxLayout(state_frame)
        state_layout.setContentsMargins(10, 4, 10, 4)
        state_title = QLabel("NAVIGATION STATE")
        state_title.setFont(QFont("Segoe UI", 10, QFont.Bold))
        state_title.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        state_value = QLabel("Tracking orientation continuously")
        state_value.setFont(QFont("Segoe UI", 11, QFont.Bold))
        state_value.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")
        state_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        state_layout.addWidget(state_title)
        state_layout.addStretch()
        state_layout.addWidget(state_value)
        orient_layout.addWidget(state_frame)
        right_panels.addWidget(orient_frame, stretch=1)

        # Guidance Indicator panel
        guide_frame = QFrame()
        guide_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        guide_frame.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 12px; border: 1px solid {BORDER_COLOR};"
        )
        add_card_shadow(guide_frame)
        guide_layout = QVBoxLayout(guide_frame)
        guide_layout.setContentsMargins(16, 13, 16, 13)
        guide_layout.setSpacing(7)

        top_guide = QHBoxLayout()
        guidance_title_box = QVBoxLayout()
        guidance_title_box.setSpacing(1)
        lbl_guide = QLabel("Guidance Indicator")
        lbl_guide.setFont(QFont("Segoe UI", 19, QFont.Bold))
        lbl_guide.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")
        lbl_guide_sub = QLabel("Real-time directional correction")
        lbl_guide_sub.setFont(QFont("Segoe UI", 12))
        lbl_guide_sub.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        guidance_title_box.addWidget(lbl_guide)
        guidance_title_box.addWidget(lbl_guide_sub)

        self.lbl_range = QLabel("WAITING")
        self.lbl_range.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.lbl_range.setAlignment(Qt.AlignCenter)
        self.lbl_range.setMinimumWidth(120)
        self.lbl_range.setFixedHeight(34)
        self.lbl_range.setStyleSheet(
            f"background: {BG_SOFT}; color: {TEXT_SUB}; border: 1px solid {BORDER_COLOR}; "
            f"padding: 5px 9px; border-radius: 8px;"
        )
        top_guide.addLayout(guidance_title_box)
        top_guide.addStretch()
        top_guide.addWidget(self.lbl_range)
        guide_layout.addLayout(top_guide)

        indicator_layout = QHBoxLayout()
        indicator_layout.setSpacing(8)
        self.circle_lbl = QLabel("•")
        self.circle_lbl.setAlignment(Qt.AlignCenter)
        self.circle_lbl.setFixedSize(100, 100)
        self.circle_lbl.setFont(QFont("Segoe UI", 34, QFont.Bold))
        self.circle_lbl.setStyleSheet(
            f"color: {TEXT_SUB}; background: {BG_SOFT}; border: 7px solid {BORDER_COLOR}; "
            f"border-radius: 50px;"
        )

        status_box = QVBoxLayout()
        status_box.setSpacing(4)
        status_box.addWidget(self.circle_lbl, alignment=Qt.AlignCenter)
        self.lbl_ontarget = QLabel("NO DATA")
        self.lbl_ontarget.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.lbl_ontarget.setAlignment(Qt.AlignCenter)
        self.lbl_ontarget.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        self.lbl_steady = QLabel("Connect a device to begin guidance")
        self.lbl_steady.setAlignment(Qt.AlignCenter)
        self.lbl_steady.setWordWrap(True)
        self.lbl_steady.setStyleSheet(f"color: {TEXT_SUB}; font-size: 11px; border: none;")
        status_box.addWidget(self.lbl_ontarget)
        status_box.addWidget(self.lbl_steady)

        arrows_box = QGridLayout()
        arrows_box.setContentsMargins(0, 0, 0, 0)
        arrows_box.setHorizontalSpacing(6)
        arrows_box.setVerticalSpacing(5)
        arrow_button_style = f"""
            QPushButton {{
                color: {TEXT_SUB};
                background: {BG_CARD};
                border: 1px solid {BORDER_COLOR};
                border-radius: 9px;
                font-size: 12px;
                font-weight: bold;
                padding: 6px 10px;
                min-width: 74px;
                min-height: 42px;
            }}
        """
        self.btn_up = QPushButton("↑\nUP")
        self.btn_left = QPushButton("←  LEFT")
        self.btn_right = QPushButton("RIGHT  →")
        self.btn_down = QPushButton("↓\nDOWN")
        for btn in (self.btn_up, self.btn_left, self.btn_right, self.btn_down):
            btn.setStyleSheet(arrow_button_style)
            btn.setEnabled(False)
        cross = QLabel("•")
        cross.setFont(QFont("Arial", 18, QFont.Bold))
        cross.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        arrows_box.addWidget(self.btn_up, 0, 1, alignment=Qt.AlignCenter)
        arrows_box.addWidget(self.btn_left, 1, 0, alignment=Qt.AlignRight)
        arrows_box.addWidget(cross, 1, 1, alignment=Qt.AlignCenter)
        arrows_box.addWidget(self.btn_right, 1, 2, alignment=Qt.AlignLeft)
        arrows_box.addWidget(self.btn_down, 2, 1, alignment=Qt.AlignCenter)

        indicator_layout.addLayout(status_box, stretch=1)
        indicator_layout.addLayout(arrows_box, stretch=2)
        guide_layout.addLayout(indicator_layout, stretch=1)

        self.lbl_guidance_hint = QLabel(
            "Move only in the highlighted direction.  ±2° = within range."
        )
        self.lbl_guidance_hint.setAlignment(Qt.AlignCenter)
        self.lbl_guidance_hint.setFixedHeight(30)
        self.lbl_guidance_hint.setStyleSheet(
            f"color: {TEXT_SUB}; background: {BG_SOFT}; border: 1px solid {BORDER_COLOR}; "
            f"border-radius: 7px; padding: 5px 8px; font-size: 10px;"
        )
        guide_layout.addWidget(self.lbl_guidance_hint)
        right_panels.addWidget(guide_frame, stretch=1)

        dash_layout.addLayout(left_column, stretch=7)
        dash_layout.addWidget(right_panel_widget, stretch=4)
        main_layout.addLayout(dash_layout, stretch=1)

        self.update_telemetry(0.0, 0.0, 0.0)
        self._update_guidance_buttons(0.0, 0.0)

    def _create_row(self, grid, row, label, target_deg):
        lbl_1 = QLabel(label.upper())
        lbl_1.setFont(QFont("Segoe UI", 17, QFont.Bold))
        lbl_1.setStyleSheet(f"color: {TEXT_SUB}; border: none; padding-right: 6px;")
        lbl_1.setMinimumWidth(100)

        lbl_curr = QLabel("0.0°")
        lbl_curr.setFont(QFont("Segoe UI", 19, QFont.Bold))
        lbl_curr.setStyleSheet(
            f"color: {TEXT_MAIN}; background: {BG_SOFT}; "
            f"border: 1px solid {BORDER_COLOR}; border-radius: 7px; "
            f"padding: 6px 8px;"
        )
        lbl_curr.setMinimumWidth(100)
        lbl_curr.setMinimumHeight(54)

        lbl_tar = QLabel(self._fmt_angle(target_deg))
        lbl_tar.setFont(QFont("Segoe UI", 19, QFont.Bold))
        lbl_tar.setStyleSheet(
            f"color: {ACCENT_BLUE}; background: #EFF6FF; "
            f"border: 1px solid #D7E7FF; border-radius: 7px; "
            f"padding: 6px 7px;"
        )
        lbl_tar.setMinimumWidth(100)
        lbl_tar.setMinimumHeight(54)

        lbl_err = QLabel("0.0°")
        lbl_err.setFont(QFont("Segoe UI", 19, QFont.Bold))
        lbl_err.setStyleSheet(
            f"color: {TEXT_SUB}; background: {BG_SOFT}; "
            f"border: 1px solid {BORDER_COLOR}; border-radius: 7px; "
            f"padding: 6px 7px;"
        )
        lbl_err.setMinimumWidth(100)
        lbl_err.setMinimumHeight(54)
        lbl_err.setAlignment(Qt.AlignCenter)

        grid.addWidget(lbl_1, row, 0, alignment=Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(lbl_curr, row, 1, alignment=Qt.AlignCenter)
        grid.addWidget(lbl_tar, row, 2, alignment=Qt.AlignCenter)
        grid.addWidget(lbl_err, row, 3, alignment=Qt.AlignCenter)

        return lbl_curr, lbl_tar, lbl_err

    # ── REAL-TIME SERIAL + AUTOMATIC TARGET CAPTURE ────────────────────────
    def toggle_serial(self):
        if self.is_serial_active:
            self._stop_target_capture()
            self.serial_timer.stop()
            self.device_status_timer.stop()
            if self.serial_port:
                try:
                    self.serial_port.close()
                except Exception:
                    pass
            self.serial_port = None
            self.is_serial_active = False
            self.btn_connect.setText("Connect Device")
            self.btn_connect.setStyleSheet(
                f"background-color: {ACCENT_BLUE}; color: white; font-weight: bold; "
                f"padding: 8px; border-radius: 4px; border: none;"
            )
            self.lbl_status.setText("●  Device Disconnected")
            self.lbl_status.setStyleSheet(f"color: {ERROR_RED}; border: none;")
            self.lbl_telemetry_badge.setText("●  DISCONNECTED")
            self.lbl_telemetry_badge.setStyleSheet(
                f"background: {BG_SOFT}; color: {TEXT_SUB}; border: 1px solid {BORDER_COLOR}; "
                f"padding: 5px 9px; border-radius: 9px;"
            )
            self.lbl_range.setText("WAITING")
            self.lbl_range.setStyleSheet(
                f"background-color: {BG_SOFT}; color: {TEXT_SUB}; border: 1px solid {BORDER_COLOR}; "
                f"padding: 4px 10px; border-radius: 4px;"
            )
            self.circle_lbl.setText("•")
            self.circle_lbl.setStyleSheet(
                f"color: {TEXT_SUB}; background: {BG_SOFT}; "
                f"border: 8px solid {BORDER_COLOR}; border-radius: 46px;"
            )
            self.lbl_ontarget.setText("NO DATA")
            self.lbl_ontarget.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
            self.lbl_steady.setText("Connect serial to track")
            self._update_guidance_buttons(0.0, 0.0)
            return

        try:
            port = self.port_input.text().strip()
            if not port:
                raise ValueError("Please enter the serial port.")
            baud = int(self.baud_input.text().strip())

            # 115200 is the recommended default for the high-rate telemetry
            # stream shown in the reference serial sketch. The timeout is kept
            # very small so the GUI never waits on an incomplete line.
            self.serial_port = serial.Serial(port, baud, timeout=0.01)
            self.serial_port.reset_input_buffer()

            self.live_data = []
            self.lbl_session_file.setText("Recording live telemetry…")
            self.is_serial_active = True

            # Enter target-capture state before the first C is transmitted.
            self.device_status = "C"
            self._send_device_status()
            self.device_status_timer.start()

            self.serial_timer.start()

            self.btn_connect.setText("Disconnect")
            self.btn_connect.setStyleSheet(
                f"background-color: {ERROR_RED}; color: white; font-weight: bold; "
                f"padding: 8px; border-radius: 4px; border: none;"
            )
            self.lbl_status.setText("●  Device Connected")
            self.lbl_status.setStyleSheet(f"color: {SUCCESS_GREEN}; border: none;")
            self.lbl_telemetry_badge.setText("●  TARGET CAPTURE")
            self.lbl_telemetry_badge.setStyleSheet(
                f"background: #EFF6FF; color: {ACCENT_BLUE}; "
                f"border: 1px solid #CFE1FF; padding: 5px 9px; border-radius: 9px;"
            )

            # Start the 10-second target acquisition immediately after the
            # input buffer has been cleared, so stale samples cannot affect
            # the target average.
            self._start_target_capture()

        except ValueError as e:
            QMessageBox.warning(self, "Connection Error", str(e))
            self.serial_port = None
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))
            self.serial_port = None

    def _start_target_capture(self):
        self.target_capture_active = True
        self.target_capture_samples = []
        self.target_capture_start = time.perf_counter()

        dialog = QDialog(self)
        dialog.setWindowTitle("Set Needle Target")
        dialog.setModal(False)
        dialog.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        dialog.setFixedSize(430, 190)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)

        title = QLabel("Set Target for Needle")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")
        title.setAlignment(Qt.AlignCenter)

        message = QLabel(
            "Place the needle at the desired target position and keep it steady.\n"
            "The system will collect real-time orientation data for 10 seconds.\n"
            "Device status output: C while capturing."
        )
        message.setWordWrap(True)
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet(f"color: {TEXT_SUB}; font-size: 11px; border: none;")

        self.lbl_target_countdown = QLabel("10.0 s remaining")
        self.lbl_target_countdown.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.lbl_target_countdown.setStyleSheet(f"color: {ACCENT_BLUE}; border: none;")
        self.lbl_target_countdown.setAlignment(Qt.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(message)
        layout.addWidget(self.lbl_target_countdown)

        self.target_capture_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self.target_capture_timer.start()

    def _target_capture_tick(self):
        if not self.target_capture_active:
            return

        elapsed = time.perf_counter() - self.target_capture_start
        remaining = max(0.0, 10.0 - elapsed)
        if hasattr(self, "lbl_target_countdown") and self.lbl_target_countdown:
            self.lbl_target_countdown.setText(f"{remaining:.1f} s remaining")

        if elapsed >= 10.0:
            self._finish_target_capture()

    def _stop_target_capture(self):
        self.target_capture_timer.stop()
        self.target_capture_active = False
        self.target_capture_start = None
        if self.target_capture_dialog is not None:
            try:
                self.target_capture_dialog.close()
                self.target_capture_dialog.deleteLater()
            except Exception:
                pass
            self.target_capture_dialog = None

    def _finish_target_capture(self):
        self.target_capture_timer.stop()
        self.target_capture_active = False

        samples = np.asarray(self.target_capture_samples, dtype=float)
        self.target_capture_start = None

        if self.target_capture_dialog is not None:
            try:
                self.target_capture_dialog.close()
                self.target_capture_dialog.deleteLater()
            except Exception:
                pass
            self.target_capture_dialog = None

        if samples.size == 0:
            self.lbl_telemetry_badge.setText("●  NO TARGET DATA")
            self.lbl_telemetry_badge.setStyleSheet(
                f"background: {ERROR_BG}; color: {ERROR_RED}; border: 1px solid #F5B6B6; "
                f"padding: 5px 9px; border-radius: 9px;"
            )
            QMessageBox.warning(
                self,
                "Target Capture Failed",
                "No valid pitch, yaw, roll samples were received during the 10-second target capture."
            )
            return

        # Arithmetic mean of the complete 10-second sample set, exactly as
        # requested. All telemetry values are treated as degrees.
        avg_pitch, avg_yaw, avg_roll = np.mean(samples, axis=0)
        self.target_pitch = float(avg_pitch)
        self.target_yaw = float(avg_yaw)
        self.target_roll = float(avg_roll)

        # Keep the 3D target vector synchronized with the orientation panel.
        self.canvas3d.set_target_direction(self.target_pitch, self.target_yaw)

        self.lbl_tar_pitch.setText(f"{self.target_pitch:.1f}°")
        self.lbl_tar_yaw.setText(f"{self.target_yaw:.1f}°")
        self.lbl_tar_roll.setText(f"{self.target_roll:.1f}°")

        # Recalculate the current/error display against the newly captured target.
        self.update_telemetry(self._last_pitch, self._last_yaw, self._last_roll)

        self.lbl_status.setText(
            f"●  Target Set — 10 s average from {len(samples)} samples"
        )
        self.lbl_status.setStyleSheet(f"color: {SUCCESS_GREEN}; border: none;")
        self.lbl_telemetry_badge.setText("●  LIVE TELEMETRY")
        self.lbl_telemetry_badge.setStyleSheet(
            f"background: {SUCCESS_BG}; color: {SUCCESS_GREEN}; "
            f"border: 1px solid #BBE7D0; padding: 5px 9px; border-radius: 9px;"
        )

        # Target capture is complete. From this point onward the MCU receives
        # G when the complete orientation is inside tolerance, otherwise R.
        self.device_status = self._get_device_status(
            self._last_pitch, self._last_yaw, self._last_roll
        )
        self._send_device_status()

    def _record_live_sample(self, pitch, yaw, roll):
        target_ready = not self.target_capture_active
        row = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "Roll (°)": float(roll),
            "Pitch (°)": float(pitch),
            "Yaw (°)": float(yaw),
            "Target Roll (°)": float(self.target_roll) if target_ready else "",
            "Target Pitch (°)": float(self.target_pitch) if target_ready else "",
            "Target Yaw (°)": float(self.target_yaw) if target_ready else "",
            "Roll Error (°)": float(roll - self.target_roll) if target_ready else "",
            "Pitch Error (°)": float(pitch - self.target_pitch) if target_ready else "",
            "Yaw Error (°)": float(yaw - self.target_yaw) if target_ready else "",
            "Phase": "Live Guidance" if target_ready else "Target Capture",
        }
        self.live_data.append(row)

    def read_serial_data(self):
        """Drain every complete waiting line, keep the newest for the GUI,
        and retain every valid sample for 10-second averaging/export.

        Expected MCU format: pitch,yaw,roll
        """
        if not self.serial_port or not self.is_serial_active:
            return

        latest = None
        try:
            # The reference serial approach uses the input-buffer length and
            # processes all waiting lines in one timer tick. This prevents a
            # high-rate stream from falling behind while still redrawing the
            # 3D view only once per GUI tick.
            while self.serial_port.in_waiting:
                raw = self.serial_port.readline()
                text = raw.decode("utf-8", errors="ignore").strip()
                if not text:
                    continue

                # Ignore partial/malformed lines. A valid telemetry record has
                # exactly two commas: pitch,yaw,roll.
                if text.count(",") != 2:
                    continue

                try:
                    pitch, yaw, roll = (float(v.strip()) for v in text.split(","))
                except (ValueError, TypeError):
                    continue

                # Collect every valid sample during the 10-second target window.
                if self.target_capture_active:
                    self.target_capture_samples.append((pitch, yaw, roll))

                self._record_live_sample(pitch, yaw, roll)
                latest = (pitch, yaw, roll)

            if latest is not None:
                self.update_telemetry(*latest)

        except Exception as e:
            print(f"Serial read error: {e}")

    def _get_device_status(self, pitch, yaw, roll):
        """Return the MCU command for the current orientation."""
        if self.target_capture_active:
            return "C"

        tolerance = 2.0
        errors = (
            abs(float(pitch) - float(self.target_pitch)),
            abs(float(yaw) - float(self.target_yaw)),
            abs(float(roll) - float(self.target_roll)),
        )
        return "G" if all(error <= tolerance for error in errors) else "R"

    def _send_device_status(self):
        """Transmit the current C/R/G state to the connected MCU."""
        if not (self.serial_port and self.is_serial_active):
            return

        command = self._get_device_status(
            self._last_pitch, self._last_yaw, self._last_roll
        )
        self.device_status = command

        try:
            self.serial_port.write(command.encode("ascii"))
            self.serial_port.flush()
        except Exception as e:
            print(f"Serial status write error: {e}")

    def export_csv(self):
        """Export the needle telemetry session using needle-specific columns."""
        if not self.live_data:
            QMessageBox.warning(self, "No Data", "No needle telemetry has been recorded yet.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Needle Telemetry CSV",
            f"needle_session_{ts}.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            import csv
            headers = [
                "Timestamp",
                "Roll (°)",
                "Pitch (°)",
                "Yaw (°)",
                "Target Roll (°)",
                "Target Pitch (°)",
                "Target Yaw (°)",
                "Roll Error (°)",
                "Pitch Error (°)",
                "Yaw Error (°)",
                "Phase",
            ]

            with open(path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for row in self.live_data:
                    writer.writerow({key: row.get(key, "") for key in headers})

            self.lbl_session_file.setText("Telemetry CSV exported successfully")
            QMessageBox.information(self, "Exported", f"✓  Saved needle telemetry to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _style_guidance_button(self, button, active=False):
        """Highlight a guidance direction when that move is required."""
        if active:
            button.setStyleSheet(f"""
                QPushButton {{
                    color: white;
                    background: {ACCENT_BLUE};
                    border: 1px solid {ACCENT_BLUE};
                    border-radius: 10px;
                    font-size: 12px;
                    font-weight: bold;
                    padding: 9px 14px;
                    min-width: 78px;
                    min-height: 44px;
                }}
            """)
        else:
            button.setStyleSheet(f"""
                QPushButton {{
                    color: {TEXT_SUB};
                    background: {BG_CARD};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 10px;
                    font-size: 11px;
                    font-weight: bold;
                    padding: 7px 12px;
                    min-width: 66px;
                    min-height: 38px;
                }}
            """)

    def _update_guidance_buttons(self, err_pitch, err_yaw):
        """
        Show which way the operator should adjust the orientation.

        Pitch:
          current below target  -> Move Up
          current above target  -> Move Down

        Yaw:
          current negative       -> Move Right
          current positive       -> Move Left
        """
        tolerance = 2.0

        # These indicators are driven directly by live telemetry:
        # Pitch below target -> UP
        # Pitch above target -> DOWN
        # Yaw below target   -> RIGHT
        # Yaw above target   -> LEFT
        need_up = err_pitch < -tolerance
        need_down = err_pitch > tolerance
        need_right = err_yaw < -tolerance
        need_left = err_yaw > tolerance

        self._style_guidance_button(self.btn_up, need_up)
        self._style_guidance_button(self.btn_down, need_down)
        self._style_guidance_button(self.btn_left, need_left)
        self._style_guidance_button(self.btn_right, need_right)

    def update_telemetry(self, pitch, yaw, roll):
        # Store raw telemetry values in degrees.
        self._last_pitch, self._last_yaw, self._last_roll = pitch, yaw, roll

        # 1. Update 3D Visualizer (always in degrees internally)
        self.canvas3d.update_needle(pitch, yaw)

        # 2. Calculate Exact Errors (degrees)
        err_pitch = pitch - self.target_pitch
        err_yaw = yaw - self.target_yaw
        err_roll = roll - self.target_roll

        # 3. Update text values in degrees.
        self.lbl_curr_pitch.setText(self._fmt_angle(pitch))
        self.lbl_curr_yaw.setText(self._fmt_angle(yaw))
        self.lbl_curr_roll.setText(self._fmt_angle(roll))

        self._format_error(self.lbl_err_pitch, err_pitch)
        self._format_error(self.lbl_err_yaw, err_yaw)
        self._format_error(self.lbl_err_roll, err_roll)

        # 4. Status Bullseye Logic
        tolerance = 2.0
        is_pitch_ok = abs(err_pitch) <= tolerance
        is_yaw_ok = abs(err_yaw) <= tolerance
        is_roll_ok = abs(err_roll) <= tolerance

        # Update the MCU command state. During the 10-second capture this is
        # always C. After capture it becomes G only when all three axes are
        # inside the ±2° tolerance; otherwise it is R.
        if self.target_capture_active:
            self.device_status = "C"
        else:
            self.device_status = "G" if (is_pitch_ok and is_yaw_ok and is_roll_ok) else "R"

        # Update directional guidance indicators in real time.
        self._update_guidance_buttons(err_pitch, err_yaw)

        if is_pitch_ok and is_yaw_ok:
            self.lbl_range.setText("✔ WITHIN RANGE")
            self.lbl_range.setStyleSheet(
                f"background: {SUCCESS_BG}; color: {SUCCESS_GREEN}; "
                f"font-weight: bold; padding: 6px 10px; border-radius: 9px; "
                f"border: 1px solid #B7E6C9;"
            )
            self.circle_lbl.setText("✓")
            self.circle_lbl.setStyleSheet(
                f"color: {SUCCESS_GREEN}; background: {SUCCESS_BG}; "
                f"border: 8px solid #BEE8CE; border-radius: 46px;"
            )
            self.lbl_ontarget.setText("ON TARGET")
            self.lbl_ontarget.setStyleSheet(f"color: {SUCCESS_GREEN}; border: none;")
            self.lbl_steady.setText("Keep the needle steady")
        else:
            self.lbl_range.setText("! ADJUSTING")
            self.lbl_range.setStyleSheet(
                f"background: {ERROR_BG}; color: {ERROR_RED}; "
                f"font-weight: bold; padding: 6px 10px; border-radius: 9px; "
                f"border: 1px solid #F5B6B6;"
            )
            self.circle_lbl.setText("!")
            self.circle_lbl.setStyleSheet(
                f"color: {ERROR_RED}; background: {ERROR_BG}; "
                f"border: 8px solid #F1B1B1; border-radius: 46px;"
            )
            self.lbl_ontarget.setText("OFF TARGET")
            self.lbl_ontarget.setStyleSheet(f"color: {ERROR_RED}; border: none;")
            self.lbl_steady.setText("Align with target direction")

    def _fmt_angle(self, deg_value):
        return f"{deg_value:.1f}°"

    def _format_error(self, label, error_deg):
        """Format the error in degrees and color it by tolerance state."""
        sign = "+" if error_deg > 0 else ""
        label.setText(f"{sign}{error_deg:.1f}°")

        text_color = SUCCESS_GREEN if abs(error_deg) <= 2.0 else ERROR_RED
        label.setStyleSheet(
            f"color: {text_color}; background: {BG_SOFT}; "
            f"border: 1px solid {BORDER_COLOR}; border-radius: 7px; "
            f"padding: 6px 8px; font-size: 19px; font-weight: bold;"
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NeedleMainWindow()
    window.show()
    sys.exit(app.exec_())
    