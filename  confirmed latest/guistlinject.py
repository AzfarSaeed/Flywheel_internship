import sys
import time
import re
import struct
from pathlib import Path
import numpy as np
import serial
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QFrame, QGridLayout, QPushButton,
                             QSizePolicy, QLineEdit, QCheckBox, QGraphicsDropShadowEffect,
                             QFileDialog, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
#  ANGLE & 3D GEOMETRY HELPERS (pure numpy)
# ============================================
def wrap_angle_deg(angle):
    """Wrap any angle in degrees into [-180.0, 180.0)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def angle_delta_deg(current, target):
    """Shortest signed angular error (current - target) in [-180.0, 180.0)."""
    return (float(current) - float(target) + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles_deg):
    """Circular mean of angles in degrees, safe across the -180/+180 wrap."""
    rad = np.radians(np.asarray(angles_deg, dtype=float))
    mean_sin = float(np.mean(np.sin(rad)))
    mean_cos = float(np.mean(np.cos(rad)))
    if abs(mean_sin) < 1e-12 and abs(mean_cos) < 1e-12:
        return float(np.mean(angles_deg))
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)))


def spherical_to_cartesian(pitch_deg, yaw_deg, length=1.0):
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)
    x = length * np.cos(pitch_rad) * np.sin(yaw_rad)
    y = length * np.cos(pitch_rad) * np.cos(yaw_rad)
    z = length * np.sin(pitch_rad)
    return np.array([x, y, z], dtype=float)


def build_needle_frame(pitch_deg, yaw_deg, roll_deg=0.0):
    """Build a smooth, singularity-free 3x3 orthonormal basis R = [e1, e2, d]
    for the needle at (pitch_deg, yaw_deg, roll_deg).

    * Column 2 (`d`): unit forward direction along the needle shaft.
    * Column 0 (`e1`): local 'right' axis, rotated by `roll_deg` around `d`.
    * Column 1 (`e2`): local 'up / bevel' axis, rotated by `roll_deg` around `d`.
    """
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)
    roll_rad = np.radians(roll_deg)

    # Forward unit vector along the needle shaft.
    d = np.array([
        np.cos(pitch_rad) * np.sin(yaw_rad),
        np.cos(pitch_rad) * np.cos(yaw_rad),
        np.sin(pitch_rad),
    ], dtype=float)

    # Smooth horizontal right vector (well-defined for all pitch/yaw, including
    # vertical pitch = +-90 deg because yaw_rad remains continuous).
    right0 = np.array([
        np.cos(yaw_rad),
        -np.sin(yaw_rad),
        0.0,
    ], dtype=float)

    # Local 'up' vector perpendicular to both forward and right0.
    up0 = np.cross(d, right0)
    up_norm = np.linalg.norm(up0)
    if up_norm > 1e-9:
        up0 /= up_norm
    else:
        up0 = np.array([0.0, 0.0, 1.0], dtype=float)

    # Apply intrinsic roll rotation around the needle shaft axis `d`.
    cos_r = np.cos(roll_rad)
    sin_r = np.sin(roll_rad)
    e1 = cos_r * right0 + sin_r * up0
    e2 = -sin_r * right0 + cos_r * up0

    return np.column_stack((e1, e2, d))


def _tapered_tube_mesh(start, basis_matrix, r_start, r_end, height, n=18, bevel_slope=0.0):
    """Build a cylinder/cone segment along `basis_matrix[:, 2]` using the full
    3D orientation frame `basis_matrix` so roll rotates the mesh cleanly.
    `bevel_slope` optionally shears the tip end to create a visible surgical bevel.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    s = np.array([0.0, 1.0])
    theta_grid, s_grid = np.meshgrid(theta, s)

    r_grid = r_start + (r_end - r_start) * s_grid
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)

    # Optional bevel cut at the distal end so roll is clearly visible on the tip.
    z_grid = s_grid * height
    if bevel_slope != 0.0:
        z_grid = z_grid + (s_grid * bevel_slope * r_start * np.sin(theta_grid))

    pts = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1)
    pts_rot = pts @ basis_matrix.T + np.asarray(start, dtype=float)
    X = pts_rot[:, 0].reshape(x_grid.shape)
    Y = pts_rot[:, 1].reshape(x_grid.shape)
    Z = pts_rot[:, 2].reshape(x_grid.shape)
    return X, Y, Z


def _roll_arc_points(start_Point, basis_matrix, roll_start_deg, roll_end_deg, radius=0.19, samples=28):
    """Return a 3D circular arc in the needle's transverse (roll) plane."""
    delta = angle_delta_deg(roll_end_deg, roll_start_deg)
    if abs(delta) < 0.3:
        return np.empty((0, 3))

    angles = np.radians(roll_start_deg + np.linspace(0.0, delta, samples))
    right0 = basis_matrix[:, 0]
    up0 = basis_matrix[:, 1]
    pts = (
        np.asarray(start_Point, dtype=float)[None, :]
        + radius * (np.cos(angles)[:, None] * right0[None, :] + np.sin(angles)[:, None] * up0[None, :])
    )
    return pts


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

    if np.sin(angle) < 1e-6:
        helper = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(a, helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        perpendicular = np.cross(a, helper)
        perpendicular /= np.linalg.norm(perpendicular)
        ts = np.linspace(0.0, angle, samples)
        pts = [radius * (np.cos(t) * a + np.sin(t) * perpendicular) for t in ts]
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
    """Real-time 3D direction + 360-degree roll view with smooth easing."""

    INITIAL_PITCH_DEG = 0.0          # Start level with the horizontal plane
    SMOOTHING_TIME_CONSTANT = 0.065  # seconds; responsive & silky-smooth at 60 Hz
    FRAME_INTERVAL_MS = 16           # nominal ~60 Hz animation timer

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6.5, 6.0), dpi=100)
        self.fig.patch.set_facecolor(BG_CARD)
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.ax = self.fig.add_subplot(111, projection="3d")
        super().__init__(self.fig)
        self.setParent(parent)

        self.target_pitch = 30.0
        self.target_yaw = 0.0
        self.target_roll = 0.0
        self.show_target = True

        # Most recent raw orientation requested by the serial-data handler.
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.last_roll = 0.0
        # Start level at 0 degrees until the first real sample arrives.
        # Raw/live sensor values are not offset or modified.
        self._desired_pitch = self.INITIAL_PITCH_DEG
        self._desired_yaw = 0.0
        self._desired_roll = 0.0

        # Independently animated orientation used for the 3D drawing.
        self._display_pitch = self.INITIAL_PITCH_DEG
        self._display_yaw = 0.0
        self._display_roll = 0.0
        self._last_frame_time = time.perf_counter()
        self._dynamic_artists = []
        self._force_render = True

        # STL model setup is isolated from live telemetry and animation state.
        self.is_stl_loaded = False
        self.stl_vectors = np.zeros((0, 3, 3), dtype=float)
        self.stl_normals = np.zeros((0, 3), dtype=float)
        self.stl_triangle_count = 0
        self.stl_path = None
        self.stl_load_error = "STL has not been loaded yet."
        self.light_dir = np.array([0.5, 0.5, 1.0], dtype=float)
        self.light_dir /= np.linalg.norm(self.light_dir)
        self._load_custom_stl()

        self._setup_scene()
        self._render_needle(
            self._display_pitch, self._display_yaw, self._display_roll
        )

        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(self.FRAME_INTERVAL_MS)
        self._animation_timer.timeout.connect(self._animate_needle)
        self._animation_timer.start()

    @staticmethod
    def _read_stl_triangles(path):
        """Read binary or ASCII STL triangles without requiring numpy-stl."""
        raw = Path(path).read_bytes()

        # Binary STL: 80-byte header, uint32 triangle count, then 50 bytes/face.
        if len(raw) >= 84:
            triangle_count = struct.unpack("<I", raw[80:84])[0]
            if triangle_count > 0 and 84 + triangle_count * 50 == len(raw):
                dtype = np.dtype([
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ])
                records = np.frombuffer(raw, dtype=dtype, offset=84, count=triangle_count)
                return records["vertices"].astype(float).copy()

        # ASCII STL: collect groups of three "vertex x y z" lines.
        text = raw.decode("utf-8", errors="ignore")
        number = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
        pattern = re.compile(
            rf"\bvertex\s+({number})\s+({number})\s+({number})",
            flags=re.IGNORECASE,
        )
        vertices = pattern.findall(text)
        if not vertices or len(vertices) % 3:
            raise ValueError("File is not a valid binary or ASCII STL mesh.")
        return np.asarray(vertices, dtype=float).reshape(-1, 3, 3)

    def _load_custom_stl(self, filename="Injection_1.stl", limit=5000):
        """Load, center, scale, and orient the injection STL for local +Z forward."""
        folder = Path(__file__).resolve().parent
        candidates = (
            filename,
            "Injection_1(3).stl",
            "Injection_1(2).stl",
            "Injection_1(1).stl",
        )
        path = next((folder / name for name in candidates if (folder / name).is_file()), None)
        if path is None:
            self.stl_load_error = (
                "STL file not found. Put Injection_1.stl in the same folder as this script."
            )
            print(f"[STL] {self.stl_load_error}")
            return

        try:
            triangles = self._read_stl_triangles(path)
            triangles = np.asarray(triangles, dtype=float)
            triangles = triangles[np.all(np.isfinite(triangles), axis=(1, 2))]
            if len(triangles) == 0:
                raise ValueError("STL contains no valid triangles.")

            # Center the complete mesh and normalize its longest dimension.
            flat = triangles.reshape(-1, 3)
            minimum = flat.min(axis=0)
            maximum = flat.max(axis=0)
            triangles -= ((minimum + maximum) / 2.0)[None, None, :]
            max_dimension = float(np.ptp(triangles.reshape(-1, 3), axis=0).max())
            if max_dimension <= 1e-12:
                raise ValueError("STL mesh has zero size.")
            triangles *= 1.1 / max_dimension

            # The provided model's longest axis is Y. Map it to local +Z,
            # which is the forward axis consumed by build_needle_frame().
            rotate_x_minus_90 = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ])
            triangles = triangles @ rotate_x_minus_90.T

            # Keep modest models intact; cap only very large meshes for GUI speed.
            if len(triangles) > limit:
                cross = np.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                )
                area = np.linalg.norm(cross, axis=1)
                keep = np.sort(np.argpartition(area, len(area) - limit)[len(area) - limit:])
                triangles = triangles[keep]

            normals = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            lengths[lengths == 0.0] = 1.0
            normals /= lengths

            self.stl_vectors = triangles
            self.stl_normals = normals
            self.stl_triangle_count = len(triangles)
            self.stl_path = path
            self.is_stl_loaded = True
            self.stl_load_error = ""
            print(f"[STL] Successfully loaded {path.name} ({len(triangles)} triangles).")
        except Exception as exc:
            self.stl_load_error = f"Could not load {path.name}: {exc}"
            print(f"[STL] {self.stl_load_error}")

    def _setup_scene(self):
        """Draw the fixed coordinate frame and floor once, not per sample."""
        self.ax.cla()
        self.ax.set_facecolor("#081725")
        self.ax.set_axis_off()
        self.ax.set_xlim([-1.40, 1.40])
        self.ax.set_ylim([-1.40, 1.40])
        # Include space below the horizontal plane so a downward-pointing
        # needle is fully visible instead of being clipped by the 3D axes.
        self.ax.set_zlim([-1.15, 1.62])
        self.ax.set_box_aspect([1.10, 1.10, 0.95])
        self.ax.view_init(elev=45, azim=-90)  # Elevated 45-degree aerial view; +X remains to the right

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
        """Shortest signed angular difference (target - current) in [-180, 180)."""
        return (float(target) - float(current) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _wrap_angle(angle):
        return (float(angle) + 180.0) % 360.0 - 180.0

    def set_show_target(self, visible):
        self.show_target = bool(visible)
        self._force_render = True
        self._render_needle(self._display_pitch, self._display_yaw, self._display_roll)
        self.draw_idle()

    def set_target_direction(self, pitch, yaw, roll=None):
        """Refresh the displayed target (pitch, yaw, and roll)."""
        self.target_pitch = float(pitch)
        self.target_yaw = float(yaw)
        if roll is not None:
            self.target_roll = float(roll)
        self._force_render = True
        self._render_needle(self._display_pitch, self._display_yaw, self._display_roll)
        self.draw_idle()

    def update_needle(self, current_pitch, current_yaw, current_roll=0.0):
        """Accept the newest raw sample (pitch, yaw, roll); animation eases smoothly."""
        try:
            pitch = float(current_pitch)
            yaw = float(current_yaw)
            roll = float(current_roll)
        except (TypeError, ValueError):
            return
        if not (np.isfinite(pitch) and np.isfinite(yaw) and np.isfinite(roll)):
            return

        self.last_pitch = pitch
        self.last_yaw = yaw
        self.last_roll = roll

        self._desired_pitch = pitch
        self._desired_yaw = self._wrap_angle(yaw)
        self._desired_roll = self._wrap_angle(roll)

    def _animate_needle(self):
        """Advance toward the latest orientation using frame-rate-independent easing."""
        now = time.perf_counter()
        dt = max(0.0, min(now - self._last_frame_time, 0.10))
        self._last_frame_time = now
        if dt <= 0.0:
            return

        previous_pitch = self._display_pitch
        previous_yaw = self._display_yaw
        previous_roll = self._display_roll

        tau = max(0.001, float(self.SMOOTHING_TIME_CONSTANT))
        alpha = 1.0 - np.exp(-dt / tau)

        self._display_pitch += alpha * (self._desired_pitch - self._display_pitch)

        yaw_delta = self._angle_delta(self._desired_yaw, self._display_yaw)
        self._display_yaw = self._wrap_angle(self._display_yaw + alpha * yaw_delta)

        roll_delta = self._angle_delta(self._desired_roll, self._display_roll)
        self._display_roll = self._wrap_angle(self._display_roll + alpha * roll_delta)

        # Snap tiny residuals to the destination so the needle settles cleanly.
        if abs(self._desired_pitch - self._display_pitch) < 0.01:
            self._display_pitch = self._desired_pitch
        if abs(self._angle_delta(self._desired_yaw, self._display_yaw)) < 0.01:
            self._display_yaw = self._desired_yaw
        if abs(self._angle_delta(self._desired_roll, self._display_roll)) < 0.01:
            self._display_roll = self._desired_roll

        pitch_moved = abs(self._display_pitch - previous_pitch) >= 0.015
        yaw_moved = abs(self._angle_delta(self._display_yaw, previous_yaw)) >= 0.015
        roll_moved = abs(self._angle_delta(self._display_roll, previous_roll)) >= 0.015

        if self._force_render or pitch_moved or yaw_moved or roll_moved:
            self._render_needle(self._display_pitch, self._display_yaw, self._display_roll)
            self.draw_idle()
            self._force_render = False

    def _render_needle(self, current_pitch, current_yaw, current_roll=0.0):
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

        # Build the needle's unrolled reference frame (roll=0) and rolled frame (roll=current_roll).
        unrolled_basis = build_needle_frame(current_pitch, current_yaw, 0.0)
        rolled_basis = build_needle_frame(current_pitch, current_yaw, current_roll)

        roll_right = rolled_basis[:, 0]   # rotates 360 deg with roll
        roll_up = rolled_basis[:, 1]      # bevel / fin direction (rotates 360 deg with roll)
        current_dir = rolled_basis[:, 2]  # needle shaft axis

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

        # Physical needle model: render the supplied STL with the same live
        # orientation frame; use the pre-existing geometry only if loading fails.
        tail_len = 0.18
        shaft_len = 0.88
        tip_len = 0.25
        tail_start = base - current_dir * tail_len
        tip_start = base + current_dir * shaft_len

        if self.is_stl_loaded:
            # The centered STL is 1.1 units long. This offset keeps its wider
            # hub close to the existing base and its tip pointing forward.
            stl_position = base + current_dir * 0.37
            rotated_triangles = self.stl_vectors @ rolled_basis.T + stl_position
            rotated_normals = self.stl_normals @ rolled_basis.T
            brightness = np.clip(rotated_normals @ self.light_dir, 0.35, 1.0)
            needle_tint = np.array([0.58, 0.64, 0.72])
            facecolors = np.clip(brightness[:, None] * needle_tint[None, :], 0.0, 1.0)

            stl_artist = Poly3DCollection(
                rotated_triangles, linewidths=0.0, antialiaseds=False
            )
            stl_artist.set_facecolors(facecolors)
            stl_artist.set_edgecolor("none")
            stl_artist.set_clip_on(False)
            self.ax.add_collection3d(stl_artist)
            self._dynamic_artists.append(stl_artist)
        else:
            # Original generated geometry: left unchanged as fallback.
            X, Y, Z = _tapered_tube_mesh(
                tail_start, rolled_basis, r_start=0.074, r_end=0.054,
                height=tail_len, n=7
            )
            self._dynamic_artists.append(self.ax.plot_surface(
                X, Y, Z, color="#7C8FA3", alpha=1.0, linewidth=0.55,
                edgecolor="#CBD5E1", shade=True, antialiased=True
            ))

            X, Y, Z = _tapered_tube_mesh(
                base, rolled_basis, r_start=0.034, r_end=0.030,
                height=shaft_len, n=24
            )
            self._dynamic_artists.append(self.ax.plot_surface(
                X, Y, Z, color="#F8FAFC", alpha=1.0, linewidth=0.22,
                edgecolor="#94A3B8", shade=True, antialiased=True
            ))

            X, Y, Z = _tapered_tube_mesh(
                tip_start, rolled_basis, r_start=0.068, r_end=0.0,
                height=tip_len, n=24, bevel_slope=-0.85
            )
            self._dynamic_artists.append(self.ax.plot_surface(
                X, Y, Z, color="#0EA5E9", alpha=1.0, linewidth=0,
                shade=True, antialiased=True
            ))

        # ── 360° ROLL VISUALIZATION FEATURES ─────────────────────────────────
        # A. Longitudinal high-contrast roll stripe along the top ("bevel up") of the shaft & hub
        stripe_start = tail_start + roll_up * 0.076
        stripe_mid = base + roll_up * 0.030
        stripe_end = tip_start + roll_up * 0.069
        self._track_plot(
            [stripe_start[0], stripe_mid[0], stripe_end[0]],
            [stripe_start[1], stripe_mid[1], stripe_end[1]],
            [stripe_start[2], stripe_mid[2], stripe_end[2]],
            color="#F59E0B", linewidth=3.0, alpha=0.98
        )

        # B. Asymmetric hub wings + prominent roll fin so full 360° rotation is unmistakable
        hub_center = base - current_dir * (0.45 * tail_len)
        wing_span = 0.14
        wing_left = hub_center - roll_right * wing_span
        wing_right = hub_center + roll_right * wing_span
        self._track_plot(
            [wing_left[0], wing_right[0]],
            [wing_left[1], wing_right[1]],
            [wing_left[2], wing_right[2]],
            color="#94A3B8", linewidth=3.4, alpha=0.95
        )

        # 3D filled Roll Fin on the hub pointing along +roll_up (0° reference / bevel direction)
        fin_base_rear = tail_start + roll_up * 0.070
        fin_base_front = base + roll_up * 0.050
        fin_peak = hub_center + roll_up * 0.21
        X_fin = np.array([
            [fin_base_rear[0], fin_base_front[0]],
            [fin_peak[0], fin_peak[0]],
        ])
        Y_fin = np.array([
            [fin_base_rear[1], fin_base_front[1]],
            [fin_peak[1], fin_peak[1]],
        ])
        Z_fin = np.array([
            [fin_base_rear[2], fin_base_front[2]],
            [fin_peak[2], fin_peak[2]],
        ])
        self._dynamic_artists.append(self.ax.plot_surface(
            X_fin, Y_fin, Z_fin, color="#F59E0B", alpha=0.95,
            shade=False, linewidth=0.8, edgecolor="#FDE68A"
        ))

        # C. 360° Roll Dial Ring around the needle shaft near the base
        dial_center = base + current_dir * 0.14
        dial_radius = 0.21
        ring_angles = np.linspace(0.0, 2.0 * np.pi, 72)
        ring_pts = (
            dial_center[None, :]
            + dial_radius * (
                np.cos(ring_angles)[:, None] * unrolled_basis[:, 0][None, :]
                + np.sin(ring_angles)[:, None] * unrolled_basis[:, 1][None, :]
            )
        )
        self._track_plot(
            ring_pts[:, 0], ring_pts[:, 1], ring_pts[:, 2],
            color="#38BDF8", linewidth=1.3, linestyle=":", alpha=0.72
        )

        # Current roll pointer on the 360° dial ring
        curr_roll_tip = dial_center + roll_up * dial_radius
        self._track_plot(
            [dial_center[0], curr_roll_tip[0]],
            [dial_center[1], curr_roll_tip[1]],
            [dial_center[2], curr_roll_tip[2]],
            color="#F59E0B", linewidth=2.8, alpha=0.98
        )
        self._track_plot(
            [curr_roll_tip[0]], [curr_roll_tip[1]], [curr_roll_tip[2]],
            marker="o", markersize=6, color="#F59E0B"
        )

        # Target roll pointer and shortest-path roll error arc on the dial ring
        if self.show_target:
            target_roll_rad = np.radians(self.target_roll)
            target_roll_vec = (
                -np.sin(target_roll_rad) * unrolled_basis[:, 0]
                + np.cos(target_roll_rad) * unrolled_basis[:, 1]
            )
            target_roll_tip = dial_center + target_roll_vec * dial_radius
            self._track_plot(
                [dial_center[0], target_roll_tip[0]],
                [dial_center[1], target_roll_tip[1]],
                [dial_center[2], target_roll_tip[2]],
                color="#39E85F", linestyle="--", linewidth=2.2, alpha=0.95
            )
            self._track_plot(
                [target_roll_tip[0]], [target_roll_tip[1]], [target_roll_tip[2]],
                marker="o", markersize=5, color="#39E85F"
            )

            # Roll error arc between target roll and current roll (shifted +90 deg so 0 deg is at +up0)
            roll_arc = _roll_arc_points(
                dial_center, unrolled_basis,
                90.0 + self.target_roll, 90.0 + current_roll,
                radius=dial_radius * 0.82, samples=28
            )
            if len(roll_arc):
                self._track_plot(
                    roll_arc[:, 0], roll_arc[:, 1], roll_arc[:, 2],
                    color="#F59E0B", linewidth=2.2, alpha=0.90
                )

        # Current Direction + Live Roll readout label in 3D space.
        lateral = np.array([-current_dir[1], current_dir[0], 0.0])
        lateral_norm = np.linalg.norm(lateral)
        if lateral_norm > 1e-8:
            lateral /= lateral_norm
        current_label = current_end + lateral * 0.10 + np.array([0.0, 0.0, 0.07])
        display_roll_360 = (current_roll % 360.0 + 360.0) % 360.0
        self._dynamic_artists.append(self.ax.text(
            current_label[0], current_label[1], current_label[2],
            f"Current Direction\nRoll: {current_roll:+.1f}° ({display_roll_360:.0f}°)",
            color="#29B6F6", fontsize=9.5, fontweight="bold"
        ))

        # Angular difference arc between current and target direction vectors.
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
        self.setStyleSheet(
            f"background-color: {BG_MAIN}; color: {TEXT_MAIN}; "
            f"font-family: 'Segoe UI', Arial, sans-serif;"
        )

        # Serial config
        self.serial_port = None
        self.is_serial_active = False

        # Visual-only mock test state. Mock samples never enter the live CSV or serial port.
        self.is_mock_active = False
        self.mock_start_time = None
        self._mock_saved_orientation = (0.0, 0.0, 0.0)

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
        self.device_status = "R"
        self.device_status_timer = QTimer()
        self.device_status_timer.setInterval(100)  # 10 status messages/second
        self.device_status_timer.timeout.connect(self._send_device_status)

        self.setup_ui()

        # Serial polling.
        self.serial_timer = QTimer()
        self.serial_timer.setInterval(20)  # UI refresh ~50 Hz; all queued samples are consumed.
        self.serial_timer.timeout.connect(self.read_serial_data)

        # Mock animation uses the exact same Needle3DCanvas update path as live data.
        self.mock_timer = QTimer(self)
        self.mock_timer.setInterval(20)
        self.mock_timer.timeout.connect(self.generate_mock_data)

    # ------------------------------------------------------------
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 14, 16, 14)
        main_layout.setSpacing(12)

        # Header: logo on the left, title/subtitle centered, connection status on the right.
        header_layout = QHBoxLayout()

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

        self.btn_mock_test = QPushButton("▶ Mock Test")
        self.btn_mock_test.setCursor(Qt.PointingHandCursor)
        self.btn_mock_test.setFixedHeight(32)
        self.btn_mock_test.setStyleSheet(
            f"background-color: {ACCENT_BLUE}; color: white; font-weight: bold; "
            f"font-size: 10px; padding: 5px 10px; border-radius: 5px; border: none;"
        )
        self.btn_mock_test.clicked.connect(self.toggle_mock_test)

        self.lbl_stl_status = QLabel("STL CHECKING")
        self.lbl_stl_status.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.lbl_stl_status.setStyleSheet(f"color: {TEXT_SUB}; border: none;")

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
        lbl_legend_roll = QLabel("● Roll Fin / 360° Dial")
        lbl_legend_roll.setStyleSheet(
            "color: #F59E0B; font-weight: bold; font-size: 12px; border: none;"
        )

        view_header.addWidget(lbl_3d)
        view_header.addWidget(self.btn_mock_test)
        view_header.addWidget(self.lbl_stl_status)
        view_header.addStretch()
        view_header.addWidget(self.chk_show_target)
        view_header.addSpacing(12)
        view_header.addWidget(lbl_legend_target)
        view_header.addSpacing(10)
        view_header.addWidget(lbl_legend_curr)
        view_header.addSpacing(10)
        view_header.addWidget(lbl_legend_roll)
        view_layout.addLayout(view_header)

        self.canvas3d = Needle3DCanvas()
        self.canvas3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if self.canvas3d.is_stl_loaded:
            self.lbl_stl_status.setText(f"STL LOADED · {self.canvas3d.stl_triangle_count} TRIANGLES")
            self.lbl_stl_status.setStyleSheet(
                f"color: {SUCCESS_GREEN}; font-weight: bold; font-size: 9px; border: none;"
            )
            self.lbl_stl_status.setToolTip(str(self.canvas3d.stl_path))
        else:
            self.lbl_stl_status.setText("STL NOT LOADED")
            self.lbl_stl_status.setStyleSheet(
                f"color: {ERROR_RED}; font-weight: bold; font-size: 9px; border: none;"
            )
            self.lbl_stl_status.setToolTip(self.canvas3d.stl_load_error)
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

        # Display order: Roll, Pitch, Yaw.
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

        orient_hint = QLabel("Tolerance  ±2.0°  •  Full 360° Roll, Pitch & Yaw tracking")
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
        lbl_guide_sub = QLabel("Real-time directional & roll correction")
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
        self.btn_roll_ccw = QPushButton("↺\nROLL -")
        self.btn_up = QPushButton("↑\nUP")
        self.btn_roll_cw = QPushButton("↻\nROLL +")
        self.btn_left = QPushButton("←  LEFT")
        self.btn_right = QPushButton("RIGHT  →")
        self.btn_down = QPushButton("↓\nDOWN")
        for btn in (
            self.btn_roll_ccw, self.btn_up, self.btn_roll_cw,
            self.btn_left, self.btn_right, self.btn_down
        ):
            btn.setStyleSheet(arrow_button_style)
            btn.setEnabled(False)
        cross = QLabel("•")
        cross.setFont(QFont("Arial", 18, QFont.Bold))
        cross.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        arrows_box.addWidget(self.btn_roll_ccw, 0, 0, alignment=Qt.AlignRight)
        arrows_box.addWidget(self.btn_up, 0, 1, alignment=Qt.AlignCenter)
        arrows_box.addWidget(self.btn_roll_cw, 0, 2, alignment=Qt.AlignLeft)
        arrows_box.addWidget(self.btn_left, 1, 0, alignment=Qt.AlignRight)
        arrows_box.addWidget(cross, 1, 1, alignment=Qt.AlignCenter)
        arrows_box.addWidget(self.btn_right, 1, 2, alignment=Qt.AlignLeft)
        arrows_box.addWidget(self.btn_down, 2, 1, alignment=Qt.AlignCenter)

        indicator_layout.addLayout(status_box, stretch=1)
        indicator_layout.addLayout(arrows_box, stretch=2)
        guide_layout.addLayout(indicator_layout, stretch=1)

        self.lbl_guidance_hint = QLabel(
            "Move or roll in the highlighted direction.  ±2° = within range."
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

        # Populate the dashboard's default readouts without overriding the
        # visualizer's 0-degree startup pose before live telemetry arrives.
        self.update_telemetry(0.0, 0.0, 0.0, update_visualizer=False)
        self._update_guidance_buttons(0.0, 0.0, 0.0)

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

    # ── STL VISUAL MOCK TEST (separate from live telemetry) ─────────────────
    def toggle_mock_test(self):
        if self.is_mock_active:
            self.mock_timer.stop()
            self.is_mock_active = False
            self.mock_start_time = None
            self.canvas3d.update_needle(*self._mock_saved_orientation)
            self.btn_mock_test.setText("▶ Mock Test")
            self.btn_mock_test.setStyleSheet(
                f"background-color: {ACCENT_BLUE}; color: white; font-weight: bold; "
                f"font-size: 10px; padding: 5px 10px; border-radius: 5px; border: none;"
            )
            if self.canvas3d.is_stl_loaded:
                self.lbl_stl_status.setText(
                    f"STL LOADED · {self.canvas3d.stl_triangle_count} TRIANGLES"
                )
                self.lbl_stl_status.setToolTip(str(self.canvas3d.stl_path))
            else:
                self.lbl_stl_status.setText("STL NOT LOADED")
                self.lbl_stl_status.setToolTip(self.canvas3d.stl_load_error)
            return

        if self.is_serial_active:
            QMessageBox.warning(
                self, "Mock Test Unavailable",
                "Disconnect the device before starting Mock Test."
            )
            return

        self._mock_saved_orientation = (
            self.canvas3d.last_pitch,
            self.canvas3d.last_yaw,
            self.canvas3d.last_roll,
        )
        self.mock_start_time = time.perf_counter()
        self.is_mock_active = True
        self.mock_timer.start()
        self.btn_mock_test.setText("■ Stop Mock Test")
        self.btn_mock_test.setStyleSheet(
            f"background-color: {ERROR_RED}; color: white; font-weight: bold; "
            f"font-size: 10px; padding: 5px 10px; border-radius: 5px; border: none;"
        )
        if self.canvas3d.is_stl_loaded:
            self.lbl_stl_status.setText(
                f"STL TESTING · {self.canvas3d.stl_triangle_count} TRIANGLES"
            )
        else:
            self.lbl_stl_status.setText("STL NOT LOADED")
            self.lbl_stl_status.setToolTip(self.canvas3d.stl_load_error)

    def generate_mock_data(self):
        """Animate pitch, yaw, and full 360-degree roll without touching serial/live CSV."""
        if not self.is_mock_active or self.mock_start_time is None:
            return
        elapsed = time.perf_counter() - self.mock_start_time
        pitch = 42.0 * np.sin(2.0 * np.pi * elapsed / 7.0)
        yaw = wrap_angle_deg(elapsed * 42.0)
        roll = wrap_angle_deg(elapsed * 90.0)
        # Directly feed the same visualizer used by read_serial_data().
        self.canvas3d.update_needle(pitch, yaw, roll)

    # ── REAL-TIME SERIAL + AUTOMATIC TARGET CAPTURE ────────────────────────
    def toggle_serial(self):
        if not self.is_serial_active and self.is_mock_active:
            QMessageBox.warning(
                self, "Mock Test Active",
                "Stop Mock Test before connecting the real device."
            )
            return
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
            self._update_guidance_buttons(0.0, 0.0, 0.0)
            return

        try:
            port = self.port_input.text().strip()
            if not port:
                raise ValueError("Please enter the serial port.")
            baud = int(self.baud_input.text().strip())

            self.serial_port = serial.Serial(port, baud, timeout=0.01)
            self.serial_port.reset_input_buffer()

            self.live_data = []
            self.lbl_session_file.setText("Recording live telemetry…")
            self.is_serial_active = True

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

        # Use circular mean for yaw and roll so samples straddling +-180 deg
        # (or 0/360 deg) average accurately without cancellation.
        self.target_pitch = float(np.mean(samples[:, 0]))
        self.target_yaw = circular_mean_deg(samples[:, 1])
        self.target_roll = circular_mean_deg(samples[:, 2])

        # Synchronize the 3D target direction and target roll marker.
        self.canvas3d.set_target_direction(
            self.target_pitch, self.target_yaw, self.target_roll
        )

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
            "Roll Error (°)": float(angle_delta_deg(roll, self.target_roll)) if target_ready else "",
            "Pitch Error (°)": float(pitch - self.target_pitch) if target_ready else "",
            "Yaw Error (°)": float(angle_delta_deg(yaw, self.target_yaw)) if target_ready else "",
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
            while self.serial_port.in_waiting:
                raw = self.serial_port.readline()
                text = raw.decode("utf-8", errors="ignore").strip()
                if not text:
                    continue

                if text.count(",") != 2:
                    continue

                try:
                    pitch, yaw, roll = (float(v.strip()) for v in text.split(","))
                except (ValueError, TypeError):
                    continue

                if not (np.isfinite(pitch) and np.isfinite(yaw) and np.isfinite(roll)):
                    continue

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
            abs(angle_delta_deg(yaw, self.target_yaw)),
            abs(angle_delta_deg(roll, self.target_roll)),
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

    def _update_guidance_buttons(self, err_pitch, err_yaw, err_roll=0.0):
        """Show which way the operator should adjust Pitch, Yaw, and Roll."""
        tolerance = 2.0

        need_up = err_pitch < -tolerance
        need_down = err_pitch > tolerance
        need_right = err_yaw < -tolerance
        need_left = err_yaw > tolerance
        need_roll_cw = err_roll < -tolerance
        need_roll_ccw = err_roll > tolerance

        self._style_guidance_button(self.btn_up, need_up)
        self._style_guidance_button(self.btn_down, need_down)
        self._style_guidance_button(self.btn_left, need_left)
        self._style_guidance_button(self.btn_right, need_right)
        self._style_guidance_button(self.btn_roll_cw, need_roll_cw)
        self._style_guidance_button(self.btn_roll_ccw, need_roll_ccw)

    def update_telemetry(self, pitch, yaw, roll, update_visualizer=True):
        # Store raw telemetry values in degrees.
        self._last_pitch, self._last_yaw, self._last_roll = pitch, yaw, roll

        # 1. Update 3D Visualizer with all 3 axes (pitch, yaw, roll).
        # The startup readout can skip this once so the initial -45-degree pose
        # remains visible until the first real serial sample arrives.
        if update_visualizer:
            self.canvas3d.update_needle(pitch, yaw, roll)

        # 2. Calculate Exact Shortest-Path Errors (degrees, safe across 360° wrap)
        err_pitch = pitch - self.target_pitch
        err_yaw = angle_delta_deg(yaw, self.target_yaw)
        err_roll = angle_delta_deg(roll, self.target_roll)

        # 3. Update text values in degrees (wrapped cleanly to [-180°, +180°] for display).
        self.lbl_curr_pitch.setText(self._fmt_angle(pitch))
        self.lbl_curr_yaw.setText(self._fmt_angle(wrap_angle_deg(yaw)))
        self.lbl_curr_roll.setText(self._fmt_angle(wrap_angle_deg(roll)))

        self._format_error(self.lbl_err_pitch, err_pitch)
        self._format_error(self.lbl_err_yaw, err_yaw)
        self._format_error(self.lbl_err_roll, err_roll)

        # 4. Status Bullseye Logic (checks Pitch, Yaw, AND Roll)
        tolerance = 2.0
        is_pitch_ok = abs(err_pitch) <= tolerance
        is_yaw_ok = abs(err_yaw) <= tolerance
        is_roll_ok = abs(err_roll) <= tolerance

        if self.target_capture_active:
            self.device_status = "C"
        else:
            self.device_status = "G" if (is_pitch_ok and is_yaw_ok and is_roll_ok) else "R"

        # Update directional + roll guidance indicators in real time.
        self._update_guidance_buttons(err_pitch, err_yaw, err_roll)

        if is_pitch_ok and is_yaw_ok and is_roll_ok:
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
            self.lbl_steady.setText("Align direction and roll with target")

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
    