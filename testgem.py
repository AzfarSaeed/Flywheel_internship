import sys
import csv
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import serial

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QGridLayout, QPushButton, QSizePolicy,
    QLineEdit, QCheckBox, QGraphicsDropShadowEffect, QFileDialog,
    QMessageBox, QDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ============================================
#  THEME CONFIGURATION (Clinical Surgical UI)
# ============================================
BG_MAIN = "#EEF2F6"       # crisp clinical slate background
BG_CARD = "#FFFFFF"       # pure white card surface
BG_SOFT = "#F8FAFC"       # subtle inset fill
VIEWPORT_BG = "#06101E"   # deep stereotactic navy-slate
TEXT_MAIN = "#0F172A"     # high-contrast slate-900
TEXT_SUB = "#64748B"      # muted label slate-500
BORDER_COLOR = "#CBD5E1"  # refined hairline border

ACCENT_BLUE = "#2563EB"   # primary UI blue
NEEDLE_BLUE = "#38BDF8"   # current needle cyan-blue
ERROR_RED = "#DC2626"     # out-of-tolerance red
SUCCESS_GREEN = "#16A34A" # on-target green
ROLL_AMBER = "#F59E0B"    # 360 roll indicator amber

SUCCESS_BG = "#ECFDF5"
SUCCESS_BORDER = "#86EFAC"
ERROR_BG = "#FEF2F2"
ERROR_BORDER = "#FCA5A5"
INFO_BG = "#EFF6FF"
INFO_BORDER = "#BFDBFE"


def add_card_shadow(widget, blur=24, y_offset=3, alpha=15):
    """Soft elevation shadow so white cards sit cleanly on the slate background."""
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
    """Build a right-handed, singularity-free 3x3 orthonormal basis R = [e1, e2, d].
    At pitch=0, yaw=0, roll=0:
      e1 (right) = +X, e2 (up/dorsal) = +Z, d (forward) = +Y.
    """
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)
    roll_rad = np.radians(roll_deg)

    d = np.array([
        np.cos(pitch_rad) * np.sin(yaw_rad),
        np.cos(pitch_rad) * np.cos(yaw_rad),
        np.sin(pitch_rad),
    ], dtype=float)

    right0 = np.array([np.cos(yaw_rad), -np.sin(yaw_rad), 0.0], dtype=float)
    up0 = np.cross(right0, d)
    up_norm = np.linalg.norm(up0)
    if up_norm > 1e-9:
        up0 /= up_norm
    else:
        up0 = np.array([0.0, 0.0, 1.0], dtype=float)

    cos_r = np.cos(roll_rad)
    sin_r = np.sin(roll_rad)
    e1 = cos_r * right0 + sin_r * up0
    e2 = -sin_r * right0 + cos_r * up0

    return np.column_stack((e1, e2, d))


def _profiled_tube_mesh(origin, basis_matrix, z_stations, r_stations, n_theta=28, bevel_cut=0.0):
    """Build a smooth multi-ring cylindrical/profiled mesh along the needle axis."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    z_arr = np.asarray(z_stations, dtype=float)
    r_arr = np.asarray(r_stations, dtype=float)

    theta_grid, z_grid = np.meshgrid(theta, z_arr)
    _, r_grid = np.meshgrid(theta, r_arr)

    x_local = r_grid * np.cos(theta_grid)
    y_local = r_grid * np.sin(theta_grid)
    z_local = z_grid.copy()

    if bevel_cut != 0.0 and len(z_arr) >= 2:
        s_norm = (z_grid - z_arr[0]) / max(1e-9, (z_arr[-1] - z_arr[0]))
        z_local += s_norm * bevel_cut * r_arr[0] * np.sin(theta_grid)

    pts = np.stack([x_local.ravel(), y_local.ravel(), z_local.ravel()], axis=1)
    pts_world = pts @ basis_matrix.T + np.asarray(origin, dtype=float)

    shape = x_local.shape
    return (
        pts_world[:, 0].reshape(shape),
        pts_world[:, 1].reshape(shape),
        pts_world[:, 2].reshape(shape),
    )


def _roll_arc_points(center_point, unrolled_basis, roll_start_deg, roll_end_deg, radius=0.18, samples=32):
    """Return a 3D circular arc in the needle's transverse (roll) plane.
    0 deg roll corresponds to dorsal +up0 (unrolled_basis[:, 1]).
    """
    delta = angle_delta_deg(roll_end_deg, roll_start_deg)
    if abs(delta) < 0.3:
        return np.empty((0, 3))

    angles = np.radians(roll_start_deg + np.linspace(0.0, delta, samples))
    right0 = unrolled_basis[:, 0]
    up0 = unrolled_basis[:, 1]
    return (
        np.asarray(center_point, dtype=float)[None, :]
        + radius * (-np.sin(angles)[:, None] * right0[None, :] + np.cos(angles)[:, None] * up0[None, :])
    )


def _direction_arc_points(v1, v2, center=np.zeros(3), radius=0.65, samples=36):
    """Return a smooth great-circle arc between two 3D unit vectors."""
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)

    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.empty((0, 3))

    a /= na
    b /= nb
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    angle = np.arccos(dot)

    if angle < np.radians(0.4):
        return np.empty((0, 3))

    if np.sin(angle) < 1e-6:
        helper = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(a, helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        perpendicular = np.cross(a, helper)
        perpendicular /= np.linalg.norm(perpendicular)
        ts = np.linspace(0.0, angle, samples)
        return np.asarray(center)[None, :] + np.asarray(
            [radius * (np.cos(t) * a + np.sin(t) * perpendicular) for t in ts]
        )

    ts = np.linspace(0.0, 1.0, samples)
    sin_angle = np.sin(angle)
    pts = []
    for t in ts:
        p = (np.sin((1.0 - t) * angle) / sin_angle * a + np.sin(t * angle) / sin_angle * b)
        p /= np.linalg.norm(p)
        pts.append(center + radius * p)
    return np.asarray(pts)


# ============================================
#  3D SURGICAL NEEDLE CANVAS
# ============================================
class Needle3DCanvas(FigureCanvas):
    """Stereotactic 3D surgical needle navigation viewport with smooth easing."""

    SMOOTHING_TIME_CONSTANT = 0.060
    FRAME_INTERVAL_MS = 16
    SPHERE_RADIUS = 1.08

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7.0, 6.2), dpi=100)
        self.fig.patch.set_facecolor(VIEWPORT_BG)
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        self.ax = self.fig.add_subplot(111, projection="3d")
        super().__init__(self.fig)
        self.setParent(parent)

        self.light_source = LightSource(azdeg=315, altdeg=42)
        self.origin = np.array([0.0, 0.0, 0.0], dtype=float)

        self.target_pitch = 30.0
        self.target_yaw = 0.0
        self.target_roll = 0.0
        self.show_target = True

        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.last_roll = 0.0
        self._desired_pitch = 0.0
        self._desired_yaw = 0.0
        self._desired_roll = 0.0

        self._display_pitch = 0.0
        self._display_yaw = 0.0
        self._display_roll = 0.0
        self._last_frame_time = time.perf_counter()
        self._dynamic_artists = []
        self._force_render = True

        self._setup_scene()
        self._render_needle(0.0, 0.0, 0.0)

        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(self.FRAME_INTERVAL_MS)
        self._animation_timer.timeout.connect(self._animate_needle)
        self._animation_timer.start()

    def set_camera_preset(self, preset_name):
        """Switch camera perspective cleanly."""
        presets = {
            "iso": (22, 38),
            "pitch": (8, 0),      # Side view looking along X to inspect Pitch (Y-Z plane)
            "yaw": (88, -90),     # Top-down view looking down Z to inspect Yaw (X-Y plane)
            "roll": (18, -160),   # Rear-axial view looking down needle hub
        }
        elev, azim = presets.get(preset_name, (22, 38))
        self.ax.view_init(elev=elev, azim=azim)
        self._force_render = True
        self._render_needle(self._display_pitch, self._display_yaw, self._display_roll)
        self.draw_idle()

    def _setup_scene(self):
        """Draw a whisper-clean, distortion-free spherical gimbal and equatorial compass."""
        self.ax.cla()
        self.ax.set_facecolor(VIEWPORT_BG)
        self.ax.set_axis_off()

        # Orthographic projection eliminates fish-eye perspective bulging on the sphere
        try:
            self.ax.set_proj_type("ortho")
        except AttributeError:
            pass

        # Tight framing so the needle is large, centered, and easy to read
        lim = 0.90
        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])
        self.ax.set_zlim([-lim, lim])
        self.ax.set_box_aspect([1.0, 1.0, 1.0])
        self.ax.view_init(elev=22, azim=38)

        sphere_r = self.SPHERE_RADIUS
        theta = np.linspace(0.0, 2.0 * np.pi, 140)

        # 1. Whisper-thin Latitude Rings (upper & lower hemispheres, very subtle)
        for lat_deg in (-60, -30, 30, 60):
            lat_rad = np.radians(lat_deg)
            r_lat = sphere_r * np.cos(lat_rad)
            z_lat = np.full_like(theta, sphere_r * np.sin(lat_rad))
            self.ax.plot(
                r_lat * np.cos(theta),
                r_lat * np.sin(theta),
                z_lat,
                color="#38BDF8",
                lw=0.55,
                alpha=0.11,
            )

        # 2. Whisper-thin Longitude Meridians (sparse every 30° so it never looks congested)
        phi_full = np.linspace(0.0, 2.0 * np.pi, 140)
        for az_deg in (0, 30, 60, 90, 120, 150):
            az_rad = np.radians(az_deg)
            x_mer = sphere_r * np.cos(phi_full) * np.sin(az_rad)
            y_mer = sphere_r * np.cos(phi_full) * np.cos(az_rad)
            z_mer = sphere_r * np.sin(phi_full)
            is_principal = (az_deg % 90 == 0)
            self.ax.plot(
                x_mer, y_mer, z_mer,
                color="#38BDF8",
                lw=0.75 if is_principal else 0.50,
                alpha=0.18 if is_principal else 0.08,
            )

        # 3. Soft Equatorial Compass Platform at z = 0 (low alpha so it never dims the needle)
        r_disc = np.linspace(0.0, sphere_r, 2)
        t_grid, r_grid = np.meshgrid(theta, r_disc)
        x_disc = r_grid * np.cos(t_grid)
        y_disc = r_grid * np.sin(t_grid)
        z_disc = np.zeros_like(x_disc)
        self.ax.plot_surface(
            x_disc, y_disc, z_disc,
            color="#0B2239", alpha=0.18, shade=False, linewidth=0
        )

        # Concentric range rings on the z = 0 equatorial plane
        for r_ring, alpha_val, lw_val in [
            (0.36, 0.20, 0.65),
            (0.72, 0.26, 0.75),
            (sphere_r, 0.68, 1.35),
        ]:
            self.ax.plot(
                r_ring * np.cos(theta),
                r_ring * np.sin(theta),
                np.zeros_like(theta),
                color="#38BDF8", lw=lw_val, alpha=alpha_val
            )

        # Subtle radial crosshairs on z = 0
        self.ax.plot([-sphere_r, sphere_r], [0, 0], [0, 0], color="#1E3A5F", lw=0.8, alpha=0.45)
        self.ax.plot([0, 0], [-sphere_r, sphere_r], [0, 0], color="#1E3A5F", lw=0.8, alpha=0.45)

        # Clean azimuth degree ticks every 15 degrees around the outer compass rim
        for deg in range(0, 360, 15):
            rad = np.radians(deg)
            is_major = (deg % 45 == 0)
            r_in = (sphere_r - 0.065) if is_major else (sphere_r - 0.038)
            r_out = sphere_r
            self.ax.plot(
                [r_in * np.sin(rad), r_out * np.sin(rad)],
                [r_in * np.cos(rad), r_out * np.cos(rad)],
                [0.0, 0.0],
                color="#7DD3FC" if is_major else "#38BDF8",
                lw=1.1 if is_major else 0.65,
                alpha=0.65 if is_major else 0.35
            )

        # 4. Principal Reference Axes (clean, crisp, non-intrusive)
        # Vertical Z axis through sphere (south pole to north pole)
        self.ax.plot(
            [0, 0], [0, 0], [-sphere_r, sphere_r],
            color="#94A3B8", lw=1.1, linestyle=":", alpha=0.55
        )
        self.ax.plot(
            [0, 0], [0, 0], [sphere_r, sphere_r + 0.08],
            color="#60A5FA", lw=1.8, alpha=0.90
        )
        self.ax.plot(
            [0], [0], [sphere_r + 0.09],
            marker="^", markersize=6, color="#60A5FA", alpha=0.95
        )

        # Positive X and Y axis lines
        self.ax.plot([0, sphere_r], [0, 0], [0, 0], color="#EF4444", lw=1.8, alpha=0.88)
        self.ax.plot([0, 0], [0, sphere_r], [0, 0], color="#10B981", lw=1.8, alpha=0.88)

        # Axis labels positioned outside the sphere rim so they never collide with the needle
        self.ax.text(
            sphere_r + 0.13, -0.04, -0.02, "+X (Right)",
            color="#F87171", fontsize=8.5, fontweight="bold", ha="center"
        )
        self.ax.text(
            0.14, sphere_r + 0.12, -0.04, "+Y (Fwd 0°)",
            color="#34D399", fontsize=8.5, fontweight="bold", ha="left"
        )
        self.ax.text(
            0.0, 0.0, sphere_r + 0.15, "+Z (Up 90°)",
            color="#93C5FD", fontsize=8.5, fontweight="bold", ha="center"
        )

    def _clear_dynamic_artists(self):
        for artist in self._dynamic_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._dynamic_artists.clear()

    def _track_plot(self, *args, **kwargs):
        artists = self.ax.plot(*args, **kwargs)
        self._dynamic_artists.extend(artists)
        return artists

    @staticmethod
    def _angle_delta(target, current):
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
        self.target_pitch = float(pitch)
        self.target_yaw = float(yaw)
        if roll is not None:
            self.target_roll = float(roll)
        self._force_render = True
        self._render_needle(self._display_pitch, self._display_yaw, self._display_roll)
        self.draw_idle()

    def update_needle(self, current_pitch, current_yaw, current_roll=0.0):
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
        self._clear_dynamic_artists()
        base = self.origin

        # Seamless outer sphere silhouette ring perpendicular to the current camera view
        elev_rad = np.radians(getattr(self.ax, "elev", 22.0))
        azim_rad = np.radians(getattr(self.ax, "azim", 38.0))
        v_cam = np.array([
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ], dtype=float)
        u1 = np.array([-np.sin(azim_rad), np.cos(azim_rad), 0.0], dtype=float)
        u2 = np.cross(v_cam, u1)
        sil_t = np.linspace(0.0, 2.0 * np.pi, 140)
        sil_pts = self.SPHERE_RADIUS * (
            np.cos(sil_t)[:, None] * u1[None, :] + np.sin(sil_t)[:, None] * u2[None, :]
        )
        self._track_plot(
            sil_pts[:, 0], sil_pts[:, 1], sil_pts[:, 2],
            color="#38BDF8", linewidth=1.0, alpha=0.28
        )

        # Check alignment status for dynamic color feedback
        err_pitch = abs(current_pitch - self.target_pitch)
        err_yaw = abs(angle_delta_deg(current_yaw, self.target_yaw))
        err_roll = abs(angle_delta_deg(current_roll, self.target_roll))
        is_aligned = (err_pitch <= 2.0) and (err_yaw <= 2.0) and (err_roll <= 2.0)

        # Pivot gimbal ring at entry point (0, 0, 0)
        theta = np.linspace(0, 2 * np.pi, 48)
        self._track_plot(
            0.055 * np.cos(theta), 0.055 * np.sin(theta), np.zeros_like(theta),
            color="#38BDF8", linewidth=1.4, alpha=0.85
        )

        # ── 1. TARGET TRAJECTORY & BULLSEYE GATE ────────────────────────────
        target_basis = build_needle_frame(self.target_pitch, self.target_yaw, self.target_roll)
        target_dir = target_basis[:, 2]
        target_len = self.SPHERE_RADIUS
        target_end = base + target_dir * target_len

        if self.show_target:
            # Target ray
            self._track_plot(
                [base[0], target_end[0]], [base[1], target_end[1]], [base[2], target_end[2]],
                color="#22C55E", linestyle="--", linewidth=2.0, alpha=0.90
            )

            # Target bullseye rings perpendicular to target trajectory at tip distance
            t_right, t_up = target_basis[:, 0], target_basis[:, 1]
            for bull_r, bull_alpha, bull_lw in [(0.045, 0.95, 2.0), (0.090, 0.55, 1.2)]:
                ring_pts = (
                    target_end[None, :]
                    + bull_r * (np.cos(theta)[:, None] * t_right[None, :] + np.sin(theta)[:, None] * t_up[None, :])
                )
                self._track_plot(
                    ring_pts[:, 0], ring_pts[:, 1], ring_pts[:, 2],
                    color="#4ADE80", linewidth=bull_lw, alpha=bull_alpha
                )

            # Center target dot
            self._track_plot(
                [target_end[0]], [target_end[1]], [target_end[2]],
                marker="o", markersize=5, color="#4ADE80"
            )

            target_label_pos = target_end + t_up * 0.11
            self._dynamic_artists.append(self.ax.text(
                target_label_pos[0], target_label_pos[1], target_label_pos[2],
                "TARGET", color="#4ADE80", fontsize=8.5, fontweight="bold", ha="center"
            ))

        # ── 2. SURGICAL NEEDLE 3D MESH (HUB + CANNULA + BEVEL TIP) ──────────
        unrolled_basis = build_needle_frame(current_pitch, current_yaw, 0.0)
        rolled_basis = build_needle_frame(current_pitch, current_yaw, current_roll)

        roll_right = rolled_basis[:, 0]
        roll_up = rolled_basis[:, 1]
        current_dir = rolled_basis[:, 2]

        # A. Ergonomic Luer-Lock Biopsy Handle (z from -0.26 to 0.00)
        z_handle = [-0.26, -0.24, -0.17, -0.15, -0.06, -0.04, 0.00]
        r_handle = [0.042, 0.058, 0.054, 0.044, 0.044, 0.056, 0.032]
        Xh, Yh, Zh = _profiled_tube_mesh(base, rolled_basis, z_handle, r_handle, n_theta=24)
        self._dynamic_artists.append(self.ax.plot_surface(
            Xh, Yh, Zh, color="#475569", lightsource=self.light_source,
            shade=True, alpha=0.98, linewidth=0, antialiased=True
        ))

        # Anodized Accent Ring on Handle Collar
        z_collar = [-0.055, -0.035]
        r_collar = [0.057, 0.057]
        Xc, Yc, Zc = _profiled_tube_mesh(base, rolled_basis, z_collar, r_collar, n_theta=24)
        self._dynamic_artists.append(self.ax.plot_surface(
            Xc, Yc, Zc, color="#0EA5E9", lightsource=self.light_source,
            shade=True, alpha=1.0, linewidth=0, antialiased=True
        ))

        # Lateral Finger Grip Wings on Hub (aligned with roll_right)
        wing_center = base - current_dir * 0.12
        wing_span = 0.13
        w_left = wing_center - roll_right * wing_span
        w_right = wing_center + roll_right * wing_span
        self._track_plot(
            [w_left[0], w_right[0]], [w_left[1], w_right[1]], [w_left[2], w_right[2]],
            color="#94A3B8", linewidth=3.5, alpha=0.95
        )

        # Dorsal 3D Orientation Fin (sits cleanly on top of the handle along +roll_up)
        f_rear_b = base - current_dir * 0.22 + roll_up * 0.052
        f_front_b = base - current_dir * 0.05 + roll_up * 0.048
        f_front_t = base - current_dir * 0.07 + roll_up * 0.135
        f_rear_t = base - current_dir * 0.20 + roll_up * 0.150
        fin_verts = [[f_rear_b, f_front_b, f_front_t, f_rear_t]]
        fin_poly = Poly3DCollection(
            fin_verts, facecolor=ROLL_AMBER, edgecolor="#FDE68A", linewidths=1.0, alpha=0.95
        )
        self.ax.add_collection3d(fin_poly)
        self._dynamic_artists.append(fin_poly)

        # B. Stainless-Steel Surgical Cannula Shaft (z from 0.00 to 0.78)
        shaft_len = 0.78
        shaft_r = 0.022
        z_shaft = np.linspace(0.0, shaft_len, 6)
        r_shaft = np.full_like(z_shaft, shaft_r)
        Xs, Ys, Zs = _profiled_tube_mesh(base, rolled_basis, z_shaft, r_shaft, n_theta=24)
        self._dynamic_artists.append(self.ax.plot_surface(
            Xs, Ys, Zs, color="#F1F5F9", lightsource=self.light_source,
            shade=True, alpha=1.0, linewidth=0, antialiased=True
        ))

        # Depth Graduation Rings along Cannula (every 0.18 units)
        ring_t = np.linspace(0, 2 * np.pi, 24)
        for z_mark in (0.18, 0.36, 0.54, 0.72):
            mark_center = base + current_dir * z_mark
            mark_pts = (
                mark_center[None, :]
                + (shaft_r * 1.04) * (
                    np.cos(ring_t)[:, None] * roll_right[None, :]
                    + np.sin(ring_t)[:, None] * roll_up[None, :]
                )
            )
            self._track_plot(
                mark_pts[:, 0], mark_pts[:, 1], mark_pts[:, 2],
                color="#334155", linewidth=1.0, alpha=0.85
            )

        # Flush Dorsal Roll Stripe along Cannula
        stripe_p0 = base + roll_up * (shaft_r * 1.06)
        stripe_p1 = base + current_dir * shaft_len + roll_up * (shaft_r * 1.06)
        self._track_plot(
            [stripe_p0[0], stripe_p1[0]],
            [stripe_p0[1], stripe_p1[1]],
            [stripe_p0[2], stripe_p1[2]],
            color=ROLL_AMBER, linewidth=2.5, alpha=0.98
        )

        # C. Precision Surgical Bevel Tip (smoothly tapers from shaft_r to 0.0)
        tip_len = 0.14
        z_tip = [shaft_len, shaft_len + tip_len * 0.45, shaft_len + tip_len]
        r_tip = [shaft_r, shaft_r * 0.62, 0.0]
        Xt, Yt, Zt = _profiled_tube_mesh(
            base, rolled_basis, z_tip, r_tip, n_theta=24, bevel_cut=-0.55
        )
        tip_color = "#22C55E" if is_aligned else "#38BDF8"
        self._dynamic_artists.append(self.ax.plot_surface(
            Xt, Yt, Zt, color=tip_color, lightsource=self.light_source,
            shade=True, alpha=1.0, linewidth=0, antialiased=True
        ))

        # Laser Trajectory Projection from Needle Tip to Target Sphere Distance
        needle_tip_pos = base + current_dir * (shaft_len + tip_len)
        ray_end_pos = base + current_dir * target_len
        self._track_plot(
            [needle_tip_pos[0], ray_end_pos[0]],
            [needle_tip_pos[1], ray_end_pos[1]],
            [needle_tip_pos[2], ray_end_pos[2]],
            color=tip_color, linewidth=2.2, linestyle="-", alpha=0.95
        )
        self._track_plot(
            [ray_end_pos[0]], [ray_end_pos[1]], [ray_end_pos[2]],
            marker="o", markersize=6, color=tip_color
        )

        # Shadow projection of current needle on the z = 0 compass floor for depth perception
        self._track_plot(
            [0.0, ray_end_pos[0]], [0.0, ray_end_pos[1]], [0.0, 0.0],
            color="#38BDF8", linewidth=1.1, linestyle=":", alpha=0.40
        )

        # ── 3. 360° AXIAL ROLL GONIOMETER DIAL ──────────────────────────────
        dial_center = base + current_dir * 0.08
        dial_radius = 0.19
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
            color="#38BDF8", linewidth=1.2, linestyle=":", alpha=0.75
        )

        # Current Roll Vector on Goniometer Dial (aligned with dorsal +roll_up)
        curr_roll_tip = dial_center + roll_up * dial_radius
        self._track_plot(
            [dial_center[0], curr_roll_tip[0]],
            [dial_center[1], curr_roll_tip[1]],
            [dial_center[2], curr_roll_tip[2]],
            color=ROLL_AMBER, linewidth=2.4, alpha=0.98
        )
        self._track_plot(
            [curr_roll_tip[0]], [curr_roll_tip[1]], [curr_roll_tip[2]],
            marker="o", markersize=5.5, color=ROLL_AMBER
        )

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
                color="#4ADE80", linestyle="--", linewidth=2.0, alpha=0.95
            )
            self._track_plot(
                [target_roll_tip[0]], [target_roll_tip[1]], [target_roll_tip[2]],
                marker="o", markersize=5.0, color="#4ADE80"
            )

            roll_arc = _roll_arc_points(
                dial_center, unrolled_basis,
                self.target_roll, current_roll,
                radius=dial_radius * 0.82, samples=28
            )
            if len(roll_arc):
                self._track_plot(
                    roll_arc[:, 0], roll_arc[:, 1], roll_arc[:, 2],
                    color=ROLL_AMBER, linewidth=2.2, alpha=0.90
                )

            # Great-circle directional error arc between Current and Target trajectories
            dir_arc = _direction_arc_points(
                current_dir, target_dir, center=base, radius=0.52, samples=28
            )
            if len(dir_arc):
                self._track_plot(
                    dir_arc[:, 0], dir_arc[:, 1], dir_arc[:, 2],
                    color="#38BDF8" if not is_aligned else "#4ADE80",
                    linewidth=2.0, linestyle="--", alpha=0.85
                )

        # ── 4. CLEAN 3D TIP LABEL & TOP-LEFT CLINICAL HUD OVERLAY ───────────
        curr_label_pos = ray_end_pos + roll_up * 0.09
        self._dynamic_artists.append(self.ax.text(
            curr_label_pos[0], curr_label_pos[1], curr_label_pos[2],
            "LOCKED" if is_aligned else "CURRENT",
            color=tip_color, fontsize=8.5, fontweight="bold", ha="center"
        ))

        display_roll_360 = (current_roll % 360.0 + 360.0) % 360.0
        total_ang_err = np.degrees(np.arccos(np.clip(np.dot(current_dir, target_dir), -1.0, 1.0)))
        hud_text = (
            f"ROLL: {current_roll:+.1f}° ({display_roll_360:03.0f}°)   │   "
            f"PITCH: {current_pitch:+.1f}°   │   "
            f"YAW: {current_yaw:+.1f}°   │   "
            f"Δ ANGLE: {total_ang_err:.1f}°"
        )
        self._dynamic_artists.append(self.ax.text2D(
            0.025, 0.945, hud_text,
            transform=self.ax.transAxes,
            color="#F8FAFC", fontsize=9.0, fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.50",
                facecolor="#0B1528",
                edgecolor="#38BDF8" if not is_aligned else "#22C55E",
                linewidth=1.2,
                alpha=0.90
            )
        ))


# ============================================
#  MAIN DASHBOARD
# ============================================
class NeedleMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Needle Navigation System")
        self.resize(1500, 880)
        self.setMinimumSize(1200, 740)
        self.setFont(QFont("Segoe UI", 11))
        self.setStyleSheet(
            f"background-color: {BG_MAIN}; color: {TEXT_MAIN}; "
            f"font-family: 'Segoe UI', Arial, sans-serif;"
        )

        # Serial config
        self.serial_port = None
        self.is_serial_active = False

        # Target positions
        self.target_pitch = 30.0
        self.target_yaw = 0.0
        self.target_roll = 0.0

        # Last raw telemetry values (always stored in degrees)
        self._last_pitch = 0.0
        self._last_yaw = 0.0
        self._last_roll = 0.0

        # Session data used by Export CSV
        self.live_data = []

        # Automatic 10-second target capture after connecting
        self.target_capture_active = False
        self.target_capture_samples = []
        self.target_capture_start = None
        self.target_capture_dialog = None
        self.target_capture_timer = QTimer()
        self.target_capture_timer.setInterval(100)
        self.target_capture_timer.timeout.connect(self._target_capture_tick)

        # Arduino/device status command transmitter
        self.device_status = "R"
        self.device_status_timer = QTimer()
        self.device_status_timer.setInterval(100)
        self.device_status_timer.timeout.connect(self._send_device_status)

        self.setup_ui()

        # Serial polling (~50 Hz)
        self.serial_timer = QTimer()
        self.serial_timer.setInterval(20)
        self.serial_timer.timeout.connect(self.read_serial_data)

    # ------------------------------------------------------------
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(18, 14, 18, 16)
        main_layout.setSpacing(14)

        # ── TOP HEADER BAR ──────────────────────────────────────────────────
        header_frame = QFrame()
        header_frame.setFixedHeight(68)
        header_frame.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 12px; border: 1px solid {BORDER_COLOR};"
        )
        add_card_shadow(header_frame, blur=18, y_offset=2, alpha=14)

        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(18, 8, 18, 8)

        # Left logo area
        logo_box = QFrame()
        logo_box.setFixedWidth(210)
        logo_box.setStyleSheet("background: transparent; border: none;")
        logo_layout = QHBoxLayout(logo_box)
        logo_layout.setContentsMargins(0, 0, 0, 0)

        self.logo_label = QLabel()
        self.logo_label.setFixedSize(150, 46)
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

        # Center title block
        center_box = QFrame()
        center_box.setStyleSheet("background: transparent; border: none;")
        center_layout = QVBoxLayout(center_box)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(1)
        center_layout.setAlignment(Qt.AlignCenter)

        lbl_title = QLabel("Smart Needle Navigation System")
        lbl_title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        lbl_title.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")
        lbl_title.setAlignment(Qt.AlignCenter)

        lbl_sub = QLabel("Precision 3D Stereotactic Orientation & Real-Time Procedure Guidance")
        lbl_sub.setFont(QFont("Segoe UI", 10))
        lbl_sub.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        lbl_sub.setAlignment(Qt.AlignCenter)

        center_layout.addWidget(lbl_title)
        center_layout.addWidget(lbl_sub)
        header_layout.addWidget(center_box, stretch=1)

        # Right connection badge
        status_box = QFrame()
        status_box.setFixedWidth(240)
        status_box.setStyleSheet("background: transparent; border: none;")
        status_layout = QHBoxLayout(status_box)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.lbl_status = QLabel("●  Device Disconnected")
        self.lbl_status.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setFixedHeight(34)
        self.lbl_status.setStyleSheet(
            f"color: {ERROR_RED}; background: {ERROR_BG}; "
            f"border: 1px solid {ERROR_BORDER}; border-radius: 17px; padding: 4px 14px;"
        )
        status_layout.addWidget(self.lbl_status)
        header_layout.addWidget(status_box)

        main_layout.addWidget(header_frame)

        # ── MAIN CONTENT AREA (LEFT 3D VIEW / RIGHT TELEMETRY & GUIDANCE) ──
        dash_layout = QHBoxLayout()
        dash_layout.setContentsMargins(0, 0, 0, 0)
        dash_layout.setSpacing(16)

        # LEFT COLUMN: 3D Viewport + Bottom Toolbar
        left_column = QVBoxLayout()
        left_column.setSpacing(12)

        view_frame = QFrame()
        view_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        view_frame.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 14px; border: 1px solid {BORDER_COLOR};"
        )
        add_card_shadow(view_frame)
        view_layout = QVBoxLayout(view_frame)
        view_layout.setContentsMargins(16, 12, 16, 14)
        view_layout.setSpacing(10)

        view_header = QHBoxLayout()
        view_header.setSpacing(6)
        lbl_3d = QLabel("3D Stereotactic Needle Viewport")
        lbl_3d.setFont(QFont("Segoe UI", 13, QFont.Bold))
        lbl_3d.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")

        # Camera Preset Buttons with active highlight state
        self._cam_buttons = {}
        btn_cam_iso = QPushButton("3D Iso")
        btn_cam_pitch = QPushButton("Side (Pitch)")
        btn_cam_yaw = QPushButton("Top (Yaw)")
        btn_cam_roll = QPushButton("Axial (Roll)")
        for b, preset in [
            (btn_cam_iso, "iso"),
            (btn_cam_pitch, "pitch"),
            (btn_cam_yaw, "yaw"),
            (btn_cam_roll, "roll"),
        ]:
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(28)
            self._cam_buttons[preset] = b
            b.clicked.connect(lambda checked, p=preset: self._on_camera_preset_clicked(p))
        self._style_camera_buttons("iso")

        self.chk_show_target = QCheckBox("Show Target")
        self.chk_show_target.setChecked(True)
        self.chk_show_target.setCursor(Qt.PointingHandCursor)
        self.chk_show_target.setStyleSheet(
            f"color: {TEXT_MAIN}; font-size: 11px; font-weight: 600; border: none;"
        )
        self.chk_show_target.stateChanged.connect(
            lambda state: self.canvas3d.set_show_target(state == Qt.Checked)
        )

        legend_pill_style = (
            f"background: {BG_SOFT}; border: 1px solid {BORDER_COLOR}; "
            f"border-radius: 6px; padding: 3px 8px; font-size: 11px; font-weight: bold;"
        )
        lbl_legend_target = QLabel("◎  Target Gate")
        lbl_legend_target.setStyleSheet(f"color: {SUCCESS_GREEN}; {legend_pill_style}")

        lbl_legend_curr = QLabel("━  Cannula")
        lbl_legend_curr.setStyleSheet(f"color: #0284C7; {legend_pill_style}")

        lbl_legend_roll = QLabel("▲  Dorsal Roll Fin")
        lbl_legend_roll.setStyleSheet(f"color: #D97706; {legend_pill_style}")

        view_header.addWidget(lbl_3d)
        view_header.addSpacing(8)
        view_header.addWidget(btn_cam_iso)
        view_header.addWidget(btn_cam_pitch)
        view_header.addWidget(btn_cam_yaw)
        view_header.addWidget(btn_cam_roll)
        view_header.addStretch()
        view_header.addWidget(self.chk_show_target)
        view_header.addSpacing(6)
        view_header.addWidget(lbl_legend_target)
        view_header.addWidget(lbl_legend_curr)
        view_header.addWidget(lbl_legend_roll)
        view_layout.addLayout(view_header)

        # Dark Framed Container for the 3D Canvas
        canvas_container = QFrame()
        canvas_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_container.setStyleSheet(
            f"background-color: {VIEWPORT_BG}; border-radius: 10px; border: 1px solid #1E293B;"
        )
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(2, 2, 2, 2)

        self.canvas3d = Needle3DCanvas()
        self.canvas3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_layout.addWidget(self.canvas3d)

        view_layout.addWidget(canvas_container, stretch=1)
        left_column.addWidget(view_frame, stretch=1)

        # Bottom Control Bar
        controls_frame = QFrame()
        controls_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        controls_frame.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 12px; border: 1px solid {BORDER_COLOR};"
        )
        add_card_shadow(controls_frame, blur=18, y_offset=2, alpha=14)

        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(14, 10, 14, 10)
        controls_layout.setSpacing(12)

        # Serial Connection Group
        serial_group = QFrame()
        serial_group.setStyleSheet(
            f"background: {BG_SOFT}; border: 1px solid {BORDER_COLOR}; border-radius: 9px;"
        )
        serial_layout = QHBoxLayout(serial_group)
        serial_layout.setContentsMargins(12, 6, 12, 6)
        serial_layout.setSpacing(8)

        serial_title = QLabel("DEVICE CONNECTION")
        serial_title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        serial_title.setStyleSheet(f"color: {TEXT_SUB}; letter-spacing: 0.5px; border: none;")

        port_label = QLabel("Port:")
        port_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        port_label.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")
        self.port_input = QLineEdit("COM3")
        self.port_input.setFixedWidth(74)
        self.port_input.setFixedHeight(32)
        self.port_input.setAlignment(Qt.AlignCenter)
        self.port_input.setStyleSheet(
            f"background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; "
            f"color: {TEXT_MAIN}; font-weight: 600; border-radius: 6px; padding: 2px 6px;"
        )

        baud_label = QLabel("Baud:")
        baud_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        baud_label.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")
        self.baud_input = QLineEdit("115200")
        self.baud_input.setFixedWidth(78)
        self.baud_input.setFixedHeight(32)
        self.baud_input.setAlignment(Qt.AlignCenter)
        self.baud_input.setStyleSheet(
            f"background: {BG_CARD}; border: 1px solid {BORDER_COLOR}; "
            f"color: {TEXT_MAIN}; font-weight: 600; border-radius: 6px; padding: 2px 6px;"
        )

        self.btn_connect = QPushButton("Connect Device")
        self.btn_connect.setCursor(Qt.PointingHandCursor)
        self.btn_connect.setFixedSize(130, 32)
        self.btn_connect.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_BLUE}; color: white; font-weight: bold; "
            f"font-size: 11px; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: #1D4ED8; }}"
        )
        self.btn_connect.clicked.connect(self.toggle_serial)

        serial_layout.addWidget(serial_title)
        serial_layout.addSpacing(4)
        serial_layout.addWidget(port_label)
        serial_layout.addWidget(self.port_input)
        serial_layout.addWidget(baud_label)
        serial_layout.addWidget(self.baud_input)
        serial_layout.addSpacing(4)
        serial_layout.addWidget(self.btn_connect)

        # Session Data / CSV Export Group
        export_group = QFrame()
        export_group.setStyleSheet(
            f"background: {BG_SOFT}; border: 1px solid {BORDER_COLOR}; border-radius: 9px;"
        )
        export_layout = QHBoxLayout(export_group)
        export_layout.setContentsMargins(12, 6, 12, 6)
        export_layout.setSpacing(10)

        export_title = QLabel("SESSION LOG")
        export_title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        export_title.setStyleSheet(f"color: {TEXT_SUB}; letter-spacing: 0.5px; border: none;")

        self.lbl_session_file = QLabel("Auto-recording telemetry")
        self.lbl_session_file.setFont(QFont("Segoe UI", 10))
        self.lbl_session_file.setStyleSheet(f"color: {TEXT_SUB}; border: none;")

        self.btn_export = QPushButton("↓  Export CSV")
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.setFixedSize(116, 32)
        self.btn_export.setStyleSheet(
            f"QPushButton {{ background-color: {SUCCESS_GREEN}; color: white; font-weight: bold; "
            f"font-size: 11px; border-radius: 6px; border: none; }}"
            f"QPushButton:hover {{ background-color: #15803D; }}"
        )
        self.btn_export.clicked.connect(self.export_csv)

        export_layout.addWidget(export_title)
        export_layout.addWidget(self.lbl_session_file, stretch=1)
        export_layout.addWidget(self.btn_export)

        controls_layout.addWidget(serial_group, stretch=3)
        controls_layout.addWidget(export_group, stretch=2)
        left_column.addWidget(controls_frame, stretch=0)

        # ── RIGHT COLUMN: ORIENTATION READOUTS + GUIDANCE INDICATOR ─────────
        right_panel_widget = QWidget()
        right_panel_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel_widget.setMinimumWidth(465)
        right_panels = QVBoxLayout(right_panel_widget)
        right_panels.setContentsMargins(0, 0, 0, 0)
        right_panels.setSpacing(14)

        # 1. Needle Orientation Card
        orient_frame = QFrame()
        orient_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        orient_frame.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 14px; border: 1px solid {BORDER_COLOR};"
        )
        add_card_shadow(orient_frame)
        orient_layout = QVBoxLayout(orient_frame)
        orient_layout.setContentsMargins(20, 16, 20, 16)
        orient_layout.setSpacing(12)

        orient_top_row = QHBoxLayout()
        orient_title_box = QVBoxLayout()
        orient_title_box.setSpacing(2)
        orient_title = QLabel("Needle Orientation")
        orient_title.setFont(QFont("Segoe UI", 15, QFont.Bold))
        orient_title.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")
        orient_sub = QLabel("Real-time angular telemetry vs. target")
        orient_sub.setFont(QFont("Segoe UI", 10))
        orient_sub.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        orient_title_box.addWidget(orient_title)
        orient_title_box.addWidget(orient_sub)

        self.lbl_telemetry_badge = QLabel("●  STANDBY")
        self.lbl_telemetry_badge.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.lbl_telemetry_badge.setAlignment(Qt.AlignCenter)
        self.lbl_telemetry_badge.setFixedHeight(28)
        self.lbl_telemetry_badge.setStyleSheet(
            f"background: {BG_SOFT}; color: {TEXT_SUB}; border: 1px solid {BORDER_COLOR}; "
            f"padding: 3px 10px; border-radius: 14px;"
        )

        degree_badge = QLabel("DEGREES (°)")
        degree_badge.setFont(QFont("Segoe UI", 9, QFont.Bold))
        degree_badge.setAlignment(Qt.AlignCenter)
        degree_badge.setFixedHeight(28)
        degree_badge.setStyleSheet(
            f"background: {INFO_BG}; color: {ACCENT_BLUE}; border: 1px solid {INFO_BORDER}; "
            f"padding: 3px 10px; border-radius: 14px;"
        )

        orient_top_row.addLayout(orient_title_box)
        orient_top_row.addStretch()
        orient_top_row.addWidget(self.lbl_telemetry_badge)
        orient_top_row.addSpacing(6)
        orient_top_row.addWidget(degree_badge)
        orient_layout.addLayout(orient_top_row)

        orient_divider = QFrame()
        orient_divider.setFixedHeight(1)
        orient_divider.setStyleSheet(f"background-color: {BORDER_COLOR}; border: none;")
        orient_layout.addWidget(orient_divider)

        # Telemetry Table Grid
        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        header_style = (
            f"color: {TEXT_SUB}; font-size: 10px; font-weight: bold; "
            f"letter-spacing: 0.8px; border: none;"
        )
        axis_header = QLabel("AXIS")
        current_header = QLabel("CURRENT")
        target_header = QLabel("TARGET")
        error_header = QLabel("ERROR")
        for h in (axis_header, current_header, target_header, error_header):
            h.setStyleSheet(header_style)

        grid.addWidget(axis_header, 0, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(current_header, 0, 1, alignment=Qt.AlignCenter)
        grid.addWidget(target_header, 0, 2, alignment=Qt.AlignCenter)
        grid.addWidget(error_header, 0, 3, alignment=Qt.AlignCenter)

        self.lbl_curr_roll, self.lbl_tar_roll, self.lbl_err_roll = self._create_row(
            grid, 1, "Roll", "Axial 360°", self.target_roll
        )
        self.lbl_curr_pitch, self.lbl_tar_pitch, self.lbl_err_pitch = self._create_row(
            grid, 2, "Pitch", "Vertical", self.target_pitch
        )
        self.lbl_curr_yaw, self.lbl_tar_yaw, self.lbl_err_yaw = self._create_row(
            grid, 3, "Yaw", "Horizontal", self.target_yaw
        )
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 3)
        grid.setColumnStretch(3, 3)
        orient_layout.addLayout(grid, stretch=1)

        orient_hint = QLabel(
            "Alignment Tolerance:  ±2.0° across Roll, Pitch, and Yaw   •   Continuous 360° tracking"
        )
        orient_hint.setAlignment(Qt.AlignCenter)
        orient_hint.setFixedHeight(30)
        orient_hint.setStyleSheet(
            f"color: {TEXT_SUB}; background: {BG_SOFT}; border: 1px solid {BORDER_COLOR}; "
            f"border-radius: 7px; font-size: 11px; font-weight: 500; padding: 4px 10px;"
        )
        orient_layout.addWidget(orient_hint)
        right_panels.addWidget(orient_frame, stretch=11)

        # 2. Guidance Indicator Card
        guide_frame = QFrame()
        guide_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        guide_frame.setStyleSheet(
            f"background-color: {BG_CARD}; border-radius: 14px; border: 1px solid {BORDER_COLOR};"
        )
        add_card_shadow(guide_frame)
        guide_layout = QVBoxLayout(guide_frame)
        guide_layout.setContentsMargins(20, 16, 20, 16)
        guide_layout.setSpacing(12)

        top_guide = QHBoxLayout()
        guidance_title_box = QVBoxLayout()
        guidance_title_box.setSpacing(2)
        lbl_guide = QLabel("Guidance Indicator")
        lbl_guide.setFont(QFont("Segoe UI", 15, QFont.Bold))
        lbl_guide.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")
        lbl_guide_sub = QLabel("Real-time directional & roll correction prompts")
        lbl_guide_sub.setFont(QFont("Segoe UI", 10))
        lbl_guide_sub.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
        guidance_title_box.addWidget(lbl_guide)
        guidance_title_box.addWidget(lbl_guide_sub)

        self.lbl_range = QLabel("WAITING")
        self.lbl_range.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.lbl_range.setAlignment(Qt.AlignCenter)
        self.lbl_range.setFixedSize(134, 30)
        self.lbl_range.setStyleSheet(
            f"background: {BG_SOFT}; color: {TEXT_SUB}; border: 1px solid {BORDER_COLOR}; "
            f"border-radius: 15px; padding: 4px 10px;"
        )
        top_guide.addLayout(guidance_title_box)
        top_guide.addStretch()
        top_guide.addWidget(self.lbl_range)
        guide_layout.addLayout(top_guide)

        guide_divider = QFrame()
        guide_divider.setFixedHeight(1)
        guide_divider.setStyleSheet(f"background-color: {BORDER_COLOR}; border: none;")
        guide_layout.addWidget(guide_divider)

        indicator_layout = QHBoxLayout()
        indicator_layout.setContentsMargins(4, 2, 4, 2)
        indicator_layout.setSpacing(16)

        # Left Bullseye Status Column
        status_col = QVBoxLayout()
        status_col.setSpacing(6)
        status_col.setAlignment(Qt.AlignCenter)

        self.circle_lbl = QLabel("•")
        self.circle_lbl.setAlignment(Qt.AlignCenter)
        self.circle_lbl.setFixedSize(88, 88)
        self.circle_lbl.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self.circle_lbl.setStyleSheet(
            f"color: {TEXT_SUB}; background: {BG_SOFT}; border: 6px solid {BORDER_COLOR}; "
            f"border-radius: 44px;"
        )

        self.lbl_ontarget = QLabel("NO DATA")
        self.lbl_ontarget.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.lbl_ontarget.setAlignment(Qt.AlignCenter)
        self.lbl_ontarget.setStyleSheet(f"color: {TEXT_SUB}; border: none;")

        self.lbl_steady = QLabel("Connect device to start")
        self.lbl_steady.setAlignment(Qt.AlignCenter)
        self.lbl_steady.setWordWrap(True)
        self.lbl_steady.setFixedWidth(145)
        self.lbl_steady.setStyleSheet(f"color: {TEXT_SUB}; font-size: 11px; border: none;")

        status_col.addWidget(self.circle_lbl, alignment=Qt.AlignCenter)
        status_col.addWidget(self.lbl_ontarget, alignment=Qt.AlignCenter)
        status_col.addWidget(self.lbl_steady, alignment=Qt.AlignCenter)

        vert_sep = QFrame()
        vert_sep.setFixedWidth(1)
        vert_sep.setStyleSheet(f"background-color: {BORDER_COLOR}; border: none;")

        # Right Directional & Roll Pad
        arrows_box = QGridLayout()
        arrows_box.setContentsMargins(0, 0, 0, 0)
        arrows_box.setHorizontalSpacing(8)
        arrows_box.setVerticalSpacing(8)

        self.btn_roll_ccw = QPushButton("↺  ROLL -")
        self.btn_up = QPushButton("↑  UP")
        self.btn_roll_cw = QPushButton("↻  ROLL +")
        self.btn_left = QPushButton("←  LEFT")
        self.btn_right = QPushButton("RIGHT  →")
        self.btn_down = QPushButton("↓  DOWN")

        for btn in (
            self.btn_roll_ccw, self.btn_up, self.btn_roll_cw,
            self.btn_left, self.btn_right, self.btn_down
        ):
            btn.setFixedSize(90, 42)
            btn.setEnabled(False)

        cross = QLabel("•")
        cross.setAlignment(Qt.AlignCenter)
        cross.setFixedSize(32, 32)
        cross.setFont(QFont("Segoe UI", 16, QFont.Bold))
        cross.setStyleSheet(
            f"color: {TEXT_SUB}; background: {BG_SOFT}; "
            f"border: 1px solid {BORDER_COLOR}; border-radius: 16px;"
        )

        arrows_box.addWidget(self.btn_roll_ccw, 0, 0, alignment=Qt.AlignCenter)
        arrows_box.addWidget(self.btn_up, 0, 1, alignment=Qt.AlignCenter)
        arrows_box.addWidget(self.btn_roll_cw, 0, 2, alignment=Qt.AlignCenter)
        arrows_box.addWidget(self.btn_left, 1, 0, alignment=Qt.AlignCenter)
        arrows_box.addWidget(cross, 1, 1, alignment=Qt.AlignCenter)
        arrows_box.addWidget(self.btn_right, 1, 2, alignment=Qt.AlignCenter)
        arrows_box.addWidget(self.btn_down, 2, 1, alignment=Qt.AlignCenter)

        indicator_layout.addLayout(status_col, stretch=2)
        indicator_layout.addWidget(vert_sep)
        indicator_layout.addLayout(arrows_box, stretch=3)
        guide_layout.addLayout(indicator_layout, stretch=1)

        self.lbl_guidance_hint = QLabel(
            "Highlighted prompts indicate the required correction direction"
        )
        self.lbl_guidance_hint.setAlignment(Qt.AlignCenter)
        self.lbl_guidance_hint.setFixedHeight(28)
        self.lbl_guidance_hint.setStyleSheet(
            f"color: {TEXT_SUB}; background: {BG_SOFT}; border: 1px solid {BORDER_COLOR}; "
            f"border-radius: 7px; font-size: 11px; padding: 4px 10px;"
        )
        guide_layout.addWidget(self.lbl_guidance_hint)
        right_panels.addWidget(guide_frame, stretch=10)

        dash_layout.addLayout(left_column, stretch=13)
        dash_layout.addWidget(right_panel_widget, stretch=9)
        main_layout.addLayout(dash_layout, stretch=1)

        self.update_telemetry(0.0, 0.0, 0.0)
        self._update_guidance_buttons(0.0, 0.0, 0.0)

    def _on_camera_preset_clicked(self, preset_name):
        self._style_camera_buttons(preset_name)
        self.canvas3d.set_camera_preset(preset_name)

    def _style_camera_buttons(self, active_preset):
        for preset, btn in self._cam_buttons.items():
            if preset == active_preset:
                btn.setStyleSheet(
                    f"QPushButton {{ background: {INFO_BG}; color: {ACCENT_BLUE}; "
                    f"border: 1px solid {INFO_BORDER}; border-radius: 6px; "
                    f"padding: 4px 9px; font-size: 11px; font-weight: 700; }}"
                )
            else:
                btn.setStyleSheet(
                    f"QPushButton {{ background: {BG_SOFT}; color: {TEXT_MAIN}; "
                    f"border: 1px solid {BORDER_COLOR}; border-radius: 6px; "
                    f"padding: 4px 9px; font-size: 11px; font-weight: 600; }}"
                    f"QPushButton:hover {{ background: {INFO_BG}; color: {ACCENT_BLUE}; border-color: {INFO_BORDER}; }}"
                )

    def _create_row(self, grid, row, label, sublabel, target_deg):
        axis_box = QWidget()
        axis_box.setStyleSheet("background: transparent; border: none;")
        axis_layout = QVBoxLayout(axis_box)
        axis_layout.setContentsMargins(0, 0, 4, 0)
        axis_layout.setSpacing(0)

        lbl_main = QLabel(label)
        lbl_main.setFont(QFont("Segoe UI", 13, QFont.Bold))
        lbl_main.setStyleSheet(f"color: {TEXT_MAIN}; border: none;")

        lbl_sub = QLabel(sublabel)
        lbl_sub.setFont(QFont("Segoe UI", 9))
        lbl_sub.setStyleSheet(f"color: {TEXT_SUB}; border: none;")

        axis_layout.addWidget(lbl_main)
        axis_layout.addWidget(lbl_sub)

        pill_height = 44

        lbl_curr = QLabel("0.0°")
        lbl_curr.setFont(QFont("Segoe UI", 14, QFont.Bold))
        lbl_curr.setAlignment(Qt.AlignCenter)
        lbl_curr.setFixedHeight(pill_height)
        lbl_curr.setMinimumWidth(96)
        lbl_curr.setStyleSheet(
            f"color: {TEXT_MAIN}; background: {BG_SOFT}; "
            f"border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 4px 8px;"
        )

        lbl_tar = QLabel(self._fmt_angle(target_deg))
        lbl_tar.setFont(QFont("Segoe UI", 14, QFont.Bold))
        lbl_tar.setAlignment(Qt.AlignCenter)
        lbl_tar.setFixedHeight(pill_height)
        lbl_tar.setMinimumWidth(96)
        lbl_tar.setStyleSheet(
            f"color: {ACCENT_BLUE}; background: {INFO_BG}; "
            f"border: 1px solid {INFO_BORDER}; border-radius: 8px; padding: 4px 8px;"
        )

        lbl_err = QLabel("0.0°")
        lbl_err.setFont(QFont("Segoe UI", 14, QFont.Bold))
        lbl_err.setAlignment(Qt.AlignCenter)
        lbl_err.setFixedHeight(pill_height)
        lbl_err.setMinimumWidth(96)
        lbl_err.setStyleSheet(
            f"color: {TEXT_SUB}; background: {BG_SOFT}; "
            f"border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 4px 8px;"
        )

        grid.addWidget(axis_box, row, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        grid.addWidget(lbl_curr, row, 1)
        grid.addWidget(lbl_tar, row, 2)
        grid.addWidget(lbl_err, row, 3)

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
                f"QPushButton {{ background-color: {ACCENT_BLUE}; color: white; font-weight: bold; "
                f"font-size: 11px; border-radius: 6px; border: none; }}"
                f"QPushButton:hover {{ background-color: #1D4ED8; }}"
            )
            self.lbl_status.setText("●  Device Disconnected")
            self.lbl_status.setStyleSheet(
                f"color: {ERROR_RED}; background: {ERROR_BG}; "
                f"border: 1px solid {ERROR_BORDER}; border-radius: 17px; padding: 4px 14px;"
            )
            self.lbl_telemetry_badge.setText("●  DISCONNECTED")
            self.lbl_telemetry_badge.setStyleSheet(
                f"background: {BG_SOFT}; color: {TEXT_SUB}; border: 1px solid {BORDER_COLOR}; "
                f"padding: 3px 10px; border-radius: 14px;"
            )
            self.lbl_range.setText("WAITING")
            self.lbl_range.setStyleSheet(
                f"background: {BG_SOFT}; color: {TEXT_SUB}; border: 1px solid {BORDER_COLOR}; "
                f"border-radius: 15px; padding: 4px 10px;"
            )
            self.circle_lbl.setText("•")
            self.circle_lbl.setStyleSheet(
                f"color: {TEXT_SUB}; background: {BG_SOFT}; "
                f"border: 6px solid {BORDER_COLOR}; border-radius: 44px;"
            )
            self.lbl_ontarget.setText("NO DATA")
            self.lbl_ontarget.setStyleSheet(f"color: {TEXT_SUB}; border: none;")
            self.lbl_steady.setText("Connect device to start")
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
                f"QPushButton {{ background-color: {ERROR_RED}; color: white; font-weight: bold; "
                f"font-size: 11px; border-radius: 6px; border: none; }}"
                f"QPushButton:hover {{ background-color: #B91C1C; }}"
            )
            self.lbl_status.setText("●  Device Connected")
            self.lbl_status.setStyleSheet(
                f"color: {SUCCESS_GREEN}; background: {SUCCESS_BG}; "
                f"border: 1px solid {SUCCESS_BORDER}; border-radius: 17px; padding: 4px 14px;"
            )
            self.lbl_telemetry_badge.setText("●  TARGET CAPTURE")
            self.lbl_telemetry_badge.setStyleSheet(
                f"background: {INFO_BG}; color: {ACCENT_BLUE}; "
                f"border: 1px solid {INFO_BORDER}; padding: 3px 10px; border-radius: 14px;"
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
        title.setFont(QFont("Segoe UI", 15, QFont.Bold))
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
                f"background: {ERROR_BG}; color: {ERROR_RED}; border: 1px solid {ERROR_BORDER}; "
                f"padding: 3px 10px; border-radius: 14px;"
            )
            QMessageBox.warning(
                self,
                "Target Capture Failed",
                "No valid pitch, yaw, roll samples were received during the 10-second target capture."
            )
            return

        self.target_pitch = float(np.mean(samples[:, 0]))
        self.target_yaw = circular_mean_deg(samples[:, 1])
        self.target_roll = circular_mean_deg(samples[:, 2])

        self.canvas3d.set_target_direction(
            self.target_pitch, self.target_yaw, self.target_roll
        )

        self.lbl_tar_pitch.setText(f"{self.target_pitch:.1f}°")
        self.lbl_tar_yaw.setText(f"{self.target_yaw:.1f}°")
        self.lbl_tar_roll.setText(f"{self.target_roll:.1f}°")

        self.update_telemetry(self._last_pitch, self._last_yaw, self._last_roll)

        self.lbl_status.setText(f"●  Target Set ({len(samples)} samples)")
        self.lbl_status.setStyleSheet(
            f"color: {SUCCESS_GREEN}; background: {SUCCESS_BG}; "
            f"border: 1px solid {SUCCESS_BORDER}; border-radius: 17px; padding: 4px 14px;"
        )
        self.lbl_telemetry_badge.setText("●  LIVE TELEMETRY")
        self.lbl_telemetry_badge.setStyleSheet(
            f"background: {SUCCESS_BG}; color: {SUCCESS_GREEN}; "
            f"border: 1px solid {SUCCESS_BORDER}; padding: 3px 10px; border-radius: 14px;"
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
                if not text or text.count(",") != 2:
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
        """Highlight a guidance direction without changing button dimensions."""
        if active:
            button.setStyleSheet(f"""
                QPushButton {{
                    color: white;
                    background: {ACCENT_BLUE};
                    border: 1px solid {ACCENT_BLUE};
                    border-radius: 9px;
                    font-size: 11px;
                    font-weight: bold;
                }}
            """)
        else:
            button.setStyleSheet(f"""
                QPushButton {{
                    color: {TEXT_SUB};
                    background: {BG_SOFT};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 9px;
                    font-size: 11px;
                    font-weight: bold;
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

    def update_telemetry(self, pitch, yaw, roll):
        self._last_pitch, self._last_yaw, self._last_roll = pitch, yaw, roll

        # 1. Update 3D Visualizer with all 3 axes (pitch, yaw, roll)
        self.canvas3d.update_needle(pitch, yaw, roll)

        # 2. Calculate Exact Shortest-Path Errors (degrees, safe across 360° wrap)
        err_pitch = pitch - self.target_pitch
        err_yaw = angle_delta_deg(yaw, self.target_yaw)
        err_roll = angle_delta_deg(roll, self.target_roll)

        # 3. Update text values in degrees (wrapped cleanly to [-180°, +180°])
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

        self._update_guidance_buttons(err_pitch, err_yaw, err_roll)

        if is_pitch_ok and is_yaw_ok and is_roll_ok:
            self.lbl_range.setText("✔  WITHIN RANGE")
            self.lbl_range.setStyleSheet(
                f"background: {SUCCESS_BG}; color: {SUCCESS_GREEN}; "
                f"font-weight: bold; padding: 4px 10px; border-radius: 15px; "
                f"border: 1px solid {SUCCESS_BORDER};"
            )
            self.circle_lbl.setText("✓")
            self.circle_lbl.setStyleSheet(
                f"color: {SUCCESS_GREEN}; background: {SUCCESS_BG}; "
                f"border: 6px solid {SUCCESS_BORDER}; border-radius: 44px;"
            )
            self.lbl_ontarget.setText("ON TARGET")
            self.lbl_ontarget.setStyleSheet(f"color: {SUCCESS_GREEN}; border: none;")
            self.lbl_steady.setText("Keep the needle steady")
        else:
            self.lbl_range.setText("!  ADJUSTING")
            self.lbl_range.setStyleSheet(
                f"background: {ERROR_BG}; color: {ERROR_RED}; "
                f"font-weight: bold; padding: 4px 10px; border-radius: 15px; "
                f"border: 1px solid {ERROR_BORDER};"
            )
            self.circle_lbl.setText("!")
            self.circle_lbl.setStyleSheet(
                f"color: {ERROR_RED}; background: {ERROR_BG}; "
                f"border: 6px solid {ERROR_BORDER}; border-radius: 44px;"
            )
            self.lbl_ontarget.setText("OFF TARGET")
            self.lbl_ontarget.setStyleSheet(f"color: {ERROR_RED}; border: none;")
            self.lbl_steady.setText("Align direction & roll with target")

    def _fmt_angle(self, deg_value):
        return f"{deg_value:.1f}°"

    def _format_error(self, label, error_deg):
        """Format the error in degrees and color both text and pill background."""
        sign = "+" if error_deg > 0 else ""
        label.setText(f"{sign}{error_deg:.1f}°")

        if abs(error_deg) <= 2.0:
            fg, bg, border = SUCCESS_GREEN, SUCCESS_BG, SUCCESS_BORDER
        else:
            fg, bg, border = ERROR_RED, ERROR_BG, ERROR_BORDER

        label.setStyleSheet(
            f"color: {fg}; background: {bg}; border: 1px solid {border}; "
            f"border-radius: 8px; padding: 4px 8px; font-size: 14pt; font-weight: bold;"
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NeedleMainWindow()
    window.show()
    sys.exit(app.exec_())