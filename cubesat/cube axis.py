import sys
import math
import pandas as pd
import numpy as np
import serial
from datetime import datetime
import csv
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QMessageBox, QLabel, QFrame, QGridLayout, 
                             QSizePolicy, QGraphicsDropShadowEffect, QComboBox, QLineEdit)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QColor 

# --- REQUIRED IMPORTS FOR STL CUBE ---
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.dates as mdates 
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from stl import mesh
except ImportError:
    print("Error: Missing libraries. Run: pip install numpy-stl matplotlib pyserial PyQt6")
    sys.exit(1)

# CONFIGURATION 
WINDOW_SIZE = 30  
ANIMATION_SPEED_MS = 33 
LOGO_FILENAME = "logo.png"
DEFAULT_SERIAL_PORT = 'COM18'  # Change this to your port
DEFAULT_BAUD_RATE = 9600

#  THEME COLORS (Warm Cream Dashboard)
COLOR_BG_MAIN = "#F3EEE5"      # Main application background
COLOR_BG_CARD = "#FBF8F1"      # Header, metric cards, and input fields
COLOR_BG_CUBE = "#E0D5C5"      # Lighter warm cream for the cube viewing area
COLOR_BG_GRAPH = "#EAE0D0"     # Lighter warm cream for sensor plot areas
COLOR_TEXT_MAIN = "#26231F"    # Dark text for the cream interface
COLOR_TEXT_SUB = "#716B62"     # Muted text for secondary labels
COLOR_BORDER = "#D9D1C5"       # Soft warm border
COLOR_CUBE_BORDER = "#333333"  # Preserve the cube panel border

# 1. STL 3D CUBE CLASS (MAXIMIZED SIZE)
class Cube3DCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), facecolor=COLOR_BG_CUBE)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setStyleSheet("background-color: transparent;")
        
        limit = 1.0
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([-limit, limit])
        
        self.ax.set_axis_off()
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_facecolor(COLOR_BG_CUBE) 

        self.light_dir = np.array([0.5, 0.5, 1.0])
        self.light_dir /= np.linalg.norm(self.light_dir)

        try:
            self.your_mesh = mesh.Mesh.from_file('Sat_Base.stl')
            self.is_stl_loaded = True
            print("Sat_Base.stl Loaded Successfully")
        except Exception as e:
            print(f"STL Error: {e}. Loading Red Fallback Square.")
            self.is_stl_loaded = False
            data = np.zeros(2, dtype=mesh.Mesh.dtype) 
            self.your_mesh = mesh.Mesh(data)
            self.your_mesh.vectors = np.array([[[ -1, -1, 0], [ 1, -1, 0], [ 1,  1, 0]], [[ -1, -1, 0], [ 1,  1, 0], [-1,  1, 0]]])

        if len(self.your_mesh.vectors) > 0:
            min_vals = self.your_mesh.vectors.min(axis=(0, 1))
            max_vals = self.your_mesh.vectors.max(axis=(0, 1))
            center = (min_vals + max_vals) / 2
            self.your_mesh.vectors -= center
            max_dim = (max_vals - min_vals).max()
            
            if max_dim > 0: 
                scale_factor = 1.8 / max_dim 
                self.your_mesh.vectors *= scale_factor
        
        self.base_vectors = self.your_mesh.vectors.copy()
        self.base_normals = self.your_mesh.normals.copy() if self.is_stl_loaded else np.array([[0,0,1]] * len(self.base_vectors))

        self.cube_poly = Poly3DCollection(self.your_mesh.vectors, alpha=1.0, linewidth=0)
        self.cube_poly.set_facecolor('#A0A0A0' if self.is_stl_loaded else '#FF4444')
        self.ax.add_collection3d(self.cube_poly)

        self.arrow_len = 1.2 
        self.x_arrow = self.ax.quiver(0, 0, 0, self.arrow_len, 0, 0, color='#FF4444', linewidth=3)
        self.y_arrow = self.ax.quiver(0, 0, 0, 0, self.arrow_len, 0, color='#44FF44', linewidth=3)
        self.z_arrow = self.ax.quiver(0, 0, 0, 0, 0, self.arrow_len, color='#4444FF', linewidth=3)

        label_dist = 1.4
        self.lbl_x = self.ax.text(label_dist, 0, 0, 'X', color='#FF4444', fontsize=12, fontweight='bold')
        self.lbl_y = self.ax.text(0, label_dist, 0, 'Y', color='#44FF44', fontsize=12, fontweight='bold')
        self.lbl_z = self.ax.text(0, 0, label_dist, 'Z', color='#4444FF', fontsize=12, fontweight='bold')
        self.labels = [self.lbl_x, self.lbl_y, self.lbl_z]
        self.label_pos = np.array([[label_dist, 0, 0], [0, label_dist, 0], [0, 0, label_dist]])

    def rotate_matrix(self, data, roll, pitch):
        roll_rad, pitch_rad = np.radians(roll), np.radians(pitch)
        Rx = np.array([[1, 0, 0], [0, np.cos(roll_rad), -np.sin(roll_rad)], [0, np.sin(roll_rad), np.cos(roll_rad)]])
        Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)], [0, 1, 0], [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
        R = Ry @ Rx.T
        return np.dot(data, R.T)

    def set_angles(self, roll, pitch):
        self.cube_poly.set_verts(self.rotate_matrix(self.base_vectors, roll, pitch))
        if self.is_stl_loaded:
            brightness = np.clip(np.dot(self.rotate_matrix(self.base_normals, roll, pitch), self.light_dir), 0.4, 1.0)
            self.cube_poly.set_facecolors(np.outer(brightness, np.array([0.7, 0.7, 0.75])))

        self.x_arrow.remove(); self.y_arrow.remove(); self.z_arrow.remove()
        base_ax = np.array([[self.arrow_len, 0, 0], [0, self.arrow_len, 0], [0, 0, self.arrow_len]])
        rot_ax = self.rotate_matrix(base_ax, roll, pitch)
        self.x_arrow = self.ax.quiver(0, 0, 0, rot_ax[0][0], rot_ax[0][1], rot_ax[0][2], color='#FF4444', linewidth=3)
        self.y_arrow = self.ax.quiver(0, 0, 0, rot_ax[1][0], rot_ax[1][1], rot_ax[1][2], color='#44FF44', linewidth=3)
        self.z_arrow = self.ax.quiver(0, 0, 0, rot_ax[2][0], rot_ax[2][1], rot_ax[2][2], color='#4444FF', linewidth=3)

        rot_lbls = self.rotate_matrix(self.label_pos, roll, pitch)
        for i, lbl in enumerate(self.labels):
            lbl.set_position((rot_lbls[i][0], rot_lbls[i][1]))
            lbl.set_3d_properties(rot_lbls[i][2])
        self.draw()

# 2. SENSOR GRAPH CANVAS
class SensorGraphCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 12), facecolor=COLOR_BG_CARD)
        self.ax_accel = self.fig.add_subplot(311)
        self.ax_gyro = self.fig.add_subplot(312)
        self.ax_mag = self.fig.add_subplot(313)
        
        for ax in [self.ax_accel, self.ax_gyro, self.ax_mag]:
            ax.set_facecolor(COLOR_BG_GRAPH)
            for spine in ax.spines.values(): spine.set_color('#A69A88'); spine.set_linewidth(1.0)
            ax.grid(True, alpha=0.28, color='#756B5E', linestyle='--')
            # Dark labels and grid styling complement the warm graph background.
            ax.tick_params(axis='both', which='both', colors=COLOR_TEXT_MAIN,
                           labelsize=8, pad=2)
            ax.xaxis.label.set_color(COLOR_TEXT_MAIN)
            ax.yaxis.label.set_color(COLOR_TEXT_MAIN)
            
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.08, hspace=0.4)
        super().__init__(self.fig)
        self.setStyleSheet("background-color: transparent;")

# 3. MAIN DASHBOARD
class DashboardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.df = None
        self.current_row = 0
        self.window_size = WINDOW_SIZE
        self.time_col_name = None 
        
        # Real-time serial data storage
        self.realtime_data = []
        self.serial_port = None
        self.is_serial_active = False
        
        self.setWindowTitle("CUBESAT TELEMETRY SOFTWARE")
        self.resize(1800, 1000)
        self.setStyleSheet(f"QMainWindow {{ background-color: {COLOR_BG_MAIN}; }}") 

        self.central_widget = QWidget()
        self.central_widget.setStyleSheet(f"background-color: {COLOR_BG_MAIN};")
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        # --- HEADER SECTION ---
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_BG_CARD};
                border-radius: 10px;
                border: 1px solid {COLOR_BORDER};
            }}
        """)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 10, 20, 10)

        # 1. LOGO
        self.logo_label = QLabel()
        self.logo_label.setStyleSheet(f"""
            QLabel {{
                background-color: {COLOR_BG_CARD}; 
                padding: 5px;
                border-radius: 8px;
                border: 1px solid {COLOR_BORDER};
            }}
        """)
        try:
            pixmap = QPixmap(LOGO_FILENAME) 
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaledToHeight(100, Qt.TransformationMode.SmoothTransformation)
                self.logo_label.setPixmap(scaled_pixmap)
                header_layout.addWidget(self.logo_label)
        except: pass
 
        header_layout.addStretch()

        # 3. TITLE
        title = QLabel("CUBESAT TELEMETRY SOFTWARE")
        title.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLOR_TEXT_MAIN}; border: none;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        self.main_layout.addWidget(header_frame)
        
        # --- MAIN CONTENT ---
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # LEFT: GRAPHS
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = SensorGraphCanvas(self)
        left_layout.addWidget(self.canvas)
        content_layout.addWidget(left_container, stretch=11)
        
        # RIGHT: CUBE + DATA 
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(20)
        
        # Cube (on top)
        self.cube_widget = Cube3DCanvas()
        self.cube_widget.setMinimumHeight(300) 
        self.cube_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        cube_container = QFrame()
        cube_container.setStyleSheet(f"background-color: {COLOR_BG_CUBE}; border-radius: 12px; border: 1px solid {COLOR_CUBE_BORDER};")
        cube_layout = QVBoxLayout(cube_container)
        cube_layout.setContentsMargins(10,10,10,10)
        cube_layout.addWidget(self.cube_widget)
        right_layout.addWidget(cube_container, stretch=6)
        
        # Data Grid (below cube)
        data_container = QWidget()
        data_layout = QGridLayout(data_container)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(15)
        
        # --- PRESSURE / HUMIDITY AND TEMPERATURE BOXES ---
        self.press_hum_frame, self.press_value, self.hum_value = self.create_dual_box("Pressure", "mbar", "Humidity", "%")
        data_layout.addWidget(self.press_hum_frame, 0, 0)
        
        # Temperature now uses the full width of its card; Power is not displayed.
        self.temp_frame, self.temp_value = self.create_single_box("Temperature", "°C")
        data_layout.addWidget(self.temp_frame, 0, 1)

        # Original bottom boxes
        self.gps_frame, self.lat_value, self.lon_value = self.create_gps_box()
        data_layout.addWidget(self.gps_frame, 1, 0)
        
        self.light_frame, self.light_vals = self.create_light_box()
        data_layout.addWidget(self.light_frame, 1, 1)

        right_layout.addWidget(data_container, stretch=1)
        content_layout.addWidget(right_container, stretch=9)
        self.main_layout.addLayout(content_layout)
        
        # --- FOOTER SECTION ---
        footer_layout = QHBoxLayout()
        footer_layout.addStretch()
        
        serial_label = QLabel("Port:")
        serial_label.setStyleSheet(f"color: {COLOR_TEXT_MAIN}; font-size: 14px; font-weight: bold;")
        footer_layout.addWidget(serial_label)
        
        self.port_input = QLineEdit(DEFAULT_SERIAL_PORT)
        self.port_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLOR_BG_CARD}; 
                color: {COLOR_TEXT_MAIN}; 
                padding: 8px; 
                border-radius: 5px;
                border: 1px solid {COLOR_BORDER};
                font-size: 14px;
            }}
        """)
        self.port_input.setFixedWidth(100)
        footer_layout.addWidget(self.port_input)
        
        baud_label = QLabel("Baud:")
        baud_label.setStyleSheet(f"color: {COLOR_TEXT_MAIN}; font-size: 14px; font-weight: bold;")
        footer_layout.addWidget(baud_label)
        
        self.baud_input = QLineEdit(str(DEFAULT_BAUD_RATE))
        self.baud_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLOR_BG_CARD}; 
                color: {COLOR_TEXT_MAIN}; 
                padding: 8px; 
                border-radius: 5px;
                border: 1px solid {COLOR_BORDER};
                font-size: 14px;
            }}
        """)
        self.baud_input.setFixedWidth(80)
        footer_layout.addWidget(self.baud_input)
        
        footer_layout.addSpacing(20)
        
        self.connect_btn = QPushButton("Connect Serial")
        self.connect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563EB; color: white; font-weight: bold; font-family: 'Segoe UI';
                font-size: 16px; padding: 12px 40px; border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #1D4ED8; }
        """)
        self.connect_btn.clicked.connect(self.connect_serial)
        footer_layout.addWidget(self.connect_btn)
        
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #10B981; color: white; font-weight: bold; font-family: 'Segoe UI';
                font-size: 16px; padding: 12px 40px; border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #059669; }
        """)
        self.export_btn.clicked.connect(self.export_csv)
        footer_layout.addWidget(self.export_btn)
        
        footer_layout.addSpacing(20)
        
        self.status_label = QLabel("Status: Not Connected")
        self.status_label.setStyleSheet(f"color: {COLOR_TEXT_SUB}; font-size: 14px; font-weight: bold;")
        footer_layout.addWidget(self.status_label)
        
        footer_layout.addStretch()
        self.main_layout.addLayout(footer_layout)
        
        self.serial_timer = QTimer()
        self.serial_timer.timeout.connect(self.read_serial_data)

    def connect_serial(self):
        if self.is_serial_active:
            self.serial_timer.stop()
            if self.serial_port:
                self.serial_port.close()
            self.is_serial_active = False
            self.connect_btn.setText("Connect Serial")
            self.connect_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2563EB; color: white; font-weight: bold; font-family: 'Segoe UI';
                    font-size: 16px; padding: 12px 40px; border-radius: 8px; border: none;
                }
                QPushButton:hover { background-color: #1D4ED8; }
            """)
            self.status_label.setText("Status: Disconnected")
            self.status_label.setStyleSheet(f"color: {COLOR_TEXT_SUB}; font-size: 14px; font-weight: bold;")
        else:
            try:
                port = self.port_input.text()
                baud = int(self.baud_input.text())
                self.serial_port = serial.Serial(port, baud, timeout=0.1)
                self.is_serial_active = True
                self.realtime_data = []
                self.serial_timer.start(ANIMATION_SPEED_MS)
                self.connect_btn.setText("Disconnect")
                self.connect_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #EF4444; color: white; font-weight: bold; font-family: 'Segoe UI';
                        font-size: 16px; padding: 12px 40px; border-radius: 8px; border: none;
                    }
                    QPushButton:hover { background-color: #DC2626; }
                """)
                self.status_label.setText(f"Status: Connected ({port})")
                self.status_label.setStyleSheet("color: #10B981; font-size: 14px; font-weight: bold;")
                QMessageBox.information(self, "Success", f"Connected to {port} at {baud} baud")
            except Exception as e:
                QMessageBox.critical(self, "Connection Error", f"Could not connect to serial port:\n{str(e)}")

    def read_serial_data(self):
        """Read real-time data from serial port"""
        if not self.serial_port or not self.is_serial_active:
            return
            
        try:
            line = self.serial_port.readline().decode().strip()
            if line:
                parts = line.split(',')
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                data_row = {'Timestamp': timestamp}
                
                if len(parts) >= 2:
                    data_row['Roll'] = float(parts[0])
                    data_row['Pitch'] = float(parts[1])
                    roll, pitch = float(parts[0]), float(parts[1])
                else:
                    return
                
                if len(parts) >= 5:
                    data_row['Ax'] = float(parts[2])
                    data_row['Ay'] = float(parts[3])
                    data_row['Az'] = float(parts[4])
                if len(parts) >= 8:
                    data_row['Gx'] = float(parts[5])
                    data_row['Gy'] = float(parts[6])
                    data_row['Gz'] = float(parts[7])
                if len(parts) >= 11:
                    data_row['Mx'] = float(parts[8])
                    data_row['My'] = float(parts[9])
                    data_row['Mz'] = float(parts[10])
                if len(parts) >= 12:
                    data_row['Pressure(Pa)'] = float(parts[11])
                if len(parts) >= 13:
                    data_row['Temp(C)'] = float(parts[12])
                if len(parts) >= 14:
                    data_row['Lat'] = float(parts[13])
                if len(parts) >= 15:
                    data_row['Long'] = float(parts[14])
                if len(parts) >= 19:
                    data_row['LDR-W'] = float(parts[15])
                    data_row['LDR-E'] = float(parts[16])
                    data_row['LDR-N'] = float(parts[17])
                    data_row['LDR-S'] = float(parts[18])
                
                # Parsing Humidity and Power
                if len(parts) >= 20:
                    data_row['Humidity(%)'] = float(parts[19])
                if len(parts) >= 21:
                    data_row['Power(mW)'] = float(parts[20])
                
                self.realtime_data.append(data_row)
                self.cube_widget.set_angles(roll, pitch)
                self.update_metrics_realtime(data_row)
                
                if len(self.realtime_data) > 0:
                    start_idx = max(0, len(self.realtime_data) - self.window_size)
                    df_window = pd.DataFrame(self.realtime_data[start_idx:])
                    self.update_graphs(df_window, start_idx)
                
                self.status_label.setText(f"Status: Connected | Data Points: {len(self.realtime_data)}")
                
        except ValueError as e:
            print(f"Data parsing error: {e} | Line: {line}")
        except Exception as e:
            print(f"Serial read error: {e}")

    def update_metrics_realtime(self, data_row):
        """Update metric boxes with real-time data"""
        if 'Pressure(Pa)' in data_row:
            self.press_value.setText(f"{data_row['Pressure(Pa)'] * 0.01:.1f}")
        if 'Temp(C)' in data_row:
            self.temp_value.setText(f"{data_row['Temp(C)']:.1f}")
            
        if 'Humidity(%)' in data_row:
            self.hum_value.setText(f"{data_row['Humidity(%)']:.1f}")
        if 'Lat' in data_row:
            self.lat_value.setText(f"{data_row['Lat']:.2f}")
        if 'Long' in data_row:
            self.lon_value.setText(f"{data_row['Long']:.2f}")
        
        ldr_keys = ['LDR-W', 'LDR-E', 'LDR-N', 'LDR-S']
        for i, key in enumerate(ldr_keys):
            if key in data_row and i < len(self.light_vals):
                self.light_vals[i].setText(f"{data_row[key]:.0f}")

    def export_csv(self):
        if len(self.realtime_data) == 0:
            QMessageBox.warning(self, "No Data", "No data to export! Connect to serial port first.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"cubesat_realtime_{timestamp}.csv"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV File", default_filename, "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                df = pd.DataFrame(self.realtime_data)
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Export Successful", f"Exported {len(self.realtime_data)} data points to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{str(e)}")

    def create_dual_box(self, title1, unit1, title2, unit2):
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background-color: {COLOR_BG_CARD}; border-radius: 12px; border: 1px solid {COLOR_BORDER}; }}")
        main_layout = QHBoxLayout(frame)
        main_layout.setContentsMargins(15, 15, 15, 15)

        def create_section(title, unit):
            layout = QVBoxLayout()
            
            lbl_title = QLabel(f"{title}({unit})")
            lbl_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            lbl_title.setStyleSheet(f"border: none; color: {COLOR_TEXT_MAIN};")
            
            val_label = QLabel("--")
            val_label.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
            val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val_label.setStyleSheet(f"border: none; color: {COLOR_TEXT_MAIN};")
            
            layout.addWidget(lbl_title)
            layout.addWidget(val_label)
            return layout, val_label

        left_layout, left_val = create_section(title1, unit1)
        right_layout, right_val = create_section(title2, unit2)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setStyleSheet(f"color: {COLOR_BORDER}; margin-left: 10px; margin-right: 10px;")

        main_layout.addLayout(left_layout)
        main_layout.addWidget(divider)
        main_layout.addLayout(right_layout)

        return frame, left_val, right_val

    def create_single_box(self, title, unit):
        """Create one metric section that fills the entire card width."""
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background-color: {COLOR_BG_CARD}; "
            f"border-radius: 12px; border: 1px solid {COLOR_BORDER}; }}"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 15, 15, 15)

        title_label = QLabel(f"{title}({unit})")
        title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title_label.setStyleSheet(f"border: none; color: {COLOR_TEXT_MAIN};")

        value_label = QLabel("--")
        value_label.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet(f"border: none; color: {COLOR_TEXT_MAIN};")

        layout.addWidget(title_label)
        layout.addStretch()
        layout.addWidget(value_label)
        layout.addStretch()

        return frame, value_label

    def create_gps_box(self):
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background-color: {COLOR_BG_CARD}; border-radius: 12px; border: 1px solid {COLOR_BORDER}; }}")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 15, 20, 15)
        
        lbl_title = QLabel("GPS")
        lbl_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        lbl_title.setStyleSheet(f"border: none; color: {COLOR_TEXT_MAIN};")
        
        # Aligned correctly to the top-left
        layout.addWidget(lbl_title, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        right_stack = QVBoxLayout(); right_stack.setSpacing(5)
        def make_row(label_txt):
            row = QHBoxLayout()
            v = QLabel("--")
            v.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
            v.setStyleSheet(f"border: none; color: {COLOR_TEXT_MAIN};")
            v.setAlignment(Qt.AlignmentFlag.AlignCenter) 
            l = QLabel(label_txt)
            l.setStyleSheet(f"border: none; color: {COLOR_TEXT_SUB};")
            l.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            l.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addStretch(); row.addWidget(v); row.addStretch(); row.addWidget(l)
            return row, v

        lat_row, lat_val = make_row("Lat")
        lon_row, lon_val = make_row("Long")
        right_stack.addLayout(lat_row); right_stack.addLayout(lon_row)
        layout.addLayout(right_stack)
        return frame, lat_val, lon_val

    def create_light_box(self):
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background-color: {COLOR_BG_CARD}; border-radius: 12px; border: 1px solid {COLOR_BORDER}; }}")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 15, 20, 15)
        
        lbl_title = QLabel("Light")
        lbl_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        lbl_title.setStyleSheet(f"border: none; color: {COLOR_TEXT_MAIN};")
        layout.addWidget(lbl_title)
        
        val_row = QHBoxLayout(); val_row.setSpacing(20); val_row.addStretch()
        labels = ["W", "E", "N", "S"]; val_labels = []
        for name in labels:
            v_layout = QVBoxLayout(); v_layout.setSpacing(2)
            v_lbl = QLabel("--"); v_lbl.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold)); v_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); v_lbl.setStyleSheet(f"border: none; color: {COLOR_TEXT_MAIN};")
            sub_lbl = QLabel(name); sub_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold)); sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); sub_lbl.setStyleSheet(f"border: none; color: {COLOR_TEXT_SUB};")
            v_layout.addWidget(v_lbl); v_layout.addWidget(sub_lbl)
            val_row.addLayout(v_layout); val_labels.append(v_lbl)
        val_row.addStretch(); layout.addLayout(val_row)
        return frame, val_labels

    def update_graphs(self, df, start_idx):
        self.canvas.ax_accel.clear(); self.canvas.ax_gyro.clear(); self.canvas.ax_mag.clear()
        x = df.index

        def plot_sensor(ax, cols, label):
            # Consistent axis colours on all three sensor graphs: X red, Y matte green, Z dark blue.
            colors = ['#C62828', '#355E3B', '#1F3A5F']
            if all(c in df.columns for c in cols):
                ax.plot(x, df[cols[0]], c=colors[0], lw=2, label='X')
                ax.plot(x, df[cols[1]], c=colors[1], lw=2, label='Y')
                ax.plot(x, df[cols[2]], c=colors[2], lw=2, label='Z')
                leg = ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
                for text in leg.get_texts(): text.set_color(COLOR_TEXT_MAIN)
                leg.get_frame().set_facecolor(COLOR_BG_CARD)
                leg.get_frame().set_edgecolor(COLOR_BORDER)
            ax.set_title(label, color=COLOR_TEXT_MAIN, fontsize=12, fontweight='bold', pad=10)
            ax.set_facecolor(COLOR_BG_GRAPH)
            ax.grid(True, alpha=0.28, color='#756B5E')
            # Keep axis numbers readable against the cream area around each plot.
            ax.tick_params(axis='both', which='both', colors=COLOR_TEXT_MAIN)
            ax.xaxis.label.set_color(COLOR_TEXT_MAIN)
            ax.yaxis.label.set_color(COLOR_TEXT_MAIN)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plot_sensor(self.canvas.ax_accel, ['Ax', 'Ay', 'Az'], 'Accelerometer')
        plot_sensor(self.canvas.ax_gyro, ['Gx', 'Gy', 'Gz'], 'Gyroscope')
        plot_sensor(self.canvas.ax_mag, ['Mx', 'My', 'Mz'], 'Magnetometer')
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DashboardWindow()
    window.show()
    sys.exit(app.exec())