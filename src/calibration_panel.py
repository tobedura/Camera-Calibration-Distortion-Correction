import os
import glob
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QComboBox, QSpinBox,
    QHBoxLayout, QVBoxLayout, QFormLayout, QMessageBox, QGroupBox,
)


class CalibrationPanel(QWidget):
    def __init__(self, output_dir: str):
        super().__init__()
        self._output_dir = output_dir
        self._camera_matrix = None
        self._dist_coeffs = None
        self._setup_ui()
        self._refresh_folders()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # 폴더 선택
        root.addWidget(QLabel("폴더 선택:"))
        folder_row = QHBoxLayout()
        self.combo_folder = QComboBox()
        self.combo_folder.currentIndexChanged.connect(self._on_folder_changed)
        folder_row.addWidget(self.combo_folder)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_folders)
        folder_row.addWidget(btn_refresh)
        root.addLayout(folder_row)

        # 체스보드 크기
        board_row = QHBoxLayout()
        board_row.addWidget(QLabel("Rows:"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(2, 20)
        self.spin_rows.setValue(10)
        board_row.addWidget(self.spin_rows)
        board_row.addWidget(QLabel("Cols:"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(2, 20)
        self.spin_cols.setValue(7)
        board_row.addWidget(self.spin_cols)
        root.addLayout(board_row)

        # 결과 표시
        result_box = QGroupBox("Calibration Results")
        form = QFormLayout()

        self.lbl_rms = QLabel("0.000")
        form.addRow("RMS Error:", self.lbl_rms)

        self.lbl_fx = QLabel("0.000")
        form.addRow("fx:", self.lbl_fx)

        self.lbl_fy = QLabel("0.000")
        form.addRow("fy:", self.lbl_fy)

        self.lbl_cx = QLabel("0.000")
        form.addRow("cx:", self.lbl_cx)

        self.lbl_cy = QLabel("0.000")
        form.addRow("cy:", self.lbl_cy)

        self.lbl_k1 = QLabel("0.000")
        form.addRow("k1:", self.lbl_k1)

        self.lbl_k2 = QLabel("0.000")
        form.addRow("k2:", self.lbl_k2)

        self.lbl_p1 = QLabel("0.000")
        form.addRow("p1:", self.lbl_p1)

        self.lbl_p2 = QLabel("0.000")
        form.addRow("p2:", self.lbl_p2)

        self.lbl_k3 = QLabel("0.000")
        form.addRow("k3:", self.lbl_k3)

        self.lbl_images = QLabel("0 / 0")
        form.addRow("사용 이미지:", self.lbl_images)

        result_box.setLayout(form)
        root.addWidget(result_box)

        self.btn_frame_selector = QPushButton("Frame Selector")
        self.btn_frame_selector.setFixedHeight(36)
        root.addWidget(self.btn_frame_selector)

        root.addStretch()

    def _refresh_folders(self):
        self.combo_folder.blockSignals(True)
        self.combo_folder.clear()
        if not os.path.isdir(self._output_dir):
            self.combo_folder.addItem("폴더 없음")
            self.combo_folder.blockSignals(False)
            return
        folders = sorted(
            d for d in os.listdir(self._output_dir)
            if d.startswith("selected_image") and os.path.isdir(os.path.join(self._output_dir, d))
        )
        if not folders:
            self.combo_folder.addItem("폴더 없음")
            self.combo_folder.blockSignals(False)
            return
        for f in folders:
            self.combo_folder.addItem(f)
        self.combo_folder.blockSignals(False)
        self._run_calibration()

    def _on_folder_changed(self, index: int):
        folder_name = self.combo_folder.currentText()
        if folder_name and not folder_name.endswith("없음"):
            self._run_calibration()

    def _run_calibration(self):
        folder_name = self.combo_folder.currentText()
        if not folder_name or folder_name.endswith("없음"):
            return

        folder_path = os.path.join(self._output_dir, folder_name)
        image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))
        if not image_paths:
            self._reset_results()
            self.lbl_images.setText("0 / 0")
            return

        rows = self.spin_rows.value()
        cols = self.spin_cols.value()

        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

        obj_points = []
        img_points = []
        total = len(image_paths)
        used = 0

        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            if found:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                obj_points.append(objp)
                img_points.append(corners_refined)
                used += 1

        if used < 3:
            self._reset_results()
            self.lbl_images.setText(f"{used} / {total} (최소 3장 필요)")
            return

        h, w = gray.shape[:2]
        rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
            obj_points, img_points, (w, h), None, None
        )
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs

        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        k1, k2, p1, p2, k3 = dist_coeffs[0]

        self.lbl_rms.setText(f"{rms:.6f}")
        self.lbl_fx.setText(f"{fx:.3f}")
        self.lbl_fy.setText(f"{fy:.3f}")
        self.lbl_cx.setText(f"{cx:.3f}")
        self.lbl_cy.setText(f"{cy:.3f}")
        self.lbl_k1.setText(f"{k1:.6f}")
        self.lbl_k2.setText(f"{k2:.6f}")
        self.lbl_p1.setText(f"{p1:.6f}")
        self.lbl_p2.setText(f"{p2:.6f}")
        self.lbl_k3.setText(f"{k3:.6f}")
        self.lbl_images.setText(f"{used} / {total}")

    def _reset_results(self):
        self._camera_matrix = None
        self._dist_coeffs = None
        for lbl in (self.lbl_rms, self.lbl_fx, self.lbl_fy, self.lbl_cx, self.lbl_cy,
                    self.lbl_k1, self.lbl_k2, self.lbl_p1, self.lbl_p2, self.lbl_k3):
            lbl.setText("0.000")

    def get_calibration(self):
        return self._camera_matrix, self._dist_coeffs
