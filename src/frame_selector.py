import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QComboBox, QSpinBox,
    QHBoxLayout, QVBoxLayout, QSlider, QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


class FrameSelectorWindow(QWidget):
    def __init__(self, output_dir: str):
        super().__init__()
        self.setWindowTitle("Frame Selector - Chessboard")
        self.setMinimumSize(900, 650)

        self._output_dir = output_dir
        self._save_dir = os.path.join(output_dir, "selected_image")
        self._cap = None
        self._total_frames = 0
        self._current_frame_index = 0
        self._current_frame = None
        self._selected_frames = {}  # {frame_index: frame(원본)}

        self._setup_ui()
        self._refresh_files()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # 상단: 파일 선택 + 체스보드 크기
        top_row = QHBoxLayout()

        top_row.addWidget(QLabel("파일:"))
        self.combo_file = QComboBox()
        self.combo_file.setMinimumWidth(200)
        self.combo_file.currentIndexChanged.connect(self._on_file_changed)
        top_row.addWidget(self.combo_file)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_files)
        top_row.addWidget(btn_refresh)

        top_row.addStretch()

        top_row.addWidget(QLabel("Rows:"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(2, 20)
        self.spin_rows.setValue(10)
        self.spin_rows.valueChanged.connect(self._update_display)
        top_row.addWidget(self.spin_rows)

        top_row.addWidget(QLabel("Cols:"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(2, 20)
        self.spin_cols.setValue(7)
        self.spin_cols.valueChanged.connect(self._update_display)
        top_row.addWidget(self.spin_cols)

        root.addLayout(top_row)

        # 영상 표시
        self.lbl_frame = QLabel("영상을 선택하세요")
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        self.lbl_frame.setStyleSheet("background: #111; color: #aaa;")
        self.lbl_frame.setMinimumHeight(400)
        root.addWidget(self.lbl_frame, stretch=1)

        # 슬라이더 + 프레임 정보
        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self.slider)

        self.lbl_frame_info = QLabel("0 / 0")
        self.lbl_frame_info.setFixedWidth(100)
        self.lbl_frame_info.setAlignment(Qt.AlignCenter)
        slider_row.addWidget(self.lbl_frame_info)

        root.addLayout(slider_row)

        # 하단 컨트롤
        ctrl_row = QHBoxLayout()

        self.lbl_detect = QLabel("")
        self.lbl_detect.setStyleSheet("font-size: 13px; font-weight: bold;")
        ctrl_row.addWidget(self.lbl_detect)

        ctrl_row.addStretch()

        self.lbl_selected = QLabel("선택: 0장")
        self.lbl_selected.setStyleSheet("font-size: 13px;")
        ctrl_row.addWidget(self.lbl_selected)

        self.btn_select = QPushButton("+ 선택")
        self.btn_select.setFixedHeight(36)
        self.btn_select.setEnabled(False)
        self.btn_select.clicked.connect(self._select_frame)
        ctrl_row.addWidget(self.btn_select)

        self.btn_deselect = QPushButton("- 해제")
        self.btn_deselect.setFixedHeight(36)
        self.btn_deselect.setEnabled(False)
        self.btn_deselect.clicked.connect(self._deselect_frame)
        ctrl_row.addWidget(self.btn_deselect)

        self.btn_save = QPushButton("저장")
        self.btn_save.setFixedHeight(36)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_frames)
        ctrl_row.addWidget(self.btn_save)

        root.addLayout(ctrl_row)

    def _refresh_files(self):
        self.combo_file.blockSignals(True)
        self.combo_file.clear()
        if not os.path.isdir(self._output_dir):
            self.combo_file.addItem("output 폴더 없음")
            self.combo_file.blockSignals(False)
            return
        avi_files = sorted(
            f for f in os.listdir(self._output_dir) if f.endswith(".avi")
        )
        if not avi_files:
            self.combo_file.addItem(".avi 파일 없음")
            self.combo_file.blockSignals(False)
            return
        for f in avi_files:
            self.combo_file.addItem(f)
        self.combo_file.blockSignals(False)
        self._open_video(avi_files[0])

    def _on_file_changed(self, index: int):
        filename = self.combo_file.currentText()
        if filename and not filename.endswith("없음"):
            self._open_video(filename)

    def _open_video(self, filename: str):
        if self._cap is not None:
            self._cap.release()
        path = os.path.join(self._output_dir, filename)
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            QMessageBox.critical(self, "오류", f"영상을 열 수 없습니다: {path}")
            return
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame_index = 0
        self._selected_frames.clear()
        self._update_selected_label()

        self.slider.setMaximum(max(0, self._total_frames - 1))
        self.slider.setValue(0)
        self._read_and_display(0)

    def _on_slider_changed(self, value: int):
        self._read_and_display(value)

    def _read_and_display(self, frame_index: int):
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._cap.read()
        if not ret:
            return
        self._current_frame_index = frame_index
        self._current_frame = frame.copy()
        self._update_display()

    def _update_display(self):
        if self._current_frame is None:
            return

        frame = self._current_frame
        display = frame.copy()
        rows = self.spin_rows.value()
        cols = self.spin_cols.value()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            cv2.drawChessboardCorners(display, (cols, rows), corners_refined, found)
            self.lbl_detect.setText("체스보드 검출됨")
            self.lbl_detect.setStyleSheet("color: green; font-size: 13px; font-weight: bold;")
            self.btn_select.setEnabled(True)
        else:
            self.lbl_detect.setText("체스보드 미검출")
            self.lbl_detect.setStyleSheet("color: red; font-size: 13px; font-weight: bold;")
            self.btn_select.setEnabled(False)

        is_selected = self._current_frame_index in self._selected_frames
        self.btn_deselect.setEnabled(is_selected)
        if is_selected:
            cv2.putText(display, "SELECTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self.lbl_frame_info.setText(
            f"{self._current_frame_index + 1} / {self._total_frames}"
        )

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.lbl_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_frame.setPixmap(pix)

    def _select_frame(self):
        if self._current_frame is not None:
            self._selected_frames[self._current_frame_index] = self._current_frame.copy()
            self._update_selected_label()
            self._update_display()

    def _deselect_frame(self):
        self._selected_frames.pop(self._current_frame_index, None)
        self._update_selected_label()
        self._update_display()

    def _update_selected_label(self):
        count = len(self._selected_frames)
        self.lbl_selected.setText(f"선택: {count}장")
        self.btn_save.setEnabled(count > 0)

    def _next_save_dir(self) -> str:
        base = os.path.join(self._output_dir, "selected_image")
        num = 1
        while os.path.isdir(f"{base}_{num:02d}"):
            num += 1
        return f"{base}_{num:02d}"

    def _save_frames(self):
        save_dir = self._next_save_dir()
        os.makedirs(save_dir)
        for i, (frame_index, frame) in enumerate(sorted(self._selected_frames.items())):
            path = os.path.join(save_dir, f"frame_{frame_index:05d}.png")
            cv2.imwrite(path, frame)
        QMessageBox.information(
            self, "저장 완료",
            f"{len(self._selected_frames)}장이 저장되었습니다.\n{save_dir}",
        )

    def closeEvent(self, event):
        if self._cap is not None:
            self._cap.release()
        event.accept()
