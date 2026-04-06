# Camera Calibration & Distortion Correction

OpenCV + PyQt5 기반 카메라 캘리브레이션 및 렌즈 왜곡 보정 프로그램

## 기능

- 웹캠 실시간 영상 녹화 (.avi)
- 좌우 분할 화면 (원본 / 필터 적용)
- Canny Edge Detection 필터
- 체스보드 기반 카메라 캘리브레이션
  - `findChessboardCorners` + `drawChessboardCorners`로 프레임 선택
  - `calibrateCamera`로 카메라 매트릭스, 왜곡 계수 산출
- `initUndistortRectifyMap` + `remap` 방식의 렌즈 왜곡 보정 필터

## 실행

```bash
uv sync
uv run python src/main.py
```

## 카메라 캘리브레이션 결과

<img src="assets/Calibration Result.png" height="300">

## 렌즈 왜곡 보정 결과

![Distortion Correction](assets/Distortion%20Correction.png)
