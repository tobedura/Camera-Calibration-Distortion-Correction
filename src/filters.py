import cv2
import numpy as np

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_canny(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    edges_combined = None
    for i in range(3):
        channel = _clahe.apply(lab[:, :, i])
        blurred = cv2.GaussianBlur(channel, (5, 5), 0)
        v = np.median(blurred)
        lower = int(max(0, 0.5 * v))
        upper = int(min(255, 1.5 * v))
        edges = cv2.Canny(blurred, lower, upper)
        if edges_combined is None:
            edges_combined = edges
        else:
            edges_combined = cv2.bitwise_or(edges_combined, edges)

    return cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2BGR)


_dc_map1 = None
_dc_map2 = None
_dc_cache_key = None


def apply_distortion_correction(frame: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    global _dc_map1, _dc_map2, _dc_cache_key

    h, w = frame.shape[:2]
    cache_key = (w, h, camera_matrix.tobytes(), dist_coeffs.tobytes())

    if _dc_cache_key != cache_key:
        _dc_map1, _dc_map2 = cv2.initUndistortRectifyMap(
            camera_matrix, dist_coeffs, None, None, (w, h), cv2.CV_32FC1
        )
        _dc_cache_key = cache_key

    corrected = cv2.remap(frame, _dc_map1, _dc_map2, interpolation=cv2.INTER_LINEAR)
    return corrected
