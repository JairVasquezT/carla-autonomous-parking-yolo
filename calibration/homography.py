import cv2
import numpy as np

# -------------------------------
# IMAGE POINTS (PIXELS)
# -------------------------------
puntos_imagen = np.array([
    [640,286],
    [498,218], [592,218], [687,218], [782,218],
    [465,370], [581,370], [697,370], [813,370],

    [431,161], [535,161], [640,161], [744,161], [848,161],
    [326,475], [483,475], [639,475], [796,475], [953,475],

    [416,115], [497,115], [577,115], [622,115],
    [657,115], [702,115], [782,115], [863,115],

    [226,605], [374,605], [523,605], [607,605],
    [672,605], [755,605], [904,605], [1053,605]
], dtype='float32')


# -------------------------------
# CARLA POINTS (METERS)
# -------------------------------
puntos_mundo = np.array([
    [0,0],
    [2.5,-3.75], [2.5,-1.25], [2.5,1.25], [2.5,3.75],
    [-2.5,-3.75], [-2.5,-1.25], [-2.5,1.25], [-2.5,3.75],

    [5.0,-6.0], [5.0,-3.0], [5.0,0.0], [5.0,3.0], [5.0,6.0],
    [-5.0,-6.0], [-5.0,-3.0], [-5.0,0.0], [-5.0,3.0], [-5.0,6.0],

    [7.4,-6.95], [7.4,-4.45], [7.4,-1.95], [7.4,-0.55],
    [7.4,0.55], [7.4,1.95], [7.4,4.45], [7.4,6.95],

    [-7.4,-6.95], [-7.4,-4.45], [-7.4,-1.95], [-7.4,-0.55],
    [-7.4,0.55], [-7.4,1.95], [-7.4,4.45], [-7.4,6.95]
], dtype='float32')


# -------------------------------
# COMPUTE HOMOGRAPHY (USING RANSAC)
# -------------------------------
H, mask = cv2.findHomography(puntos_imagen, puntos_mundo, cv2.RANSAC)

print("Computed homography matrix:")
print(H)


# -------------------------------
# TEST FUNCTION
# -------------------------------
def pixel_a_metros(px, py, matriz_h):
    punto = np.array([[[px, py]]], dtype='float32')
    convertido = cv2.perspectiveTransform(punto, matriz_h)
    return convertido[0][0]


# -------------------------------
# VALIDATION
# -------------------------------
errores = []

for (px, py), (X_real, Y_real) in zip(puntos_imagen, puntos_mundo):
    X_est, Y_est = pixel_a_metros(px, py, H)
    err = np.sqrt((X_est - X_real)**2 + (Y_est - Y_real)**2)
    errores.append(err)

print(f"\nMean error: {np.mean(errores):.4f} m")
print(f"Max error: {np.max(errores):.4f} m")


# -------------------------------
# SIMPLE TEST
# -------------------------------
test_x, test_y = pixel_a_metros(640, 286, H)
print(f"\nTest (0,0): ({test_x:.2f}, {test_y:.2f})")