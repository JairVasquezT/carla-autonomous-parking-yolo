import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- 1. PATHS AND DLL SETUP ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
carla_root = os.environ.get('CARLA_ROOT', r'C:\carla')
build_path = os.path.join(carla_root, 'Build')
if os.path.exists(build_path):
    for root, dirs, files in os.walk(build_path):
        if any(f.endswith('.dll') for f in files):
            try: os.add_dll_directory(root)
            except: pass

sys.path.append(os.path.join(carla_root, 'PythonAPI', 'carla'))
import carla

# --- 2. HOMOGRAPHY MATRIX (CALIBRATION) ---
# Computed from the 35-point calibration
H = np.array([
    [ -1.35928836e-08, -5.39979638e-02,  1.54643018e+01],
    [ 3.89898819e-02,  2.71380886e-05, -2.49401315e+01],
    [ 1.84855595e-07,  2.18687425e-03,  1.00000000e+00]
])

# Citroën C3 dimensions (meters: length, width, height)
L, W, HT = 4.0, 1.8, 1.5

test_points = [
    (0, 0, 0), (2.5, -3.75, 0), (2.5, -1.25, 0), (2.5, 1.25, 0),
    (2.5, 3.75, 0), (-2.5, -3.75, 180), (-2.5, -1.25, 180), (-2.5, 1.25, 180), (-2.5, 3.75, 180),
    # 4 extra points with different rotations
    (0, 5.0, 45), (4.0, 0, 90), (-4.0, 2.0, 135), (0, -5.0, 270)
]

frames_per_point = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('parking_evaluation.mp4', fourcc, 20.0, (1280, 720)) # Must match the camera resolution


def pixel_a_metros(px, py, matrix_h):
    """Transform pixels into CARLA X, Y coordinates using the homography"""
    punto = np.array([[[px, py]]], dtype='float32')
    res = cv2.perspectiveTransform(punto, matrix_h)
    return res[0][0] # [X, Y] in meters

def proyectar_punto_3d(x, y, z, world_2_cam, K):
    """Project a 3D world point onto the 2D image using the camera"""
    p_world = np.array([x, y, z, 1])
    p_cam = np.dot(world_2_cam, p_world)
    # CARLA to OpenCV coordinate conversion
    p_cv = np.array([p_cam[1], -p_cam[2], p_cam[0]])
    if p_cv[2] > 0.1:
        p_2d = np.dot(K, p_cv)
        p_2d /= p_2d[2]
        return (int(p_2d[0]), int(p_2d[1]))
    return None

def obtener_puntos_cubo_mundo(centro_x, centro_y, yaw_deg):
    """Generate the 8 box corners in CARLA 3D world space"""
    rad = np.radians(yaw_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]]) # Rotation matrix
    
    l, w = L/2, W/2
    puntos_mundo = []
    
    for i in range(8):
        # Local coordinates (relative to the center)
        lx = l if i in [0, 1, 4, 5] else -l
        ly = w if i in [0, 3, 4, 7] else -w
        lz = 0 if i < 4 else HT # Base or roof
        
        # Rotate and translate
        rot = np.dot(R, [lx, ly])
        puntos_mundo.append([centro_x + rot[0], centro_y + rot[1], lz])
        
    return np.array(puntos_mundo)

def draw_reconstructed_cube(image, points_2d, color=(0, 255, 255)):
    """Draw the edges of the reconstructed box"""
    if len(points_2d) != 8: return image
    for i in range(4):
        # Base
        cv2.line(image, points_2d[i], points_2d[(i+1)%4], color, 2)
        # Roof
        cv2.line(image, points_2d[i+4], points_2d[((i+1)%4)+4], color, 2)
        # Pillars
        cv2.line(image, points_2d[i], points_2d[i+4], color, 2)
    return image



def main():
    # Load YOLO
    print("Loading YOLO...")
    model = YOLO(os.path.join(PROJECT_ROOT, 'models', 'best.pt'))

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Original camera configuration
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '1280')
    cam_bp.set_attribute('image_size_y', '720')
    cam_tr = carla.Transform(carla.Location(x=-11.0, y=0.0, z=11.0), carla.Rotation(pitch=-52.0))
    camera = world.spawn_actor(cam_bp, cam_tr)
    
    # Intrinsic matrix K (FOV 90, 1280x720)
    f = 1280 / (2.0 * np.tan(90.0 * np.pi / 360.0))
    K = np.array([[f, 0, 640], [0, f, 360], [0, 0, 1]])

    # Spawn the car at a random position for testing
    bp_c3 = world.get_blueprint_library().find('vehicle.citroen.c3')
    vehicle = world.spawn_actor(bp_c3, carla.Transform(carla.Location(x=0.5, y=-3.25, z=0.1), carla.Rotation(yaw=45)))

    image_data = {'frame': None}
    camera.listen(lambda img: image_data.update({'frame': np.reshape(np.frombuffer(img.raw_data, dtype="uint8"), (720, 1280, 4))[:, :, :3]}))

    try:
        for x_p, y_p, yaw_p in test_points:
            # 1. Teleport the vehicle to the new spawn point
            new_transform = carla.Transform(
                carla.Location(x=x_p, y=y_p, z=0.1),
                carla.Rotation(yaw=yaw_p)
            )
            vehicle.set_transform(new_transform)
            
            # Short pause so CARLA physics can settle
            for _ in range(10): world.tick()
            
            print(f"Recording position: x={x_p}, y={y_p}, yaw={yaw_p}")

            # 2. Record 30 frames for this spawn point
            for _ in range(30):
                world.tick()
                
                if image_data['frame'] is not None:
                    frame = image_data['frame'].copy()
                    results = model(frame, conf=0.5, verbose=False)
                    
                    if len(results[0].boxes) > 0:
                        box = results[0].boxes[0].xyxy[0].cpu().numpy()
                        x_min_bb, y_min_bb, x_max_bb, y_max_bb = box
                        
                        cv2.rectangle(frame, (int(x_min_bb), int(y_min_bb)), (int(x_max_bb), int(y_max_bb)), (255, 0, 0), 2)
                        
                        cx_yolo_objetivo = (x_min_bb + x_max_bb) / 2.0
                        y_base_objetivo = y_max_bb
                        
                        yaw = vehicle.get_transform().rotation.yaw
                        world_2_cam = np.array(camera.get_transform().get_inverse_matrix())
                        
                        punto_inicial = pixel_a_metros(cx_yolo_objetivo, y_base_objetivo, H)
                        centro_x, centro_y = punto_inicial[0], punto_inicial[1]
                        
                        # --- POSITION OPTIMIZATION ---
                        eps = 0.1
                        for iteracion in range(6):
                            p_3d = obtener_puntos_cubo_mundo(centro_x, centro_y, yaw)
                            p_2d = [proyectar_punto_3d(p[0], p[1], p[2], world_2_cam, K) for p in p_3d]
                            
                            if None in p_2d: break
                            
                            cx_actual = sum(p[0] for p in p_2d) / 8.0
                            y_bajo_actual = max(p[1] for p in p_2d)
                            
                            err_x = cx_yolo_objetivo - cx_actual
                            err_y = y_base_objetivo - y_bajo_actual
                            
                            if abs(err_x) < 1.0 and abs(err_y) < 1.0: break
                                
                            # Derivatives (X perturbation)
                            p_3d_dx = obtener_puntos_cubo_mundo(centro_x + eps, centro_y, yaw)
                            p_2d_dx = [proyectar_punto_3d(p[0], p[1], p[2], world_2_cam, K) for p in p_3d_dx]
                            cx_dx = sum(p[0] for p in p_2d_dx) / 8.0
                            ybajo_dx = max(p[1] for p in p_2d_dx)
                            
                            # Derivatives (Y perturbation)
                            p_3d_dy = obtener_puntos_cubo_mundo(centro_x, centro_y + eps, yaw)
                            p_2d_dy = [proyectar_punto_3d(p[0], p[1], p[2], world_2_cam, K) for p in p_3d_dy]
                            cx_dy = sum(p[0] for p in p_2d_dy) / 8.0
                            ybajo_dy = max(p[1] for p in p_2d_dy)
                            
                            J11, J12 = (cx_dx - cx_actual) / eps, (cx_dy - cx_actual) / eps
                            J21, J22 = (ybajo_dx - y_bajo_actual) / eps, (ybajo_dy - y_bajo_actual) / eps
                            
                            det = (J11 * J22) - (J12 * J21)
                            if abs(det) < 1e-6: break
                            
                            centro_x += (J22 * err_x - J12 * err_y) / det * 0.8
                            centro_y += (-J21 * err_x + J11 * err_y) / det * 0.8

                        # --- FINAL DRAWING ---
                        puntos_3d_final = obtener_puntos_cubo_mundo(centro_x, centro_y, yaw)
                        puntos_2d_final = [proyectar_punto_3d(p[0], p[1], p[2], world_2_cam, K) for p in puntos_3d_final]
                        puntos_2d_final = [p for p in puntos_2d_final if p is not None]

                        if len(puntos_2d_final) == 8:
                            draw_reconstructed_cube(frame, puntos_2d_final, color=(0, 255, 0))
                            
                            base_pts = puntos_3d_final[:4]
                            centro_real_x, centro_real_y = np.mean(base_pts[:, 0]), np.mean(base_pts[:, 1])
                            
                            cv2.putText(frame, f"Center: {centro_real_x:.2f}m, {centro_real_y:.2f}m",
                                        (int(x_min_bb), int(y_min_bb)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 3. Write frame to the video and show it in a window
                    out.write(frame)
                    cv2.imshow("Parking evaluation", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        out.release()
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()