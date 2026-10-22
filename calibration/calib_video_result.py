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
    [ -7.72403192e-08, -5.42594896e-02,  1.55323124e+01],
    [ 3.91398322e-02,  1.91169448e-04, -2.50827893e+01],
    [3.47746953e-06,  2.20134695e-03,  1.00000000e+00]
])

def transformar_a_metros(px, py, matriz_h):
    """Convert image pixels to CARLA X, Y meters"""
    punto = np.array([[[px, py]]], dtype='float32')
    convertido = cv2.perspectiveTransform(punto, matriz_h)
    return convertido[0][0] # Returns [X_Carla, Y_Carla]

class VisualCalibrator:
    def __init__(self, camera_actor, width=1280, height=720, fov=90.0):
        self.camera = camera_actor
        self.W, self.H = width, height
        f = width / (2.0 * np.tan(fov * np.pi / 360.0))
        self.K = np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1]])

    def get_cube_points(self, vehicle):
        world_2_cam = np.array(self.camera.get_transform().get_inverse_matrix())
        bb = vehicle.bounding_box
        verts = bb.get_world_vertices(vehicle.get_transform())
        points_2d = []
        for v in verts:
            p_cam = np.dot(world_2_cam, np.array([v.x, v.y, v.z, 1]))
            p_cv = np.array([p_cam[1], -p_cam[2], p_cam[0]])
            if p_cv[2] > 0.1:
                p_2d = np.dot(self.K, p_cv)
                p_2d /= p_2d[2]
                points_2d.append((int(p_2d[0]), int(p_2d[1])))
        return points_2d

    def draw_3d_cube(self, image, points_2d, color=(0, 255, 0)):
        if len(points_2d) < 8: return image
        for i in range(4):
            cv2.line(image, points_2d[i], points_2d[(i+1)%4], color, 2)
            cv2.line(image, points_2d[i+4], points_2d[((i+1)%4)+4], color, 2)
            cv2.line(image, points_2d[i], points_2d[i+4], color, 2)
        return image

def main():
    actor_list = []
    
    # 1. Load YOLO
    print("Loading YOLO...")
    model = YOLO(os.path.join(PROJECT_ROOT, 'models', 'best.pt'))

    # 2. Video recording setup
    video_filename = 'calibration_result_meters_1.5.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_filename, fourcc, 20.0, (1280, 720))

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    try:
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Spawn camera
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1280')
        cam_bp.set_attribute('image_size_y', '720')
        cam_tr = carla.Transform(carla.Location(x=-11.0, y=0.0, z=11.0), carla.Rotation(pitch=-52.0))
        camera = world.spawn_actor(cam_bp, cam_tr)
        actor_list.append(camera)

        bp_c3 = world.get_blueprint_library().find('vehicle.citroen.c3')
        vehicle = world.spawn_actor(bp_c3, carla.Transform(carla.Location(z=2)))
        actor_list.append(vehicle)

        calib = VisualCalibrator(camera)
        image_data = {'frame': None}
        camera.listen(lambda img: image_data.update({'frame': np.reshape(np.frombuffer(img.raw_data, dtype="uint8"), (720, 1280, 4))[:, :, :3]}))

        test_points = [(0, 0, 0), (2.5, -3.75, 180), (2.5, -1.25, 180), (2.5, 1.25, 180), 
                       (2.5, 3.75, 180), (-2.5, -3.75, 0), (-2.5, -1.25, 0), (-2.5, 1.25, 0), (-2.5, 3.75, 0)]

        # --- LOOP OVER POSITIONS ---
        for idx, (xc, yc, yawc) in enumerate(test_points):
            print(f"Moving to position {idx}: ({xc}, {yc})")
            vehicle.set_transform(carla.Transform(carla.Location(x=xc, y=yc, z=0.1), carla.Rotation(yaw=yawc)))
            
            # --- RECORD 30 FRAMES (1.5 seconds at 20 FPS) ---
            for _ in range(30):
                world.tick() # Advance the simulation
                
                if image_data['frame'] is not None:
                    frame = image_data['frame'].copy()
                    
                    # YOLO inference
                    results = model(frame, conf=0.5, verbose=False)
                    
                    for r in results:
                        for box in r.boxes:
                            b = box.xyxy[0].cpu().numpy()
                            px_c = (b[0] + b[2]) / 2
                            py_c = b[3] # Bottom of the car

                            # Meters via homography
                            coords_m = transformar_a_metros(px_c, py_c, H)
                            xm, ym = coords_m[0], coords_m[1]

                            # Visualization
                            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
                            texto = f"YOLO -> X: {xm:.2f}m, Y: {ym:.2f}m"
                            cv2.putText(frame, texto, (int(b[0]), int(b[1]) - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Show the error against CARLA ground truth
                            error = np.sqrt((xm-xc)**2 + (ym-yc)**2)
                            cv2.putText(frame, f"Error: {error:.3f}m", (50, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # 3D box
                    pts_3d = calib.get_cube_points(vehicle)
                    frame = calib.draw_3d_cube(frame, pts_3d)

                    out_video.write(frame)
                    cv2.imshow("Real-time calibration", frame)
                    cv2.waitKey(1)

    finally:
        out_video.release()
        cv2.destroyAllWindows()
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        for actor in actor_list:
            if actor.is_alive: actor.destroy()
        print(f"Video saved: {video_filename}")

if __name__ == '__main__':
    main()