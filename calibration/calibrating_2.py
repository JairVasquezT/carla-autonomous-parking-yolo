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

class VisualCalibrator:
    def __init__(self, camera_actor, width=1280, height=720, fov=90.0):
        self.camera = camera_actor
        self.W = width
        self.H = height
        # Intrinsic matrix K for the CARLA camera
        f = width / (2.0 * np.tan(fov * np.pi / 360.0))
        self.K = np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1]])

    def get_cube_points(self, vehicle):
        world_2_cam = np.array(self.camera.get_transform().get_inverse_matrix())
        bb = vehicle.bounding_box
        verts = bb.get_world_vertices(vehicle.get_transform())
        points_2d = []
        for v in verts:
            p_world = np.array([v.x, v.y, v.z, 1])
            p_cam = np.dot(world_2_cam, p_world)
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
    
    def get_stable_base_center(self, vehicle):
        """
        Compute the exact center of the box base (ground) in pixels
        by averaging the 4 bottom vertices of the 3D box.
        """
        # 1. Get the camera matrix
        world_2_cam = np.array(self.camera.get_transform().get_inverse_matrix())
        
        # 2. Get the 8 box vertices in the world
        bb = vehicle.bounding_box
        verts = bb.get_world_vertices(vehicle.get_transform())
        
        # In CARLA's bounding_box layout:
        # The first 4 points (0,1,2,3) are usually the base (lowest Z)
        # or they can be filtered by height.
        points_2d = []
        for v in verts:
            p_cam = np.dot(world_2_cam, np.array([v.x, v.y, v.z, 1]))
            p_cv = np.array([p_cam[1], -p_cam[2], p_cam[0]])
            if p_cv[2] > 0.1:
                p_2d = np.dot(self.K, p_cv)
                p_2d /= p_2d[2]
                points_2d.append([p_2d[0], p_2d[1], v.z]) # Keep pixel X, Y and world Z

        if len(points_2d) < 8:
            return None

        # 3. Find the 4 points with the lowest Z (the ground)
        # Sort them by real world Z
        points_2d.sort(key=lambda x: x[2])
        base_points = np.array(points_2d[:4]) # The 4 points closest to the ground

        # 4. Average those 4 points in PIXELS
        cx_px = np.mean(base_points[:, 0])
        cy_px = np.mean(base_points[:, 1])

        return int(cx_px), int(cy_px)

# --- 3. MAIN FUNCTION ---
def main():
    actor_list = []
    results_table = []
    
    # Load the trained model
    print("Loading YOLO model...")
    model = YOLO(os.path.join(PROJECT_ROOT, 'models', 'best.pt')) 

    # Test points (x, y, yaw)
    test_points = [
    # --- CENTER ---
        (0, 0, 0),

        # --- INNER RING (±2.5) ---
        (2.5, -3.75, 0), (2.5, -1.25, 0), (2.5, 1.25, 0), (2.5, 3.75, 0),
        (-2.5, -3.75, 180), (-2.5, -1.25, 180), (-2.5, 1.25, 180), (-2.5, 3.75, 180),

        # --- MIDDLE RING (±5.0) ---
        (5.0, -6.0, 0), (5.0, -3.0, 0), (5.0, 0.0, 0), (5.0, 3.0, 0), (5.0, 6.0, 0),
        (-5.0, -6.0, 180), (-5.0, -3.0, 180), (-5.0, 0.0, 180), (-5.0, 3.0, 180), (-5.0, 6.0, 180),

        # --- OUTER RING (±7.4) ---
        (7.4, -6.95, 0), (7.4, -4.45, 0), (7.4, -1.95, 0), (7.4, -0.55, 0),
        (7.4, 0.55, 0), (7.4, 1.95, 0), (7.4, 4.45, 0), (7.4, 6.95, 0),

        (-7.4, -6.95, 180), (-7.4, -4.45, 180), (-7.4, -1.95, 180), (-7.4, -0.55, 180),
        (-7.4, 0.55, 180), (-7.4, 1.95, 180), (-7.4, 4.45, 180), (-7.4, 6.95, 180),
    ]

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    try:
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Spawn camera (original configuration)
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1280')
        cam_bp.set_attribute('image_size_y', '720')
        cam_transform = carla.Transform(
            carla.Location(x=-11.0, y=0.0, z=11.0), 
            carla.Rotation(pitch=-52.0, yaw=0.0, roll=0.0)
        )
        camera = world.spawn_actor(cam_bp, cam_transform)
        actor_list.append(camera)

        # Visual calibration aid
        calib = VisualCalibrator(camera)

        # Image buffer
        image_data = {'frame': None}
        camera.listen(lambda image: image_data.update({'frame': np.reshape(np.frombuffer(image.raw_data, dtype="uint8"), (720, 1280, 4))[:, :, :3]}))

        # Spawn Citroën C3
        bp_c3 = world.get_blueprint_library().find('vehicle.citroen.c3')
        vehicle = world.spawn_actor(bp_c3, carla.Transform(carla.Location(z=2)))
        actor_list.append(vehicle)

        print(f"\n{'Pos':<5} | {'X_Real':<7} | {'Y_Real':<7} | {'X_YOLO_PX':<10} | {'Y_YOLO_PX':<10}")
        print("-" * 65)

        # --- INSIDE THE test_points LOOP ---
        for i, (xc, yc, yawc) in enumerate(test_points):
            vehicle.set_transform(carla.Transform(carla.Location(x=xc, y=yc, z=0.2), carla.Rotation(yaw=yawc)))
            for _ in range(30): world.tick()

            # GET A STABLE CENTER (BOX BASE)
            centro_estable = calib.get_stable_base_center(vehicle)
            
            if centro_estable is not None:
                cx_px, cy_px = centro_estable
                
                # Draw for visual feedback
                frame = image_data['frame'].copy()
                cv2.circle(frame, (cx_px, cy_px), 8, (0, 0, 255), -1) # Red dot at the real center
                
                # Also draw the box to show the orientation
                pts_3d = calib.get_cube_points(vehicle)
                frame = calib.draw_3d_cube(frame, pts_3d)
                
                cv2.imshow("Calibrating stable center", frame)
                cv2.waitKey(100)

                # SAVE TO SPREADSHEET
                # cx_px and cy_px are the real ground-contact center of the car
                results_table.append((i, xc, yc, cx_px, cy_px))

    finally:
        cv2.destroyAllWindows()
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        for actor in actor_list:
            if actor.is_alive: actor.destroy()
        print("\n" + "="*50)
        print("COPY THESE VALUES FOR THE HOMOGRAPHY COMPUTATION")
        print("="*50)
        print("X_Px,Y_Px,X_Carla,Y_Carla") # CSV header

        for r in results_table:
            # r[3] is cx_px, r[4] is cy_px, r[1] is xc, r[2] is yc
            print(f"{r[3]},{r[4]},{r[1]},{r[2]}") 
        print("="*50)
        
        print("\nTest finished.")

if __name__ == '__main__':
    main()