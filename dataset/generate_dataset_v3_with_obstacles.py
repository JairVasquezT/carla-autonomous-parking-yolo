import carla
import numpy as np
import cv2
import os
import queue
import time
import random

# --- CONFIGURATION ---
W, H = 1280, 720
FOV = 90.0
SAVE_PATH = "dataset_parking_c3"
os.makedirs(f"{SAVE_PATH}/images", exist_ok=True)
os.makedirs(f"{SAVE_PATH}/labels", exist_ok=True)

class DatasetManager:
    def __init__(self, camera, vehicle):
        self.camera = camera
        self.vehicle = vehicle
        f = W / (2.0 * np.tan(FOV * np.pi / 360.0))
        self.K = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]])

    def get_bbox_2d(self):
        if not self.vehicle.is_alive: return None
        world_2_cam = np.array(self.camera.get_transform().get_inverse_matrix())
        verts = self.vehicle.bounding_box.get_world_vertices(self.vehicle.get_transform())
        
        coords_2d = []
        for v in verts:
            p_world = np.array([v.x, v.y, v.z, 1])
            p_cam = np.dot(world_2_cam, p_world)
            p_cv = np.array([p_cam[1], -p_cam[2], p_cam[0]])

            if p_cv[2] > 0.1:
                p_2d = np.dot(self.K, p_cv)
                p_2d /= p_2d[2]
                x_pix = np.clip(p_2d[0], 0, W)
                y_pix = np.clip(p_2d[1], 0, H)
                coords_2d.append([x_pix, y_pix])

        if len(coords_2d) < 4: return None 

        x_pts = [p[0] for p in coords_2d]
        y_pts = [p[1] for p in coords_2d]
        return [min(x_pts), min(y_pts), max(x_pts), max(y_pts)]

    def save_data(self, image_data, bbox):
        x_min, y_min, x_max, y_max = bbox
        x_center = ((x_min + x_max) / 2.0) / W
        y_center = ((y_min + y_max) / 2.0) / H
        w_norm = (x_max - x_min) / W
        h_norm = (y_max - y_min) / H

        img = np.reshape(np.frombuffer(image_data.raw_data, dtype=np.uint8), (H, W, 4))[:, :, :3]
        ts = int(time.time() * 1000000)
        cv2.imwrite(f"{SAVE_PATH}/images/{ts}.jpg", img)
        with open(f"{SAVE_PATH}/labels/{ts}.txt", "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

def main():
    actor_list = []
    ambient_list = [] # Separate list for the parked Prius cars
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    
    try:
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        bp_library = world.get_blueprint_library()
        bp_c3 = bp_library.find('vehicle.citroen.c3')
        bp_prius = bp_library.find('vehicle.toyota.prius')

        # 1. BEV camera
        cam_bp = bp_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(W))
        cam_bp.set_attribute('image_size_y', str(H))
        cam_bp.set_attribute('fov', str(FOV))
        cam_tr = carla.Transform(carla.Location(x=-11.0, y=0.0, z=11.0), carla.Rotation(pitch=-52.0))
        camera = world.spawn_actor(cam_bp, cam_tr)
        actor_list.append(camera)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # 2. Slot definitions
        parking_slots = {str(i+1): carla.Location(x=2.5 if i<4 else -2.5, y=[-3.75, -1.25, 1.25, 3.75][i%4], z=0.1) for i in range(8)}

        print("\n--- STARTING GENERATION WITH PARKED CARS ---\n")

        for lado in ["derecha", "izquierda"]:
            for s_id, s_loc in parking_slots.items():
                
                # --- CLEAN UP PREVIOUS ACTORS ---
                for a in ambient_list:
                    if a.is_alive: a.destroy()
                ambient_list = []

                # --- SPAWN PARKED CARS (PRIUS) ---
                # Fill every slot except the current one (s_id)
                for other_id, other_loc in parking_slots.items():
                    if other_id != s_id:
                        # Rotation: 180 for the top row (x=2.5), 0 for the bottom row (x=-2.5)
                        rot = carla.Rotation(yaw=180 if other_loc.x > 0 else 0)
                        p_car = world.try_spawn_actor(bp_prius, carla.Transform(other_loc, rot))
                        if p_car:
                            ambient_list.append(p_car)

                # --- C3 MANEUVER ---
                is_arriba = int(s_id) <= 4
                sx = s_loc.x + 4.9 if is_arriba else s_loc.x - 4.9
                sy = s_loc.y - 3.2 if lado == "izquierda" else s_loc.y + 3.2
                syaw = -90 if lado == "izquierda" else 90
                
                vehicle = world.spawn_actor(bp_c3, carla.Transform(carla.Location(x=sx, y=sy, z=0.1), carla.Rotation(yaw=syaw)))
                actor_list.append(vehicle)
                
                for _ in range(40): world.tick(); image_queue.get() 

                if is_arriba and lado == "derecha":
                    steer_val, condicion_giro = 1.0, lambda y: y < 1.0
                elif not is_arriba and lado == "derecha":
                    steer_val, condicion_giro = -1.0, lambda y: y < 0 and y < 179.0
                elif is_arriba and lado == "izquierda":
                    steer_val, condicion_giro = -1.0, lambda y: y > -1.0
                else:
                    steer_val, condicion_giro = 1.0, lambda y: y > 0 and y > -179.0

                mgr = DatasetManager(camera, vehicle)
                step_counter = 0
                print(f"Generating: slot {s_id} - {lado} (with parked Prius cars)")

                # --- STEERING PHASE ---
                while True:
                    world.tick(); data = image_queue.get(); step_counter += 1
                    yaw_actual = vehicle.get_transform().rotation.yaw
                    vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=steer_val, reverse=True))
                    
                    if step_counter % 5 == 0:
                        box = mgr.get_bbox_2d()
                        if box: mgr.save_data(data, box)

                    if condicion_giro(yaw_actual):
                        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                        for _ in range(15): world.tick(); image_queue.get()
                        break

                # --- FINAL STRAIGHT REVERSE ---
                while True:
                    world.tick(); data = image_queue.get(); step_counter += 1
                    loc = vehicle.get_location()
                    vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=0.0, reverse=True))

                    if step_counter % 5 == 0:
                        box = mgr.get_bbox_2d()
                        if box: mgr.save_data(data, box)

                    stop = (loc.x <= 2.52) if is_arriba else (loc.x >= -2.52)
                    if stop:
                        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                        for _ in range(10): 
                            world.tick(); data = image_queue.get()
                            box = mgr.get_bbox_2d(); 
                            if box: mgr.save_data(data, box)
                        break

                vehicle.destroy()
                if vehicle in actor_list: actor_list.remove(vehicle)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for a in actor_list + ambient_list:
            if a.is_alive: a.destroy()
        print("Dataset finished.")

if __name__ == '__main__':
    main()