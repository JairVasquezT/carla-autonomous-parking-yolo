import carla
import numpy as np
import cv2
import os
import queue
import time

# --- CONFIGURATION ---
W, H = 1280, 720
FOV = 90.0
SAVE_PATH = "dataset_parking_final"
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
            
            # Corrected axis transformation for CARLA
            p_cv = np.array([p_cam[1], -p_cam[2], p_cam[0]])

            if p_cv[2] > 0.1: # Point is in front of the camera
                p_2d = np.dot(self.K, p_cv)
                p_2d /= p_2d[2]
                
                # CLIP: if the point falls outside 1280x720, clamp it to the border
                # Instead of discarding the image, clip the box to the border
                x_pix = np.clip(p_2d[0], 0, W)
                y_pix = np.clip(p_2d[1], 0, H)
                coords_2d.append([x_pix, y_pix])

        if len(coords_2d) < 4: return None # At least half of the car must be visible

        x_pts = [p[0] for p in coords_2d]
        y_pts = [p[1] for p in coords_2d]
        
        x_min, x_max = min(x_pts), max(x_pts)
        y_min, y_max = min(y_pts), max(y_pts)

        # TOLERANT FILTER:
        # Discard only if the bounding box is tiny (car very far away)
        if (x_max - x_min) < 10 or (y_max - y_min) < 10:
            return None

        return [x_min, y_min, x_max, y_max]

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
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    
    try:
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # 1. BEV camera
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(W))
        cam_bp.set_attribute('image_size_y', str(H))
        cam_bp.set_attribute('fov', str(FOV))
        cam_tr = carla.Transform(carla.Location(x=-11.0, y=0.0, z=11.0), carla.Rotation(pitch=-52.0))
        camera = world.spawn_actor(cam_bp, cam_tr)
        actor_list.append(camera)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        # 2. Validation test
        print("\n--- STARTING VALIDATION TEST ---")
        bp_c3 = world.get_blueprint_library().find('vehicle.citroen.c3')
        test_car = world.spawn_actor(bp_c3, carla.Transform(carla.Location(x=0, y=0, z=0.1)))
        
        for _ in range(40): world.tick(); image_queue.get()
        
        mgr = DatasetManager(camera, test_car)
        bbox = mgr.get_bbox_2d()
        if bbox:
            print(f"OK! X: {bbox[0]:.1f}-{bbox[2]:.1f} | Y: {bbox[1]:.1f}-{bbox[3]:.1f}")
        
        test_car.destroy()
        print("--- STARTING DATASET GENERATION ---\n")

        # 3. Maneuver loop
        slots = {str(i+1): carla.Location(x=2.5 if i<4 else -2.5, y=[-3.75, -1.25, 1.25, 3.75][i%4], z=0.1) for i in range(8)}
        
        for lado in ["derecha", "izquierda"]:
            for s_id, s_loc in slots.items():
                is_arriba = int(s_id) <= 4
                
                ref_x = s_loc.x + 4.9 if is_arriba else s_loc.x - 4.9
                ref_y = s_loc.y - 3.2 if lado == "izquierda" else s_loc.y + 3.2
                
                spawn_x = ref_x
                spawn_y = ref_y + 8.0 if lado == "izquierda" else ref_y - 8.0
                syaw = -90 if lado == "izquierda" else 90
                
                vehicle = world.spawn_actor(bp_c3, carla.Transform(carla.Location(x=spawn_x, y=spawn_y, z=0.1), carla.Rotation(yaw=syaw)))
                actor_list.append(vehicle)
                
                for _ in range(40):
                    world.tick()
                    image_queue.get() 

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
                print(f"Generating: slot {s_id} - {lado}")

                # --- CONTROL LOOP WITH SAFETY CHECKS ---
                fase = "APROXIMACION" # APROXIMACION (approach), GIRO (steer), RETROCESO (reverse)
                maniobra_activa = True

                while maniobra_activa and vehicle.is_alive:
                    world.tick()
                    data = image_queue.get()
                    step_counter += 1
                    
                    loc_actual = vehicle.get_location()
                    yaw_actual = vehicle.get_transform().rotation.yaw

                    if fase == "APROXIMACION":
                        vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))
                        llegada = (lado == "izquierda" and loc_actual.y <= ref_y+0.17) or \
                                  (lado == "derecha" and loc_actual.y >= ref_y-0.17)
                        if llegada:
                            fase = "GIRO"
                            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                            for _ in range(10): world.tick(); image_queue.get()

                    elif fase == "GIRO":
                        vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=steer_val, reverse=True))
                        if condicion_giro(yaw_actual):
                            fase = "RETROCESO"
                            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                            for _ in range(15): world.tick(); image_queue.get()

                    elif fase == "RETROCESO":
                        vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=0.0, reverse=True))
                        stop = (loc_actual.x <= 2.52) if is_arriba else (loc_actual.x >= -2.52)
                        if stop:
                            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                            # Final captures
                            for _ in range(10): 
                                world.tick()
                                data = image_queue.get()
                                box = mgr.get_bbox_2d()
                                if box: mgr.save_data(data, box)
                            maniobra_activa = False

                    # Save every 5 steps
                    if step_counter % 5 == 0 and maniobra_activa:
                        box = mgr.get_bbox_2d()
                        if box: mgr.save_data(data, box)

                # --- FINAL ACTOR CLEANUP ---
                if vehicle.is_alive:
                    vehicle.destroy()
                if vehicle in actor_list:
                    actor_list.remove(vehicle)
                
                # Empty the image queue
                while not image_queue.empty():
                    image_queue.get()
                print(f"  > Done.")

    finally:
        # Destroy everything on exit
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        print("Dataset finished successfully.")

if __name__ == '__main__':
    main()