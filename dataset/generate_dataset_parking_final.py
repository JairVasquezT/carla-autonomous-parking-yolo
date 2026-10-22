import carla
import numpy as np
import cv2
import os
import queue

# --- DATASET SETUP ---
DATASET_NAME = "dataset_parking_final"
SAVE_PATH = os.path.join(os.getcwd(), DATASET_NAME)
os.makedirs(os.path.join(SAVE_PATH, "images"), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, "labels"), exist_ok=True)

class DatasetGenerator:
    def __init__(self, camera_actor, vehicle_actor):
        self.camera = camera_actor
        self.vehicle = vehicle_actor
        self.img_width = 1280
        self.img_height = 720
        self.fov = 90.0

    def get_camera_intrinsic_matrix(self):
        f = self.img_width / (2.0 * np.tan(self.fov * np.pi / 360.0))
        K = np.array([[f, 0, self.img_width / 2.0],
                     [0, f, self.img_height / 2.0],
                     [0, 0, 1]])
        return K

    def process_and_save(self, carla_image):
        """Convert the CARLA image and save image + YOLO label"""
        # 1. Convert image to OpenCV format
        array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (carla_image.height, carla_image.width, 4))
        frame = array[:, :, :3] # Drop the alpha channel

        # 2. Project the 3D bbox to 2D
        world_2_camera = np.array(self.camera.get_transform().get_inverse_matrix())
        K = self.get_camera_intrinsic_matrix()
        bb = self.vehicle.bounding_box
        verts = bb.get_world_vertices(self.vehicle.get_transform())
        
        x_max, y_max, x_min, y_min = -1, -1, 10000, 10000
        for v in verts:
            p = np.array([v.x, v.y, v.z, 1])
            p_camera = np.dot(world_2_camera, p)
            if p_camera[2] > 0:
                p_2d = np.dot(K, p_camera[:3])
                p_2d /= p_2d[2]
                x_min, x_max = min(x_min, p_2d[0]), max(x_max, p_2d[0])
                y_min, y_max = min(y_min, p_2d[1]), max(y_max, p_2d[1])

        # 3. Check visibility and save
        if x_min < self.img_width and x_max > 0 and y_min < self.img_height and y_max > 0:
            x_min, x_max = max(0, x_min), min(self.img_width, x_max)
            y_min, y_max = max(0, y_min), min(self.img_height, y_max)
            
            x_center = ((x_min + x_max) / 2.0) / self.img_width
            y_center = ((y_min + y_max) / 2.0) / self.img_height
            w = (x_max - x_min) / self.img_width
            h = (y_max - y_min) / self.img_height

            fname = f"{carla_image.frame}_{self.vehicle.id}"
            cv2.imwrite(os.path.join(SAVE_PATH, "images", f"{fname}.jpg"), frame)
            with open(os.path.join(SAVE_PATH, "labels", f"{fname}.txt"), "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

def main():
    actor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0) # Longer timeout
    
    try:
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 0.05 
        world.apply_settings(settings)

        # Queue used to synchronize images
        image_queue = queue.Queue()

        # --- SPAWN CAMERA (BEV) ---
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1280')
        cam_bp.set_attribute('image_size_y', '720')
        cam_bp.set_attribute('fov', '90')
        cam_transform = carla.Transform(carla.Location(x=-11.0, y=0.0, z=11.0), carla.Rotation(pitch=-52.0))
        camera = world.spawn_actor(cam_bp, cam_transform)
        actor_list.append(camera)

        # The camera always listens and pushes frames into the queue
        camera.listen(image_queue.put)

        parking_slots = {
            "1": carla.Location(x=2.5, y=-3.75, z=0.1), "2": carla.Location(x=2.5, y=-1.25, z=0.1),
            "3": carla.Location(x=2.5, y=1.25, z=0.1), "4": carla.Location(x=2.5, y=3.75, z=0.1),
            "5": carla.Location(x=-2.5, y=-3.75, z=0.1), "6": carla.Location(x=-2.5, y=-1.25, z=0.1),
            "7": carla.Location(x=-2.5, y=1.25, z=0.1), "8": carla.Location(x=-2.5, y=3.75, z=0.1),
        }

        bp_c3 = world.get_blueprint_library().find('vehicle.citroen.c3')

        for lado in ["derecha", "izquierda"]:
            for slot_id, slot_loc in parking_slots.items():
                is_arriba = int(slot_id) <= 4
                
                # --- POSITION SETUP ---
                # Reference position (where the steering starts)
                ref_x = slot_loc.x + 4.9 if is_arriba else slot_loc.x - 4.9
                ref_y = slot_loc.y - 3.2 if lado == "izquierda" else slot_loc.y + 3.2
                
                # SPAWN position (10 m back depending on the side)
                # Left side: spawn 10 m to the right (+10 in Y) and drive left
                # Right side: spawn 10 m to the left (-10 in Y) and drive right
                spawn_x = ref_x
                spawn_y = ref_y + 10.0 if lado == "izquierda" else ref_y - 10.0
                spawn_yaw = -90 if lado == "izquierda" else 90
                
                vehicle = world.try_spawn_actor(bp_c3, carla.Transform(
                    carla.Location(x=spawn_x, y=spawn_y, z=0.5), 
                    carla.Rotation(yaw=spawn_yaw)))
                
                if not vehicle: continue
                actor_list.append(vehicle)

                gen = DatasetGenerator(camera, vehicle)
                
                # Steering logic
                if is_arriba and lado == "derecha": steer, cond = 1.0, lambda y: y < 1.0
                elif not is_arriba and lado == "derecha": steer, cond = -1.0, lambda y: y < 0 and y < 179.0
                elif is_arriba and lado == "izquierda": steer, cond = -1.0, lambda y: y > -1.0
                else: steer, cond = 1.0, lambda y: y > 0 and y > -179.0

                print(f"Starting approach and maneuver: slot {slot_id} - {lado}")
                
                step_count = 0
                maniobra_activa = True
                fase_aproximacion = True  # Approach phase
                fase_giro = False

                while maniobra_activa:
                    world.tick()
                    data = image_queue.get() 
                    
                    current_loc = vehicle.get_location()

                    # --- PHASE 1: STRAIGHT FORWARD (APPROACH) ---
                    if fase_aproximacion:
                        vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))
                        
                        # Check whether the reference Y position was reached
                        reached_left = (lado == "izquierda" and current_loc.y <= ref_y)
                        reached_right = (lado == "derecha" and current_loc.y >= ref_y)
                        
                        if reached_left or reached_right:
                            fase_aproximacion = False
                            fase_giro = True
                            # Brief braking to settle the physics before steering
                            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                            world.tick()

                    # --- PHASE 2: STEERING IN REVERSE ---
                    elif fase_giro:
                        vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=steer, reverse=True))
                        if cond(vehicle.get_transform().rotation.yaw):
                            fase_giro = False
                    
                    # --- PHASE 3: FINAL STRAIGHT REVERSE ---
                    else:
                        vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=0.0, reverse=True))
                        if (is_arriba and current_loc.x <= 2.52) or (not is_arriba and current_loc.x >= -2.52):
                            maniobra_activa = False

                    # Capture data continuously during every phase
                    step_count += 1
                    if step_count % 5 == 0:
                        gen.process_and_save(data)

                # --- SHUTDOWN ---
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                for _ in range(10):
                    world.tick()
                    data = image_queue.get()
                    gen.process_and_save(data)

                vehicle.destroy()
                while not image_queue.empty(): image_queue.get()

    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        for actor in actor_list:
            if actor.is_alive: actor.destroy()
        print(f"Dataset finished. See: {SAVE_PATH}")

if __name__ == '__main__':
    main()