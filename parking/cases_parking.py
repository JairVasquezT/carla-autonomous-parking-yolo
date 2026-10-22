import carla
import numpy as np
import cv2
import queue

# --- VIDEO SETUP ---
W, H = 1280, 720
FPS = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('parking_maneuver.mp4', fourcc, FPS, (W, H))

def main():
    actor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    
    try:
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 1.0 / FPS # Matches the video frame rate
        world.apply_settings(settings)

        # --- BEV CAMERA SETUP ---
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(W))
        cam_bp.set_attribute('image_size_y', str(H))
        cam_tr = carla.Transform(carla.Location(x=-11.0, y=0.0, z=11.0), carla.Rotation(pitch=-52.0))
        camera = world.spawn_actor(cam_bp, cam_tr)
        actor_list.append(camera)

        image_queue = queue.Queue()
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
                spawn_x = slot_loc.x + 4.9 if is_arriba else slot_loc.x - 4.9
                spawn_y = slot_loc.y - 3.2 if lado == "izquierda" else slot_loc.y + 3.2
                spawn_yaw = -90 if lado == "izquierda" else 90
                
                vehicle = world.try_spawn_actor(bp_c3, carla.Transform(carla.Location(x=spawn_x, y=spawn_y, z=0.1), carla.Rotation(yaw=spawn_yaw)))
                if not vehicle: continue
                actor_list.append(vehicle)
                
                # Let the car settle while recording idle frames
                for _ in range(30):
                    world.tick()
                    img = image_queue.get()
                    # Convert image for the video
                    frame = np.reshape(np.frombuffer(img.raw_data, dtype=np.uint8), (H, W, 4))[:, :, :3]
                    video_out.write(frame)

                # Steering logic
                if is_arriba and lado == "derecha":
                    steer_val, condicion_giro = 1.0, lambda y: y < 1.0
                elif not is_arriba and lado == "derecha":
                    steer_val, condicion_giro = -1.0, lambda y: y < 0 and y < 179.0
                elif is_arriba and lado == "izquierda":
                    steer_val, condicion_giro = -1.0, lambda y: y > -1.0
                else:
                    steer_val, condicion_giro = 1.0, lambda y: y > 0 and y > -179.0

                # --- 2. STEERING PHASE ---
                while True:
                    world.tick()
                    img = image_queue.get()
                    frame = np.reshape(np.frombuffer(img.raw_data, dtype=np.uint8), (H, W, 4))[:, :, :3]
                    video_out.write(frame)
                    
                    yaw_actual = vehicle.get_transform().rotation.yaw
                    vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=steer_val, reverse=True))

                    if condicion_giro(yaw_actual):
                        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                        for _ in range(10): 
                            world.tick()
                            video_out.write(np.reshape(np.frombuffer(image_queue.get().raw_data, dtype=np.uint8), (H, W, 4))[:, :, :3])
                        break

                # --- 3. FINAL STRAIGHT REVERSE ---
                while True:
                    world.tick()
                    img = image_queue.get()
                    frame = np.reshape(np.frombuffer(img.raw_data, dtype=np.uint8), (H, W, 4))[:, :, :3]
                    video_out.write(frame)
                    
                    loc = vehicle.get_location()
                    vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=0.0, reverse=True))

                    stop = (loc.x <= 2.52) if is_arriba else (loc.x >= -2.52)
                    if stop:
                        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                        for _ in range(20): # Final pause once parked
                            world.tick()
                            video_out.write(np.reshape(np.frombuffer(image_queue.get().raw_data, dtype=np.uint8), (H, W, 4))[:, :, :3])
                        break

                vehicle.destroy()
                if vehicle in actor_list: actor_list.remove(vehicle)
                print(f"Maneuver {slot_id} {lado} recorded.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        video_out.release() # Required to finalize the file
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for a in actor_list:
            if a.is_alive: a.destroy()
        print("Video saved as 'parking_maneuver.mp4'")

if __name__ == '__main__':
    main()