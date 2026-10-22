import carla
import math
import time

def main():
    actor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0) 
    
    try:
        world = client.get_world()
        
        # --- 1. SYNCHRONOUS MODE SETUP ---
        settings = world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 0.05 # Fixed 20 FPS for physics and camera
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # --- 2. SPAWN FIXED CAMERA ---
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_transform = carla.Transform(
            carla.Location(x=-9.0, y=0.0, z=4.0), 
            carla.Rotation(pitch=-33.0, yaw=0.0, roll=0.0)
        )
        camera = world.spawn_actor(cam_bp, cam_transform)
        actor_list.append(camera)
        
        # Automatic recording
        camera.listen(lambda image: image.save_to_disk('_out_parking_2/%06d.png' % image.frame))

        # --- 3. SPAWN VEHICLES ---
        # Main vehicle (Citroën C3)
        bp_c3 = blueprint_library.find('vehicle.citroen.c3')
        spawn_point = carla.Transform(carla.Location(x=7.4, y=-2.0, z=0.1), carla.Rotation(yaw=-90))
        vehicle = world.spawn_actor(bp_c3, spawn_point)
        actor_list.append(vehicle)

        # Parked cars (Toyota Prius) used as obstacles
        bp_prius = blueprint_library.find('vehicle.toyota.prius')
        spawn_2 = carla.Transform(carla.Location(x=-2.5, y=3.6, z=0.1), carla.Rotation(yaw=180))
        spawn_3 = carla.Transform(carla.Location(x=2.5, y=-1.25, z=0.1), carla.Rotation(yaw=0))
        
        vehicle_2 = world.spawn_actor(bp_prius, spawn_2)
        vehicle_3 = world.spawn_actor(bp_prius, spawn_3)
        actor_list.append(vehicle_2)
        actor_list.append(vehicle_3)

        # A few ticks so the cars drop to the ground and physics settles
        for _ in range(10):
            world.tick()

        # --- 4. MANEUVER 1: STEER UNTIL YAW = 0 ---
        print("Steering... target yaw = 0")
        while True:
            world.tick()
            tf = vehicle.get_transform()
            yaw = tf.rotation.yaw
            
            # From -90 we need to steer left to reach 0
            # Negative steer is left, positive is right.
            vehicle.apply_control(carla.VehicleControl(throttle=0.45, steer=-1.0, reverse=True))

            # Stop condition: close to 0 degrees
            if abs(yaw) < 3.0:
                print(f"Yaw reached: {yaw:.2f}°. Braking.")
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                for _ in range(20): world.tick() # Full stop
                break

        # --- 5. MANEUVER 2: REVERSE UNTIL X = 2.5 ---
        print("Reversing until X = 2.5")
        while True:
            world.tick()
            loc = vehicle.get_location()
            
            # Straight reverse
            vehicle.apply_control(carla.VehicleControl(throttle=0.45, steer=0.0, reverse=True))

            # Stop condition
            if loc.x <= 2.45:
                print(f"X position reached: {loc.x:.2f}. Parked.")
                # Hand brake to finish
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                for _ in range(20): world.tick()
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # --- 6. CLEANUP ---
        print("Disabling synchronous mode...")
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        
        print('Destroying actors...')
        for actor in actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        print('Done.')

if __name__ == '__main__':
    main()