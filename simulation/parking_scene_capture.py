import carla
import time

def main():
    actor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0) 
    
    try:
        world = client.get_world()
        
        # --- SYNCHRONOUS MODE ---
        settings = world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 0.05 
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # 1. Fixed camera
        cam_transform = carla.Transform(
            carla.Location(x=-9.0, y=0.0, z=4.0), 
            carla.Rotation(pitch=-33.0, yaw=0.0, roll=0.0)
        )
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        camera = world.spawn_actor(cam_bp, cam_transform)
        actor_list.append(camera)

        
        vehicle_bp = blueprint_library.find('vehicle.citroen.c3')
        vehicle_bp_2 = blueprint_library.find('vehicle.toyota.prius')
        vehicle_bp_3 = blueprint_library.find('vehicle.toyota.prius')

        spawn_point = carla.Transform(carla.Location(x=5.0, y=-10.0, z=0.1), carla.Rotation(yaw=90))
        spawn_point_2 = carla.Transform(carla.Location(x=-2.5, y=3.6, z=0.1), carla.Rotation(yaw=0))
        spawn_point_3 = carla.Transform(carla.Location(x=2.5, y=3.6, z=0.1), carla.Rotation(yaw=180))

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        vehicle_2 = world.spawn_actor(vehicle_bp_2, spawn_point_2)
        vehicle_3 = world.spawn_actor(vehicle_bp_3, spawn_point_3)
        # Keep both in the list for cleanup
        actor_list.append(vehicle)
        actor_list.append(vehicle_2)

        # 3. Recording
        camera.listen(lambda image: image.save_to_disk('_out_fijo_2/%06d.png' % image.frame))

        # --- "WAKE UP" THE WORLD ---
        for _ in range(10):
            world.tick()

        # 4. Start moving the main vehicle
        print("Accelerating...")
        vehicle.apply_control(carla.VehicleControl(throttle=0.5))

        contador_frames = 0
        while True:
            world.tick() 
            contador_frames += 1
            
            loc = vehicle.get_location()
            
            if contador_frames % 10 == 0:
                print(f"Frame: {contador_frames} | Y position: {loc.y:.2f}")

            # Exit condition
            if loc.y >= 10.0:
                print("Target reached.")
                break
            
            if contador_frames > 1000:
                print("Timeout.")
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # BACK TO ASYNCHRONOUS MODE
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        
        print('Cleaning up and exiting...')
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        print('Done.')

if __name__ == '__main__':
    main()