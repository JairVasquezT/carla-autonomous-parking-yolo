import carla
import time

def main():
    # --- CALIBRATION SETTINGS ---
    MODO_SINCRO = True   # False for fast calibration, True for smooth recording
    GRABAR_FOTOS = True  # False to avoid filling the disk while testing
    # ------------------------------

    actor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    try:
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = MODO_SINCRO
        if MODO_SINCRO:
            settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        
        # Camera
        cam_transform = carla.Transform(carla.Location(x=-9.0, y=0.0, z=4.0), carla.Rotation(pitch=-33.0))
        camera = world.spawn_actor(bp_lib.find('sensor.camera.rgb'), cam_transform)
        actor_list.append(camera)

        if GRABAR_FOTOS:
            camera.listen(lambda image: image.save_to_disk('_out_contorno/%06d.png' % image.frame))

        # Car (start: 7, -15)
        spawn_point = carla.Transform(carla.Location(x=7.0, y=-15.0, z=0.5), carla.Rotation(yaw=90))
        vehicle = world.spawn_actor(bp_lib.find('vehicle.tesla.model3'), spawn_point)
        actor_list.append(vehicle)

        stage = 1
        print("Starting trajectory to the RIGHT...")

        while True:
            if MODO_SINCRO: world.tick()
            else: world.wait_for_tick()

            loc = vehicle.get_location()
            rot = vehicle.get_transform().rotation.yaw
            control = carla.VehicleControl(throttle=0.3, steer=0.0)

            # STAGE 1: towards corner (7, 7)
            if stage == 1:
                if loc.y > 3.0: # Start steering 4 m before 7
                    control.steer = 0.6 # POSITIVE = RIGHT
                
                # If the car already points right (yaw about 180 or -180)
                if rot > 170 or rot < -170:
                    stage = 2
                    print("Stage 2: towards X=-7")

            # STAGE 2: towards corner (-7, 7)
            elif stage == 2:
                if loc.x < -3.0: # Before reaching -7
                    control.steer = 0.6 # RIGHT
                
                if -100 < rot < -80: # Already pointing down (negative Y)
                    stage = 3
                    print("Stage 3: towards the end point")

            # STAGE 3: finish
            elif stage == 3:
                if loc.y < -7.0:
                    print("Goal reached.")
                    break

            vehicle.apply_control(control)

    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for actor in actor_list: actor.destroy()
        print("Cleanup complete.")

if __name__ == '__main__':
    main()