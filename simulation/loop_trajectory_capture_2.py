import carla
import time

def main():
    MODO_SINCRO = True   
    GRABAR_FOTOS = True  

    actor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    try:
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = MODO_SINCRO
        if MODO_SINCRO: settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()
        
        # Fixed camera
        cam_transform = carla.Transform(carla.Location(x=-9.0, y=0.0, z=4.0), carla.Rotation(pitch=-33.0))
        camera = world.spawn_actor(bp_lib.find('sensor.camera.rgb'), cam_transform)
        actor_list.append(camera)

        if GRABAR_FOTOS:
            camera.listen(lambda image: image.save_to_disk('_out_contorno_2/%06d.png' % image.frame))

        # Car (start: 7, -15)
        spawn_point = carla.Transform(carla.Location(x=7.0, y=-15.0, z=0.5), carla.Rotation(yaw=90))
        vehicle = world.spawn_actor(bp_lib.find('vehicle.tesla.model3'), spawn_point)
        actor_list.append(vehicle)

        stage = 1
        distancia_final = 0
        print("Starting full trajectory...")

        while True:
            if MODO_SINCRO: world.tick()
            else: world.wait_for_tick()

            loc = vehicle.get_location()
            rot = vehicle.get_transform().rotation.yaw
            control = carla.VehicleControl(throttle=0.3, steer=0.0)

            # STAGE 1: first corner (turn right towards negative X)
            if stage == 1:
                if loc.y > 3.0: control.steer = 0.6
                if rot > 170 or rot < -170: stage = 2

            # STAGE 2: second corner (turn right towards negative Y)
            elif stage == 2:
                if loc.x < -3.0: control.steer = 0.6
                if -100 < rot < -80: stage = 3

            # STAGE 3: third corner (turn right towards positive X)
            elif stage == 3:
                if loc.y < -3.0: control.steer = 0.6
                # When the car faces the X axis again (yaw close to 0)
                if -10 < rot < 10: 
                    stage = 4
                    distancia_final = loc.x + 5.0 # Stop point
                    print(f"Stage 4: final straight. Stop at X = {distancia_final:.1f}")

            # STAGE 4: final 14 m straight and stop
            elif stage == 4:
                control.steer = 0.0
                if loc.x >= distancia_final:
                    print("Route finished. Braking.")
                    control.throttle = 0.0
                    control.brake = 1.0
                    vehicle.apply_control(control)
                    for _ in range(20): world.tick() # Make sure the stop is visible in the video
                    break

            vehicle.apply_control(control)

    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for actor in actor_list: actor.destroy()
        print("Finished.")

if __name__ == '__main__':
    main()