import carla
import time

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    try:
        # 1. Spawn the test car
        bp = world.get_blueprint_library().find('vehicle.citroen.c3')
        spawn_point = carla.Transform(carla.Location(x=7.4, y=-2.0, z=0.1), carla.Rotation(yaw=-90))
        vehicle = world.try_spawn_actor(bp, spawn_point)
        
        print(f"Car spawned at: {spawn_point.location}")

        
        print("reverse and steer...")
        vehicle.apply_control(carla.VehicleControl(throttle=0.45, steer=-1.0, reverse=True))
        time.sleep(3.2)


        control_reversa = carla.VehicleControl(throttle=0.45, steer=0.0, reverse=True)
        vehicle.apply_control(control_reversa)
        time.sleep(3)

        # Maneuver 6: hand brake (drift/lock)
        print("Hand brake on!")
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, hand_brake=True))
        time.sleep(2)

    finally:
        print("Cleaning up...")
        vehicle.destroy()

if __name__ == '__main__':
    main()