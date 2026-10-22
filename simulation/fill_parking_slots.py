import carla

def main():
    actor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    try:
        world = client.get_world()
        
        # --- POSITION SETUP ---
        # Parking slot coordinates
        parking_slots = {
            "1": carla.Transform(carla.Location(x=2.5, y=-3.75, z=0.5), carla.Rotation(yaw=0)),
            "2": carla.Transform(carla.Location(x=2.5, y=-1.25, z=0.5), carla.Rotation(yaw=0)),
            "3": carla.Transform(carla.Location(x=2.5, y=1.25, z=0.5), carla.Rotation(yaw=0)),
            "4": carla.Transform(carla.Location(x=2.5, y=3.75, z=0.5), carla.Rotation(yaw=0)),
            "5": carla.Transform(carla.Location(x=-2.5, y=-3.75, z=0.5), carla.Rotation(yaw=180)), # Facing the opposite way
            "6": carla.Transform(carla.Location(x=-2.5, y=-1.25, z=0.5), carla.Rotation(yaw=180)),
            "7": carla.Transform(carla.Location(x=-2.5, y=1.25, z=0.5), carla.Rotation(yaw=180)),
            "8": carla.Transform(carla.Location(x=-2.5, y=3.75, z=0.5), carla.Rotation(yaw=180)),
        }

        # --- USER INPUT ---
        print("\n--- PARKING SETUP ---")
        print("Available slots: 1, 2, 3, 4 (right) | 5, 6, 7, 8 (left)")
        entrada = input("Enter the OCCUPIED slot numbers (comma separated): ")
        
        # Parse the input (e.g. "1,2,5" -> ["1", "2", "5"])
        ocupadas = [x.strip() for x in entrada.split(',')]

        # --- SPAWN ---
        blueprint_library = world.get_blueprint_library()
        bp_prius = blueprint_library.find('vehicle.toyota.prius')

        for id_pos in ocupadas:
            if id_pos in parking_slots:
                transform = parking_slots[id_pos]
                vehicle = world.try_spawn_actor(bp_prius, transform)
                if vehicle:
                    actor_list.append(vehicle)
                    print(f"Slot {id_pos} OCCUPIED by a Prius.")
                else:
                    print(f"Error: could not fill slot {id_pos} (possible collision).")
            else:
                print(f"Warning: '{id_pos}' is not a valid slot.")

        # Keep the script alive to inspect the result
        print("\nVehicles spawned. Press Ctrl+C to clean up and exit.")
        while True:
            world.wait_for_tick()

    except KeyboardInterrupt:
        print("\nStopping simulation...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print(f"Destroying {len(actor_list)} actors...")
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        print("Done.")

if __name__ == '__main__':
    main()