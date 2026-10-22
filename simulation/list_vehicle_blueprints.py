import carla

def main():
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    
    # Get every 'vehicle' blueprint from the library
    blueprints = world.get_blueprint_library().filter('vehicle.*')

    print(f"{'Vehicle ID':<40} | {'Category'}")
    print("-" * 60)

    # Sort the list for readability
    for bp in sorted(blueprints, key=lambda x: x.id):
        # Some blueprints have attributes such as 'number_of_wheels'
        wheels = bp.get_attribute('number_of_wheels')
        print(f"{bp.id:<40} | {wheels} wheels")

if __name__ == '__main__':
    main()