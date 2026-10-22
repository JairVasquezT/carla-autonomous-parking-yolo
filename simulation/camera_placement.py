import carla

client = carla.Client('localhost', 2000)
world = client.get_world()

# Convert Unreal centimeters to CARLA meters
unreal_x, unreal_y, unreal_z = -900, 0, 400
carla_x, carla_y, carla_z = unreal_x / 100.0, unreal_y / 100.0, unreal_z / 100.0

# Place the camera
transform = carla.Transform(
    carla.Location(x=carla_x, y=carla_y, z=carla_z),
    carla.Rotation(pitch=-33.0, yaw=0.0, roll=0.0)
)

# Create the RGB camera
cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '1280')
cam_bp.set_attribute('image_size_y', '720')
camera = world.spawn_actor(cam_bp, transform)

# --- DISTANCE CHECK ---
# Print how far each object is from the camera
actores = world.get_actors().filter('vehicle.*')
for v in actores:
    dist = v.get_location().distance(transform.location)
    print(f"Vehicle {v.type_id} is {dist:.2f} m from the camera.")

# Save the image
camera.listen(lambda img: img.save_to_disk('fotos_parking/comprobacion_escala.png'))

import time
time.sleep(2)
camera.destroy()