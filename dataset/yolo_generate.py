#!/usr/bin/env python3
"""
YOLO dataset generator for CARLA - projection matrix fix
Strict axis transformation: UE4 (X-fwd, Y-right, Z-up) -> OpenCV (X-right, Y-down, Z-fwd)
"""

import carla
import numpy as np
import cv2
import os
import logging
import queue
import time
from pathlib import Path

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================

# Paths
DATASET_DIR = Path(__file__).parent / "dataset_parking_c3"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# Create directories
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Camera parameters
CAMERA_POSITION = carla.Location(x=-11.0, y=0.0, z=11.0)
CAMERA_ROTATION = carla.Rotation(pitch=-52.0, yaw=0.0, roll=0.0)
IMG_WIDTH = 1280
IMG_HEIGHT = 720
FOV = 90.0

# 16 parking trajectories (8 slots × 2 sides)
PARKING_MANEUVERS = [
    # Slots 1-4 RIGHT (x_spawn=7.4)
    {"slot": "1", "lado": "derecha", "spawn_x": 7.4, "spawn_y": -0.55, "spawn_yaw": 90, "steer_angle": 0.5},
    {"slot": "2", "lado": "derecha", "spawn_x": 7.4, "spawn_y": 1.95, "spawn_yaw": 90, "steer_angle": 0.5},
    {"slot": "3", "lado": "derecha", "spawn_x": 7.4, "spawn_y": 4.45, "spawn_yaw": 90, "steer_angle": 0.5},
    {"slot": "4", "lado": "derecha", "spawn_x": 7.4, "spawn_y": 6.95, "spawn_yaw": 90, "steer_angle": 0.5},
    
    # Slots 1-4 LEFT (x_spawn=7.4)
    {"slot": "1", "lado": "izquierda", "spawn_x": 7.4, "spawn_y": -6.95, "spawn_yaw": -90, "steer_angle": -0.5},
    {"slot": "2", "lado": "izquierda", "spawn_x": 7.4, "spawn_y": -4.45, "spawn_yaw": -90, "steer_angle": -0.5},
    {"slot": "3", "lado": "izquierda", "spawn_x": 7.4, "spawn_y": -1.95, "spawn_yaw": -90, "steer_angle": -0.5},
    {"slot": "4", "lado": "izquierda", "spawn_x": 7.4, "spawn_y": 0.55, "spawn_yaw": -90, "steer_angle": -0.5},
    
    # Slots 5-8 RIGHT (x_spawn=-7.4)
    {"slot": "5", "lado": "derecha", "spawn_x": -7.4, "spawn_y": -0.55, "spawn_yaw": 90, "steer_angle": 0.5},
    {"slot": "6", "lado": "derecha", "spawn_x": -7.4, "spawn_y": 1.95, "spawn_yaw": 90, "steer_angle": 0.5},
    {"slot": "7", "lado": "derecha", "spawn_x": -7.4, "spawn_y": 4.45, "spawn_yaw": 90, "steer_angle": 0.5},
    {"slot": "8", "lado": "derecha", "spawn_x": -7.4, "spawn_y": 6.95, "spawn_yaw": 90, "steer_angle": 0.5},
    
    # Slots 5-8 LEFT (x_spawn=-7.4)
    {"slot": "5", "lado": "izquierda", "spawn_x": -7.4, "spawn_y": -6.95, "spawn_yaw": -90, "steer_angle": -0.5},
    {"slot": "6", "lado": "izquierda", "spawn_x": -7.4, "spawn_y": -4.45, "spawn_yaw": -90, "steer_angle": -0.5},
    {"slot": "7", "lado": "izquierda", "spawn_x": -7.4, "spawn_y": -1.95, "spawn_yaw": -90, "steer_angle": -0.5},
    {"slot": "8", "lado": "izquierda", "spawn_x": -7.4, "spawn_y": 0.55, "spawn_yaw": -90, "steer_angle": -0.5},
]

# ==============================================================================
# PROJECTION FUNCTIONS
# ==============================================================================

def get_camera_intrinsic_matrix(width, height, fov):
    """
    Compute the intrinsic matrix K for OpenCV.
    
    Args:
        width: Image width (pixels)
        height: Image height (pixels)
        fov: Horizontal field of view (degrees)
    
    Returns:
        3x3 intrinsic matrix
    """
    # Focal length from the horizontal FOV
    f = width / (2.0 * np.tan(fov * np.pi / 360.0))
    
    K = np.array([
        [f, 0, width / 2.0],
        [0, f, height / 2.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    logger.debug(f"Intrinsic matrix K:\n{K}")
    logger.debug(f"Focal length: {f:.2f}")
    
    return K


def project_bbox_3d_to_2d(vehicle, camera_transform, K):
    """
    Project the vehicle 3D bounding box to 2D image coordinates.
    
    Strict axis transformation:
    - UE4: X-forward, Y-right, Z-up
    - OpenCV: X-right, Y-down, Z-forward
    
    Args:
        vehicle: CARLA vehicle actor
        camera_transform: Camera transform
        K: 3x3 intrinsic matrix
    
    Returns:
        Tuple (x_min, x_max, y_min, y_max, visible_count), or None on error
    """
    try:
        # 1. Get the bounding box vertices in world coordinates
        verts = vehicle.bounding_box.get_world_vertices(vehicle.get_transform())
        
        # 2. Transformation matrix: world -> camera
        world_2_camera = np.array(camera_transform.get_inverse_matrix())
        
        coords_2d = []
        
        for vertex in verts:
            # Point in world coordinates (homogeneous)
            p_world = np.array([vertex.x, vertex.y, vertex.z, 1.0])
            
            # Transform to camera space (4x4 matrix @ 4-vector)
            p_camera = np.dot(world_2_camera, p_world)
            
            # AXIS TRANSFORMATION: UE4 -> OpenCV
            # UE4: X-fwd, Y-right, Z-up
            # OpenCV: X-right, Y-down, Z-fwd
            x_pixel = p_camera[1]          # UE4 Y -> OpenCV X (width)
            y_pixel = -p_camera[2]         # UE4 -Z -> OpenCV Y (height, inverted)
            z_depth = p_camera[0]          # UE4 X -> OpenCV Z (depth)
            
            # 3. Project only if in front of the camera
            if z_depth > 0.01:  # Avoid division by zero
                # Projection: u = f * x / z + cx, v = f * y / z + cy
                u = (K[0, 0] * x_pixel / z_depth) + K[0, 2]
                v = (K[1, 1] * y_pixel / z_depth) + K[1, 2]
                
                coords_2d.append((u, v))
        
        # 4. Validation: at least 3 visible points are needed
        if len(coords_2d) < 3:
            logger.warning(f"Only {len(coords_2d)} visible points (needs ≥3)")
            return None
        
        # 5. Compute bounding box
        x_coords = [p[0] for p in coords_2d]
        y_coords = [p[1] for p in coords_2d]
        
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # DEBUG: log when out of range
        if x_min < 0 or x_max > IMG_WIDTH or y_min < 0 or y_max > IMG_HEIGHT:
            logger.warning(
                f"⚠️ BBox partially outside: X({x_min:.1f}-{x_max:.1f}) "
                f"Y({y_min:.1f}-{y_max:.1f}) - will be clipped"
            )
        
        # Clamp to the valid image area
        x_min_clipped = max(0, int(np.floor(x_min)))
        x_max_clipped = min(IMG_WIDTH, int(np.ceil(x_max)))
        y_min_clipped = max(0, int(np.floor(y_min)))
        y_max_clipped = min(IMG_HEIGHT, int(np.ceil(y_max)))
        
        # Make sure the area is still positive
        if x_max_clipped <= x_min_clipped or y_max_clipped <= y_min_clipped:
            logger.warning("Clipped bbox has zero area")
            return None
        
        return (x_min_clipped, x_max_clipped, y_min_clipped, y_max_clipped, len(coords_2d))
        
    except Exception as e:
        logger.error(f"3D->2D projection error: {e}")
        return None


def save_yolo_annotation(image_path, bbox_result, img_width, img_height):
    """
    Save an annotation in YOLO format (normalized center + normalized width/height).
    
    Args:
        image_path: Path of the saved image
        bbox_result: Tuple (x_min, x_max, y_min, y_max, visible_count)
        img_width: Image width
        img_height: Image height
    """
    if bbox_result is None:
        return False
    
    x_min, x_max, y_min, y_max, _ = bbox_result
    
    # Convert to YOLO format: class x_center y_center width height (all normalized 0-1)
    center_x = (x_min + x_max) / 2.0 / img_width
    center_y = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Class 0 = vehicle
    yolo_line = f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
    
    # Save label
    label_path = image_path.replace("/images/", "/labels/").replace(".png", ".txt")
    with open(label_path, "w") as f:
        f.write(yolo_line)
    
    logger.debug(f"✓ Label saved: {label_path}")
    return True


def process_frame(carla_image, vehicle, camera_transform, K, frame_counter):
    """
    Process one frame: project the bbox, validate it and save image + label.
    
    Args:
        carla_image: Raw CARLA image
        vehicle: Vehicle actor
        camera_transform: Camera transform
        K: Intrinsic matrix
        frame_counter: Frame number used in the file name
    
    Returns:
        True if saved, False otherwise
    """
    # Convert CARLA image to OpenCV
    array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (carla_image.height, carla_image.width, 4))
    array = array[:, :, :3]  # CARLA gives BGRA; keep BGR
    
    # Project bbox
    bbox_result = project_bbox_3d_to_2d(vehicle, camera_transform, K)
    
    if bbox_result is None:
        return False
    
    x_min, x_max, y_min, y_max, visible_count = bbox_result
    
    # Save image
    image_filename = IMAGES_DIR / f"frame_{frame_counter:06d}.png"
    cv2.imwrite(str(image_filename), array)
    
    # Save YOLO label
    success = save_yolo_annotation(str(image_filename), bbox_result, IMG_WIDTH, IMG_HEIGHT)
    
    if success:
        logger.info(
            f"✓ Frame {frame_counter}: {visible_count} visible vertices, "
            f"BBox=[{x_min}-{x_max}, {y_min}-{y_max}]"
        )
    
    return success


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    client = None
    world = None
    camera = None
    actor_list = []
    
    try:
        # Connect to CARLA
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        logger.info("✓ Connected to CARLA")
        
        # Map setup
        world.load_map_layer(carla.MapLayer.All)
        
        # Get blueprint library
        blueprint_lib = world.get_blueprint_library()
        
        # Create RGB camera
        camera_bp = blueprint_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(IMG_WIDTH))
        camera_bp.set_attribute("image_size_y", str(IMG_HEIGHT))
        camera_bp.set_attribute("fov", str(FOV))
        
        camera_transform = carla.Transform(CAMERA_POSITION, CAMERA_ROTATION)
        camera = world.spawn_actor(camera_bp, camera_transform)
        actor_list.append(camera)
        
        logger.info(f"✓ Camera created at {CAMERA_POSITION} with pitch={CAMERA_ROTATION.pitch}°")
        
        # Synchronized image queue
        image_queue = queue.Queue(maxsize=2)
        camera.listen(image_queue.put)
        
        # Intrinsic matrix
        K = get_camera_intrinsic_matrix(IMG_WIDTH, IMG_HEIGHT, FOV)
        
        # Vehicle blueprint
        vehicle_bp = blueprint_lib.find("vehicle.citroen.c3")
        
        frame_counter = 0
        saved_count = 0
        
        # ==== RUN 16 MANEUVERS ====
        for maneuver_idx, maneuver in enumerate(PARKING_MANEUVERS):
            logger.info(
                f"\n{'='*60}"
                f"\n[{maneuver_idx+1}/16] Iniciando maniobra: "
                f"Slot {maneuver['slot']} ({maneuver['lado']})"
                f"\n{'='*60}"
            )
            
            # Spawn vehicle
            spawn_transform = carla.Transform(
                carla.Location(x=maneuver['spawn_x'], y=maneuver['spawn_y'], z=0.1),
                carla.Rotation(yaw=maneuver['spawn_yaw'])
            )
            
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
            if not vehicle:
                logger.warning(f"❌ Could not spawn vehicle for maneuver {maneuver_idx+1}")
                continue
            
            actor_list.append(vehicle)
            
            # Wait for the car to settle
            logger.debug("Waiting for the vehicle to settle...")
            for _ in range(10):
                world.tick()
            
            # Run maneuver (15 seconds = ~300 ticks at 20 FPS)
            logger.info("Starting frame capture...")
            maniobra_frames = 0
            
            for step in range(300):
                world.tick()
                
                # Apply control
                steer = maneuver['steer_angle']
                vehicle.apply_control(carla.VehicleControl(
                    throttle=0.35,
                    steer=steer,
                    reverse=True
                ))
                
                # Try to get an image (non-blocking)
                try:
                    carla_image = image_queue.get_nowait()
                except queue.Empty:
                    continue
                
                # Process frame (save every 5 frames)
                if step % 5 == 0:
                    if process_frame(carla_image, vehicle, camera_transform, K, frame_counter):
                        saved_count += 1
                        maniobra_frames += 1
                    frame_counter += 1
            
            # Final braking
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            for _ in range(10):
                world.tick()
            
            logger.info(f"✓ Maneuver complete: {maniobra_frames} frames saved")
            
            # Destroy vehicle
            vehicle.destroy()
            actor_list.remove(vehicle)
        
        # Final summary
        logger.info(
            f"\n{'='*60}"
            f"\n✓ GENERATION COMPLETE"
            f"\n{'='*60}"
            f"\nFrames guardados: {saved_count}"
            f"\nDirectorio: {DATASET_DIR}"
            f"\nImages: {IMAGES_DIR}"
            f"\nEtiquetas: {LABELS_DIR}"
            f"\n{'='*60}"
        )
        
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}", exc_info=True)
    
    finally:
        # Cleanup
        if camera is not None:
            camera.destroy()
        
        for actor in actor_list:
            try:
                actor.destroy()
            except:
                pass
        
        logger.info("✓ Cleanup complete")


if __name__ == "__main__":
    main()
