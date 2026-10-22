import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import time
import queue

# --- 1. PATHS AND DLL SETUP ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
carla_root = os.environ.get('CARLA_ROOT', r'C:\carla')
build_path = os.path.join(carla_root, 'Build')
if os.path.exists(build_path):
    for root, dirs, files in os.walk(build_path):
        if any(f.endswith('.dll') for f in files):
            try: os.add_dll_directory(root)
            except: pass

sys.path.append(os.path.join(carla_root, 'PythonAPI', 'carla'))
import carla

# --- GEOMETRIC CONSTANTS ---
H = np.array([
    [ -1.35928836e-08, -5.39979638e-02,  1.54643018e+01],
    [ 3.89898819e-02,  2.71380886e-05, -2.49401315e+01],
    [ 1.84855595e-07,  2.18687425e-03,  1.00000000e+00]
])

L, W, HT = 4.0, 1.8, 1.5  # Citroën C3 length, width, height (m)

# --- VIDEO SETUP ---
W_vid, H_vid = 1280, 720
FPS = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('parking_maneuver_yolo.mp4', fourcc, FPS, (W_vid, H_vid))

# --- GEOMETRY FUNCTIONS ---
def pixel_a_metros(px, py, matrix_h):
    """Transform pixels into CARLA X, Y coordinates using the homography"""
    punto = np.array([[[px, py]]], dtype='float32')
    res = cv2.perspectiveTransform(punto, matrix_h)
    return res[0][0]

def proyectar_punto_3d(x, y, z, world_2_cam, K):
    """Project a 3D world point onto the 2D image using the camera"""
    p_world = np.array([x, y, z, 1])
    p_cam = np.dot(world_2_cam, p_world)
    p_cv = np.array([p_cam[1], -p_cam[2], p_cam[0]])
    if p_cv[2] > 0.1:
        p_2d = np.dot(K, p_cv)
        p_2d /= p_2d[2]
        return (int(p_2d[0]), int(p_2d[1]))
    return None

def obtener_puntos_cubo_mundo(centro_x, centro_y, yaw_deg):
    """Generate the 8 box corners in CARLA 3D world space"""
    rad = np.radians(yaw_deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]])
    
    l, w = L/2, W/2
    puntos_mundo = []
    
    for i in range(8):
        lx = l if i in [0, 1, 4, 5] else -l
        ly = w if i in [0, 3, 4, 7] else -w
        lz = 0 if i < 4 else HT
        
        rot = np.dot(R, [lx, ly])
        puntos_mundo.append([centro_x + rot[0], centro_y + rot[1], lz])
        
    return np.array(puntos_mundo)

def draw_reconstructed_cube(image, points_2d, color=(0, 255, 255)):
    """Draw the edges of the reconstructed box"""
    if len(points_2d) != 8: return image
    for i in range(4):
        cv2.line(image, points_2d[i], points_2d[(i+1)%4], color, 2)
        cv2.line(image, points_2d[i+4], points_2d[((i+1)%4)+4], color, 2)
        cv2.line(image, points_2d[i], points_2d[i+4], color, 2)
    return image

def obtener_posicion_visual(frame, vehicle, camera, model, K, H_mat, world_2_cam, x_init=None, y_init=None):
    centro_x_real, centro_y_real = None, None
    frame_out = frame.copy()
    
    results = model(frame, conf=0.5, verbose=False)
    
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy()
        x_min_bb, y_min_bb, x_max_bb, y_max_bb = box
        alto_bb = y_max_bb - y_min_bb
        
        cv2.rectangle(frame_out, (int(x_min_bb), int(y_min_bb)), (int(x_max_bb), int(y_max_bb)), (255, 0, 0), 2)
        
        cx_yolo = (x_min_bb + x_max_bb) / 2.0
        
        # Ignore the shadow: raise the target base by 5% of the box height
        # so it targets the wheels rather than the darkened asphalt.
        y_base_yolo = y_max_bb - (alto_bb * 0.05) 
        
        if x_init is not None and y_init is not None:
            centro_x, centro_y = x_init, y_init
        else:
            p_homo = pixel_a_metros(cx_yolo, y_base_yolo, H_mat)
            centro_x, centro_y = p_homo[0], p_homo[1]
        
        yaw = vehicle.get_transform().rotation.yaw
        eps = 0.02
        alpha = 0.5 
        
        for _ in range(12):
            p_3d = obtener_puntos_cubo_mundo(centro_x, centro_y, yaw)
            p_2d = [proyectar_punto_3d(p[0], p[1], p[2], world_2_cam, K) for p in p_3d]
            
            if None in p_2d: break
            
            # Instead of averaging points, use the center of the bounding box
            # spanned by the projected 3D points, which is exactly what YOLO measures.
            x_coords_act = [p[0] for p in p_2d]
            y_coords_act = [p[1] for p in p_2d]
            cx_act = (min(x_coords_act) + max(x_coords_act)) / 2.0 
            y_act = max(y_coords_act)
            
            err_x = (cx_yolo - cx_act) * 2.0
            err_y = y_base_yolo - y_act
            if abs(err_x) < 0.4 and abs(err_y) < 0.2: break

            # Jacobian w.r.t. X (same min/max logic)
            p_3d_dx = obtener_puntos_cubo_mundo(centro_x + eps, centro_y, yaw)
            p_2d_dx = [proyectar_punto_3d(p[0], p[1], p[2], world_2_cam, K) for p in p_3d_dx]
            if None in p_2d_dx: break
            x_coords_dx = [p[0] for p in p_2d_dx]
            cx_dx = (min(x_coords_dx) + max(x_coords_dx)) / 2.0
            yb_dx = max([p[1] for p in p_2d_dx])
            
            # Jacobian w.r.t. Y (same min/max logic)
            p_3d_dy = obtener_puntos_cubo_mundo(centro_x, centro_y + eps, yaw)
            p_2d_dy = [proyectar_punto_3d(p[0], p[1], p[2], world_2_cam, K) for p in p_3d_dy]
            if None in p_2d_dy: break
            x_coords_dy = [p[0] for p in p_2d_dy]
            cx_dy = (min(x_coords_dy) + max(x_coords_dy)) / 2.0
            yb_dy = max([p[1] for p in p_2d_dy])
            
            J11 = (cx_dx - cx_act) / eps
            J12 = (cx_dy - cx_act) / eps
            J21 = (yb_dx - y_act) / eps
            J22 = (yb_dy - y_act) / eps
            
            det = (J11 * J22) - (J12 * J21)
            if abs(det) < 1e-6: break
            
            centro_x += (J22 * err_x - J12 * err_y) / det * alpha
            centro_y += (-J21 * err_x + J11 * err_y) / det * alpha
        
        # Temporal smoothing (EMA)
        alpha_ema = 0.7
        if x_init is not None:
            centro_x = alpha_ema * centro_x + (1.0 - alpha_ema) * x_init
        if y_init is not None:
            centro_y = alpha_ema * centro_y + (1.0 - alpha_ema) * y_init
        
        # Drawing and results
        # (after the optimization loop)

        # 5. Final drawing and results
        p_final_3d = obtener_puntos_cubo_mundo(centro_x, centro_y, yaw)
        p_final_2d = [proyectar_punto_3d(p[0], p[1], p[2], world_2_cam, K) for p in p_final_3d]
        
        if all(p is not None for p in p_final_2d):
            # --- BOX ALIGNMENT ---
            # 1. Compute the center of the box base (points 0-3 lie on the ground)
            p_base_2d = p_final_2d[:4]
            x_coords_base = [p[0] for p in p_base_2d]
            y_coords_base = [p[1] for p in p_base_2d]
            
            cx_base_cubo = (min(x_coords_base) + max(x_coords_base)) / 2.0
            y_max_base_cubo = max(y_coords_base) # Real ground contact point

            # 2. Compute the shift so that the box BASE drives the drawing
            dx_shift = cx_yolo - cx_base_cubo
            dy_shift = y_base_yolo - y_max_base_cubo
            
            # 3. Apply the shift to all 8 points for drawing

            p_final_2d_ajustados = [(int(p[0] + dx_shift), int(p[1] + dy_shift)) for p in p_final_2d]

            # --- 4. UPDATE WORLD COORDINATES (CORRECTED) ---
            # 4.1 Project the car center (Z=0) to find its pixel
            p_centro_pixel = proyectar_punto_3d(centro_x, centro_y, 0, world_2_cam, K)
            if p_centro_pixel is not None:
                # 4.2 Add the shift so the center matches the corrected drawing
                px_centro_ajustado = p_centro_pixel[0] + dx_shift
                py_centro_ajustado = p_centro_pixel[1] + dy_shift
                
                # 4.3 Pass the CENTER pixel (not the bumper) through the homography
                p_metros_final = pixel_a_metros(px_centro_ajustado, py_centro_ajustado, H_mat)
                centro_x_real = p_metros_final[0]
                centro_y_real = p_metros_final[1]
            else:
                centro_x_real, centro_y_real = centro_x, centro_y

            
            # Draw the box and the label
            draw_reconstructed_cube(frame_out, p_final_2d_ajustados, color=(0, 255, 0))
            
            cv2.putText(frame_out, f"Pos: {centro_x_real:.2f}x, {centro_y_real:.2f}y",
                        (int(x_min_bb), int(y_min_bb)-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return centro_x_real, centro_y_real, frame_out


def main():
    actor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    
    try:
        # --- Load YOLO ---
        print("Loading YOLO...")
        model = YOLO(os.path.join(PROJECT_ROOT, 'models', 'best_v2.pt'))
        
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 1.0 / FPS # Matches the video frame rate
        world.apply_settings(settings)

        # --- BEV CAMERA SETUP ---
        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(W_vid))
        cam_bp.set_attribute('image_size_y', str(H_vid))
        cam_tr = carla.Transform(carla.Location(x=-11.0, y=0.0, z=11.0), carla.Rotation(pitch=-52.0))
        camera = world.spawn_actor(cam_bp, cam_tr)
        actor_list.append(camera)

        # --- Intrinsic matrix K ---
        f = W_vid / (2.0 * np.tan(90.0 * np.pi / 360.0))
        K = np.array([[f, 0, W_vid/2], [0, f, H_vid/2], [0, 0, 1]])
        
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
                last_x, last_y = None, None
                is_arriba = int(slot_id) <= 4
                spawn_x = slot_loc.x + 4.9 if is_arriba else slot_loc.x - 4.9
                spawn_y = slot_loc.y - 3.2+8 if lado == "izquierda" else slot_loc.y + 3.2-8
                spawn_yaw = -90 if lado == "izquierda" else 90
                
                vehicle = world.try_spawn_actor(bp_c3, carla.Transform(carla.Location(x=spawn_x, y=spawn_y, z=0.1), carla.Rotation(yaw=spawn_yaw)))
                if not vehicle: continue
                actor_list.append(vehicle)
                
                # Let the car settle while recording idle frames
                for _ in range(30):
                    world.tick()
                    img = image_queue.get()
                    frame = np.reshape(np.frombuffer(img.raw_data, dtype=np.uint8), (H_vid, W_vid, 4))[:, :, :3]
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

                
                # --- 0. INITIAL FORWARD APPROACH ---
                while True:
                    world.tick()
                    img = image_queue.get()
                    
                    # Use the camera matrix of the current frame
                    world_2_cam = np.array(img.transform.get_inverse_matrix())
                    frame = np.reshape(np.frombuffer(img.raw_data, dtype=np.uint8), (H_vid, W_vid, 4))[:, :, :3]
                    
                    vx, vy, frame_dibujado = obtener_posicion_visual(frame, vehicle, camera, model, K, H, world_2_cam, last_x, last_y)
                    
                    if vy is not None:
                        last_x, last_y = vx, vy
                        
                        # Maneuver parameters
                        dist_objetivo = 3.2
                        margen_frenado = 0.05 
                        
                        # Stop logic
                        reached_left = (lado == "izquierda" and vy <= (slot_loc.y - dist_objetivo + margen_frenado))
                        reached_right = (lado == "derecha" and vy >= (slot_loc.y + dist_objetivo - margen_frenado))

                        if reached_left or reached_right:
                            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                            for _ in range(15):
                                world.tick()
                                _ = image_queue.get() # Keep the buffer empty
                                video_out.write(frame_dibujado) 
                            break

                        # Proportional speed control
                        throttle_p = 0.4 if abs(vy - (slot_loc.y + (dist_objetivo if lado=="derecha" else -dist_objetivo))) > 0.4 else 0.15
                        vehicle.apply_control(carla.VehicleControl(throttle=throttle_p, steer=0.0))
                    else:
                        # Safe mode if the detection is lost
                        vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))

                    video_out.write(frame_dibujado)


                # --- 1. STEERING + REVERSE MANEUVER ---
                # (Uses yaw_actual and steer_val)
                # --- 2. STEERING PHASE ---
                while True:
                    world.tick()
                    img = image_queue.get()
                    frame = np.reshape(np.frombuffer(img.raw_data, dtype=np.uint8), (H_vid, W_vid, 4))[:, :, :3]
                    world_2_cam = np.array(camera.get_transform().get_inverse_matrix())
                    vx, vy, frame_dibujado = obtener_posicion_visual(frame, vehicle, camera, model, K, H, world_2_cam, last_x, last_y)
                    
                    # 2. KEEP POSITION FOR THE NEXT FRAME
                    if vx is not None:
                        last_x, last_y = vx, vy
                        
                    yaw_actual = vehicle.get_transform().rotation.yaw
                    vehicle.apply_control(carla.VehicleControl(throttle=0.40, steer=steer_val, reverse=True))
                    video_out.write(frame_dibujado)  # Write frame with YOLO box + 3D box overlay

                    if condicion_giro(yaw_actual):
                        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                        for _ in range(10): 
                            world.tick()
                            img_temp = image_queue.get()
                            frame_temp = np.reshape(np.frombuffer(img_temp.raw_data, dtype=np.uint8), (H_vid, W_vid, 4))[:, :, :3]
                            video_out.write(frame_temp)
                        break

                # --- 2. FINAL STRAIGHT REVERSE ---
                while True:
                    world.tick()
                    img = image_queue.get()
                    
                    world_2_cam = np.array(img.transform.get_inverse_matrix())
                    frame = np.reshape(np.frombuffer(img.raw_data, dtype=np.uint8), (H_vid, W_vid, 4))[:, :, :3]
                    
                    centro_x_visual, centro_y_visual, frame_dibujado = obtener_posicion_visual(
                        frame, vehicle, camera, model, K, H, world_2_cam, last_x, last_y
                    )
                    
                    stop = False # Reset the stop flag

                    if centro_x_visual is not None:
                        last_x, last_y = centro_x_visual, centro_y_visual

                        # Speed control (inside the if to avoid a NoneType error)
                        vel_retroceso = 0.35 if abs(centro_x_visual) > 1.0 else 0.15
                        vehicle.apply_control(carla.VehicleControl(throttle=vel_retroceso, steer=0.0, reverse=True))

                        # Final stop logic
                        margen_final = 0.05 
                        if is_arriba:
                            stop = (centro_x_visual <= (2.55 + margen_final))
                        else:
                            stop = (centro_x_visual >= (-2.55 - margen_final))
                    else:
                        # If the detection is lost, reverse very slowly for safety
                        vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=0.0, reverse=True))
                    
                    video_out.write(frame_dibujado)

                    if stop:
                        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                        for _ in range(20):
                            world.tick()
                            _ = image_queue.get()
                            video_out.write(frame_dibujado)
                        break
                vehicle.destroy()
                if vehicle in actor_list: actor_list.remove(vehicle)
                print(f"Maneuver {slot_id} {lado} recorded.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        video_out.release()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for a in actor_list:
            if a.is_alive: a.destroy()
        print("Video saved as 'parking_maneuver_yolo.mp4'")

if __name__ == '__main__':
    main()