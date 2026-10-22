import numpy as np
import cv2

class Carla3DHelper:
    def __init__(self, camera_actor, w=1280, h=720, fov=90.0):
        self.camera = camera_actor
        self.w = w
        self.h = h
        # Exact K for the CARLA camera
        f = w / (2.0 * np.tan(fov * np.pi / 360.0))
        self.K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])

    def get_car_points_2d(self, vehicle_actor):
        """
        Return the 8 projected 2D corners of the 3D box
        using CARLA's GROUND-TRUTH pose (for validation or a perfect demo).
        """
        world_2_cam = np.array(self.camera.get_transform().get_inverse_matrix())
        # Vehicle bounding box vertices
        bb = vehicle_actor.bounding_box
        verts = bb.get_world_vertices(vehicle_actor.get_transform())
        
        points_2d = []
        for v in verts:
            p_world = np.array([v.x, v.y, v.z, 1])
            p_cam = np.dot(world_2_cam, p_world)
            
            # CARLA to OpenCV axes (Y, -Z, X)
            p_cv = np.array([p_cam[1], -p_cam[2], p_cam[0]])
            
            if p_cv[2] > 0.1: # Only if in front of the camera
                p_2d = np.dot(self.K, p_cv)
                p_2d /= p_2d[2]
                points_2d.append((int(p_2d[0]), int(p_2d[1])))
        
        return points_2d

    def draw_cube(self, img, pts, color=(0, 255, 0)):
        """Draw the box edges if all 8 points are available"""
        if len(pts) != 8: return img
        
        # Box edges (CARLA vertex indices)
        # Base: 0-1-2-3, Roof: 4-5-6-7, Pillars: 0-4, 1-5, 2-6, 3-7
        for i in range(4):
            cv2.line(img, pts[i], pts[(i+1)%4], color, 2)       # Base
            cv2.line(img, pts[i+4], pts[((i+1)%4)+4], color, 2) # Techo
            cv2.line(img, pts[i], pts[i+4], color, 2)           # Columnas
        return img