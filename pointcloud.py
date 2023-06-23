import moderngl
import numpy as np
from moderngl_window.opengl.vao import VAO

class PointCloud:
    def __init__(self, points: np.ndarray, colors: np.ndarray = None) -> None:
        # to provide a uniform camera experience
        self.points = normalize(points)
        if colors is None:
            points = np.random.rand(1000, 3).astype(np.float32)
            colors = np.random.rand(1000, 3).astype(np.float32)
        self.colors = colors
        self.vao = self.create_pc()

    def vao(self) -> VAO:
        return self.vao

    def create_pc(self) -> VAO:
        vao = VAO(mode=moderngl.POINTS)
        vbo = self.points.astype('f4').tobytes()
        vao.buffer(vbo, '3f', 'in_position')
        vbo = self.colors.astype('f4').tobytes()
        vao.buffer(vbo, '3f', 'in_color')
        return vao
    

def normalize(points: np.ndarray) -> np.ndarray:
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0) 
    ranges = max_coords - min_coords
    scaling_factor = 2.0 / np.max(ranges)
    normalized_points = (points - min_coords) * scaling_factor - 1.0
    return normalized_points