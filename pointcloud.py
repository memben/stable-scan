import moderngl
import numpy as np
from moderngl_window.opengl.vao import VAO

class PointCloud:
    def __init__(self, points: np.ndarray, colors: np.ndarray = None, point_size: float = 3.0) -> None:
        # to provide a uniform camera experience
        self.points = normalize(points)
        self.point_size = point_size
        if colors is None:
            points = np.random.rand(1000, 3).astype(np.float32)
            colors = np.random.rand(1000, 3).astype(np.float32)
        self.colors = colors
        self.vao = None

    def get_vao(self) -> VAO:
        if self.vao is None:
            self.vao = self.create_pc()
        return self.vao
    
    def get_va_from(self, ctx: moderngl.Context, program: moderngl.Program) -> moderngl.VertexArray:
        '''Create a vertex array from the points and colors.'''
        vbo_points = ctx.buffer(self.points.astype('f4').tobytes())
        vbo_colors = ctx.buffer(self.colors.astype('f4').tobytes())
        va = ctx.vertex_array(program, [(vbo_points, '3f', 'in_position'), (vbo_colors, '3f', 'in_color')])
        return va
        

    def create_pc(self) -> VAO:
        vao = VAO(mode=moderngl.POINTS)
        vbo = self.points.astype('f4').tobytes()
        vao.buffer(vbo, '3f', 'in_position')
        vbo = self.colors.astype('f4').tobytes()
        vao.buffer(vbo, '3f', 'in_color')
        return vao
    
# TODO(memben): Slighly shifts the point cloud one pixel to the bottom and right.
def normalize(points: np.ndarray) -> np.ndarray:
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0) 
    ranges = max_coords - min_coords
    scaling_factor = 2.0 / np.max(ranges)
    normalized_points = (points - min_coords) * scaling_factor - 1.0
    return normalized_points