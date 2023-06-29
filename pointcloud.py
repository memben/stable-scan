import moderngl
import numpy as np
from PIL import Image
from moderngl_window.opengl.vao import VAO

class PointCloud:
    def __init__(self, points: np.ndarray, colors: np.ndarray = None, point_size: float = 3.0) -> None:
        # to provide a uniform camera experience
        self.points = normalize(points)
        self.point_size = point_size
        if colors is None:
            colors = np.random.rand(points.shape[0], 3).astype(np.float32)  
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
    
    def retexture(self, texture: Image, ids: np.ndarray) -> None:
        '''Given a texture and a 2D array of ids, retexture the point cloud.'''
        texture.show()
        for y in range(texture.height):
            for x in range(texture.width):
                color = texture.getpixel((x, y))
                color = np.array(color, dtype=np.float32) / 255.0
                id = ids[y, x]
                if id == 2**32 - 1: continue
                self.colors[id] = color  
        self.vao = None
    
    def exclusive_retexture(self, texture: Image, ids: np.ndarray) -> None:
        '''Discard all points except those with ids in the ids array.'''
        self.retexture(texture, ids)
    
        u_ids = np.unique(ids.flatten())
        u_ids = u_ids[u_ids != 2**32 - 1]
        self.points = self.points[u_ids]
        self.colors = self.colors[u_ids]

    
    
# TODO(memben): Slighly shifts the point cloud one pixel to the bottom and right.
def normalize(points: np.ndarray) -> np.ndarray:
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0) 
    ranges = max_coords - min_coords
    scaling_factor = 2.0 / np.max(ranges)
    normalized_points = (points - min_coords) * scaling_factor - 1.0
    return normalized_points