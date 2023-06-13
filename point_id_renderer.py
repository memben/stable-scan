import moderngl
import numpy as np
from PIL import Image

def points_to_id(points: np.ndarray, MVP: np.ndarray, width: int, height: int, debug=False) -> np.ndarray:
    """Given the points as a numpy array of shape (n_points, 3) 
    and the MVP (4x4) matrix, return the numpy array of shape (width, height) 
    where each cell contains the id + 1 of the point that was rendered to that pixel. 
    Note that id = 0 means that no point was rendered. """
    id_buffer = PointIdRenderer().render(points, MVP, width, height, debug)
    return id_buffer

class PointIdRenderer:
    def __init__(self) -> None:
        vertex_shader = self.read_shader_file('vertex_id_color.vert')
        fragment_shader = self.read_shader_file('vertex_id_color.frag')

        self.context = moderngl.create_standalone_context()
        self.program = self.context.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        self.context.enable(moderngl.PROGRAM_POINT_SIZE)
        self.context.enable(moderngl.DEPTH_TEST)
        self.context.disable(moderngl.BLEND)
        self.context.multisample = False

    def render(self, points: np.ndarray, MVP, width, height, debug, point_size=1.0) -> np.ndarray:
        vbo = self.context.buffer(points.astype('f4').tobytes())
        # ids are 1-indexed, 0 is reserved for empty pixels
        ids = np.arange(1, points.shape[0] + 1)
        ibo = self.context.buffer(ids.astype('f4').tobytes())
        vao = self.context.vertex_array(self.program, [(vbo, '3f', 'vertex_position'), (ibo, 'f', 'vertex_index')])
        self.program['MVP'].write(MVP.astype('f4').tobytes())
        fbo = self.context.framebuffer(self.context.renderbuffer((width, height)))
        fbo.use()
        # set background to black
        fbo.clear(0.0, 0.0, 0.0, 0.0)

        vao.render(mode=moderngl.POINTS, vertices=points.shape[0])
        self.context.finish()
        if debug:
            self.debug_image(fbo)
        buffer = fbo.read(components=4, alignment=1)
        return self.buffer_to_id(buffer, width, height)

    def debug_image(self, fbo):
        # Note: alpha channel cannot be displayed
        Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()

    def buffer_to_id(self, buffer: bytes, width: int, height: int) -> np.ndarray:
        # Note: we invert the y axis here
        dt = np.dtype(np.uint32)
        # Little endian
        dt = dt.newbyteorder('<')
        rgba = np.frombuffer(buffer, dtype=dt).reshape((height, width))
        return rgba

    def read_shader_file(self, filename):
        with open(filename, 'r') as file:
            return file.read()
