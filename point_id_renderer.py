import moderngl
import numpy as np

def points_to_id(points: np.ndarray, width: int, height: int) -> np.ndarray:
    id_buffer = PointIdRenderer().render(points, width, height)
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

    def render(self, points: np.ndarray, width, height, point_size=1.0) -> np.ndarray:
        vbo = self.context.buffer(points.astype('f4').tobytes())
        ids = np.arange(1, points.shape[0] + 1)
        ibo = self.context.buffer(ids.astype('f4').tobytes())
        vao = self.context.vertex_array(self.program, [(vbo, '3f', 'vertex_position'), (ibo, 'f', 'vertex_index')])
        fbo = self.context.framebuffer(self.context.renderbuffer((width, height)))
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)

        vao.render(mode=moderngl.POINTS, vertices=points.shape[0])
        self.context.finish()

        buffer = fbo.read(components=4, alignment=1)
        return self.buffer_to_id(buffer, width, height)

    def buffer_to_id(self, buffer: bytes, width: int, height: int) -> np.ndarray:
        # Note: we invert the y axis here
        dt = np.dtype(np.uint32)
        # Big endian
        dt = dt.newbyteorder('>')
        rgba = np.frombuffer(buffer, dtype=dt).reshape((height, width))
        return rgba

    def read_shader_file(self, filename):
        with open(filename, 'r') as file:
            return file.read()
