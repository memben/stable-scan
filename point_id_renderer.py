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
        self.context = moderngl.create_standalone_context()
        self.program = self.context.program(vertex_shader=self.VERTEX_SHADER, fragment_shader=self.FRAGMENT_SHADER)

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

    VERTEX_SHADER = '''
    #version 330
    // inspired by https://github.com/isl-org/Open3D/blob/master/cpp/open3d/visualization/shader/glsl/PickingVertexShader.glsl
    in vec3 vertex_position;
    in float vertex_index;
    uniform mat4 MVP;

    out vec4 fragment_color;

    void main()
    {
        float r, g, b, a;
        gl_Position = MVP * vec4(vertex_position, 1);

        r = mod(vertex_index, 256.0) / 255.0;
        g = mod(floor(vertex_index / 256.0), 256.0) / 255.0;
        b = mod(floor(vertex_index / 65536.0), 256.0) / 255.0;
        a = floor(vertex_index / 16777216.0) / 255.0;
        fragment_color = vec4(r, g, b, a);
    }
    '''

    FRAGMENT_SHADER = '''
    #version 330
    in vec4 fragment_color;
    out vec4 f_color;

    void main() {
        f_color = fragment_color;
    }
    '''
