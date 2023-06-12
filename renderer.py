import moderngl
import numpy as np
from PIL import Image

ctx = moderngl.create_standalone_context()

prog = ctx.program(

    vertex_shader='''
        #version 330
        in vec3 in_vert;
        in vec3 in_color;
        out vec3 v_color;
        void main() {
            v_color = in_color;
            gl_Position = vec4(in_vert, 1.0);
            gl_PointSize = 10.0;
        }
    ''',

    fragment_shader='''
        #version 330
        in vec3 v_color;
        out vec3 f_color;
        void main() {
            f_color = v_color;
        }
    ''',

)

x = np.random.rand(50) - 0.5
y = np.random.rand(50) - 0.5
z = np.random.rand(50) - 0.5
x *= 2.0
y *= 2.0
z *= 2.0
r = np.ones(50)
g = np.zeros(50)
b = np.zeros(50)

vertices = np.dstack([x, y, z, r, g, b])
ctx.enable_only(moderngl.PROGRAM_POINT_SIZE)
vbo = ctx.buffer(vertices.astype('f4').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_color')
fbo = ctx.simple_framebuffer((512, 512))
fbo.use()
fbo.clear(0.0, 0.0, 0.0, 1.0)
vao.render(mode=moderngl.POINTS, vertices=50)
Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()