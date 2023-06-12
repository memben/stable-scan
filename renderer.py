import moderngl
import numpy as np
from PIL import Image

ctx = moderngl.create_standalone_context()

prog = ctx.program(

    # inspired by https://github.com/isl-org/Open3D/blob/master/cpp/open3d/visualization/shader/glsl/PickingVertexShader.glsl
    vertex_shader='''
        #version 330

        in vec3 vertex_position;
        in float vertex_index;
        //uniform mat4 MVP;

        out vec4 fragment_color;

        void main()
        {
            float r, g, b, a;
            //gl_Position = MVP * vec4(vertex_position, 1);
            gl_Position = vec4(vertex_position, 1);

            r = floor(vertex_index / 16777216.0) / 255.0;
            g = mod(floor(vertex_index / 65536.0), 256.0) / 255.0;
            b = mod(floor(vertex_index / 256.0), 256.0) / 255.0;
            a = mod(vertex_index, 256.0) / 255.0;
            fragment_color = vec4(r, g, b, a);
        }
    ''',

    fragment_shader='''
        #version 330
        in vec4 fragment_color;
        out vec4 f_color;
        void main() {
            f_color = fragment_color;
        }
    ''',

)

width = 1024
height = 1024
n_points = width * height
# for each pixel create a point, with the index encoded in the color
points = np.zeros((n_points, 4), dtype=np.float32)
for y in range(1, height + 1):
    for x in range(1, width + 1):
        i = (y-1) * width + (x-1)
        xn = x / width * 2 - 1
        yn = y / height * 2 - 1
        points[i] = (xn, yn, 0, i)

ctx.enable(moderngl.PROGRAM_POINT_SIZE)
ctx.enable(moderngl.DEPTH_TEST)
ctx.disable(moderngl.BLEND)
ctx.multisample = False

vbo = ctx.buffer(points.astype('f4').tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'vertex_position', 'vertex_index')

fbo = ctx.framebuffer(ctx.renderbuffer((width, height)))
fbo.use()
fbo.clear(0.0, 0.0, 0.0, 1.0)
vao.render(mode=moderngl.POINTS, vertices=n_points)

ctx.finish()

rgba: bytes = fbo.read(components=4, alignment=1)
for x in range(0, 100, 4):
    r = rgba[x]
    g = rgba[x + 1]
    b = rgba[x + 2]
    a = rgba[x + 3]
    if r != 0 or g != 0 or b != 0 or a != 0:
        print(f'{x} {r} {g} {b} {a}')


Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()