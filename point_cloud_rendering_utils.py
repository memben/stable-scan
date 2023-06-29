import moderngl
import numpy as np
import pointcloud
import depth_utils
from PIL import Image
from pathlib import Path


def inject_string(shader_code, injection):
    # "#version xxx" must be the first line
    lines = shader_code.split('\n')
    index = 0

    for i, line in enumerate(lines):
        if line.strip() != "":
            index = i + 1
            break

    lines.insert(index, injection)
    return '\n'.join(lines)

def get_program(ctx: moderngl.Context, path: str) -> moderngl.Program:
    shader_file_path = Path(__file__).parent / path
    with shader_file_path.open("r") as shader_file:
        shader_source = shader_file.read()
    program = ctx.program(
        vertex_shader=inject_string(shader_source, '#define VERTEX_SHADER'),
        fragment_shader=inject_string(shader_source, '#define FRAGMENT_SHADER')
    )
    return program

# NOTE(memben): having ctx as an argument is a workaround for moderngl_window's context management.
def obtain_point_ids(ctx: moderngl.Context, pcd: pointcloud.PointCloud, mvp: np.ndarray, width: int, height: int, debug=False) -> np.ndarray:
    """Given the point cloud and the MVP (4x4) matrix, return the numpy array of shape (width, height) 
    where each cell contains the id + 1 of the point that was rendered to that pixel. 
    Note that id = 0 means that no point was rendered. """
    
    def buffer_to_id(buffer: bytes, width: int, height: int) -> np.ndarray:
        # Note: we invert the y axis here
        dt = np.dtype(np.uint32)
        # Little endian
        dt = dt.newbyteorder('<')
        rgba = np.frombuffer(buffer, dtype=dt).reshape((height, width))
        return rgba
    
    program = get_program(ctx, 'shaders/point_id.glsl')

    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.BLEND)
    ctx.multisample = False

    program['mvp'].write(mvp.astype('f4').tobytes())
    # TODO(memben): Fix distorted point color for values > 1.0
    program['point_size'].value = 1.0

    fbo = ctx.framebuffer(ctx.renderbuffer((width, height)))
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0) # set background to black
    pcd.get_va_from(ctx, program).render(mode=moderngl.POINTS, vertices=pcd.points.shape[0])
    ctx.finish()
    if debug:
        # Alpha channel cannot be displayed
        Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show(title="Point ID Capture")
    buffer = fbo.read(components=4, alignment=1)
    return buffer_to_id(buffer, width, height)

def create_screen_image(source: moderngl.Framebuffer) -> Image:
    # Taken from the moderngl_window's screenshot function
    mode = 'RGB'
    alignment = 1
    image = Image.frombytes(
        mode,
        (
            source.viewport[2] - source.viewport[0],
            source.viewport[3] - source.viewport[1],
        ),
        source.read(viewport=source.viewport, alignment=alignment),
    )
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

def create_depth_image(ctx: moderngl.Context, pcd: pointcloud.PointCloud, mvp: np.ndarray, width: int, height: int, debug=False) -> Image:
    """Given the point cloud and the MVP (4x4) matrix, return the numpy array of shape (width, height) 
    where each cell contains the depth of the point that was rendered to that pixel. 
    Note that depth = 0 means that no point was rendered. """
    program = get_program(ctx, 'shaders/point_color.glsl')

    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.DEPTH_TEST)

    program['mvp'].write(mvp.astype('f4').tobytes())
    program['point_size'].value = pcd.point_size

    tex_depth = ctx.depth_texture((width, height))  # implicit -> dtype='f4', components=1
    fbo_depth = ctx.framebuffer(depth_attachment=tex_depth)
    fbo_depth.use()
    fbo_depth.clear(depth=1.0)
    pcd.get_va_from(ctx, program).render(mode=moderngl.POINTS, vertices=pcd.points.shape[0])
    ctx.finish()
    depth_from_dbo = np.frombuffer(tex_depth.read(), dtype=np.dtype('f4')).reshape((width, height)[::-1])
    depth_from_dbo = np.flip(depth_from_dbo, axis=0)
    return depth_utils.create_depth_image(depth_from_dbo, filter=True)

def test_obtain_point_ids():
    width, height = 512, 512
    n_points = width * height
    MVP = np.eye(4, dtype=np.float32)
    points = np.zeros((n_points, 3), dtype=np.float32)
    # Create test point cloud
    idx = 0
    for y in range(height):
        for x in range(width):
            points[idx] = [x, y, 0]
            idx += 1
    # NOTE(memben): Minor normalization error, making top row and left column black.
    pcd = pointcloud.PointCloud(points)
    ctx = moderngl.create_standalone_context()
    ids = obtain_point_ids(ctx, pcd, MVP, width, height, debug=True)
    print(ids)

def test_obtain_depth():
    width, height = 512, 512
    n_points = width * height
    MVP = np.eye(4, dtype=np.float32)
    ROTATION = np.pi / 4 # 45 degrees
    # around x axis
    ROTATION_M = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(ROTATION), -np.sin(ROTATION)],
        [0.0, np.sin(ROTATION), np.cos(ROTATION)]
    ], dtype=np.float32)
    points = np.zeros((n_points, 3), dtype=np.float32)
    # Tilded plane
    idx = 0
    for y in range(height):
        for x in range(width):
            points[idx] = [x, y, 0]
            idx += 1
    # Rotate the plane
    points = points @ ROTATION_M
    pcd = pointcloud.PointCloud(points)
    ctx = moderngl.create_standalone_context()
    depth_image = create_depth_image(ctx, pcd, MVP, width, height, debug=True)
    depth_image.show()

if __name__ == '__main__':
    test_obtain_point_ids()
    test_obtain_depth()