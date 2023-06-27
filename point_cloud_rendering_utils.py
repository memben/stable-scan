import moderngl
import numpy as np
import pointcloud
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
    
    shader_file_path = Path(__file__).parent / 'shaders' / 'point_id.glsl'
    with shader_file_path.open("r") as shader_file:
        shader_source = shader_file.read()

    program = ctx.program(
        vertex_shader=inject_string(shader_source, '#define VERTEX_SHADER'),
        fragment_shader=inject_string(shader_source, '#define FRAGMENT_SHADER')
    )
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.BLEND)
    ctx.multisample = False

    program['mvp'].write(mvp.astype('f4').tobytes())

    fbo = ctx.framebuffer(ctx.renderbuffer((width, height)))
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0) # set background to black
    pcd.get_va_from(ctx, program).render(mode=moderngl.POINTS, vertices=pcd.points.shape[0])
    ctx.finish()
    if debug:
        # Alpha channel cannot be displayed
        Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()
    buffer = fbo.read(components=4, alignment=1)
    return buffer_to_id(buffer, width, height)

if __name__ == '__main__':
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