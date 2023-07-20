from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

import depth_utils
import pointcloud

# POINT SIZE OPITMIZED FOR 512x512
POINT_SIZE = 1.5


def inject_definition(shader_code, injection):
    # "#version xxx" must be the first line
    lines = shader_code.split("\n")
    index = 0

    for i, line in enumerate(lines):
        if line.strip() != "":
            index = i + 1
            break

    lines.insert(index, injection)
    return "\n".join(lines)


def get_program(ctx: moderngl.Context, path: str) -> moderngl.Program:
    shader_file_path = Path(__file__).parent / path
    with shader_file_path.open("r") as shader_file:
        shader_source = shader_file.read()
    program = ctx.program(
        vertex_shader=inject_definition(shader_source, "#define VERTEX_SHADER"),
        fragment_shader=inject_definition(shader_source, "#define FRAGMENT_SHADER"),
    )
    return program


# NOTE(memben): having ctx as an argument is a workaround for moderngl_window's context management.
def obtain_point_ids(
    ctx: moderngl.Context,
    pcd: pointcloud.PointCloud,
    mvp: np.ndarray,
    width: int,
    height: int,
    debug=False,
) -> np.ndarray:
    """Given the point cloud and the MVP (4x4) matrix, return the numpy array of shape (width, height)
    where each cell contains the id of the point that was rendered to that pixel.
    Note that id = 2*32 - 1 means that no point was rendered."""

    def buffer_to_id(buffer: bytes, width: int, height: int) -> np.ndarray:
        dt = np.dtype(np.uint32)
        # Little endian
        dt = dt.newbyteorder("<")
        rgba = np.frombuffer(buffer, dtype=dt).reshape((height, width))
        return rgba

    program = get_program(ctx, "shaders/point_id.glsl")

    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.BLEND)
    ctx.multisample = False

    program["mvp"].write(mvp.astype("f4").tobytes())
    # NOTE(memben): distorted point color for values > 1.0 have been a problem in the past
    program["point_size"].value = POINT_SIZE

    fbo = ctx.framebuffer(ctx.renderbuffer((width, height)))
    fbo.use()
    fbo.clear(1.0, 1.0, 1.0, 1.0)  # white background
    pcd.get_va_from(ctx, program).render(
        mode=moderngl.POINTS, vertices=pcd._points.shape[0]
    )
    ctx.finish()
    buffer = fbo.read(components=4, alignment=1)
    ids = buffer_to_id(buffer, width, height)
    # in OpenGL the origin is at the bottom left corner
    ids = np.flip(ids, axis=0)
    if debug:
        # Alpha channel cannot be displayed
        img = Image.frombytes("RGB", fbo.size, fbo.read(), "raw", "RGB", 0, -1)
        img.show("Point IDs")
        img.save("point_ids.png")

        print(f"Captured {len(np.unique(ids))} unique ids.")
    return ids


def render_pointcloud(
    ctx: moderngl.Context,
    pcd: pointcloud.PointCloud,
    mvp: np.ndarray,
    width: int,
    height: int,
    debug=False,
) -> Image:
    """Render the point cloud and return the screen image."""

    program = get_program(ctx, "shaders/point_color.glsl")

    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.disable(moderngl.BLEND)
    ctx.multisample = False

    program["mvp"].write(mvp.astype("f4").tobytes())
    program["point_size"].value = POINT_SIZE
    fbo = ctx.framebuffer(ctx.renderbuffer((width, height)))
    fbo.use()
    fbo.clear(1.0, 1.0, 1.0, 1.0)  # white background
    pcd.get_va_from(ctx, program).render(
        mode=moderngl.POINTS, vertices=pcd._points.shape[0]
    )
    ctx.finish()
    img = Image.frombytes("RGB", fbo.size, fbo.read(), "raw", "RGB", 0, -1)

    if debug:
        img.show("Screen Image")

    return img


def create_screen_image(ctx: moderngl.Framebuffer, width: int, height: int) -> Image:
    # Taken from the moderngl_window's screenshot function
    source = ctx.screen
    mode = "RGB"
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
    # custom downscale
    # image = image.resize((width, height), Image.BILINEAR)
    return image


def create_depth_image(
    ctx: moderngl.Context,
    pcd: pointcloud.PointCloud,
    mvp: np.ndarray,
    width: int,
    height: int,
    filter: bool = True,
    debug=False,
) -> Image:
    """Given the point cloud and the MVP (4x4) matrix, return the numpy array of shape (width, height)
    where each cell contains the depth of the point that was rendered to that pixel.
    Note that depth = 0 means that no point was rendered."""
    program = get_program(ctx, "shaders/point_color.glsl")

    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.enable(moderngl.DEPTH_TEST)

    program["mvp"].write(mvp.astype("f4").tobytes())
    program["point_size"].value = POINT_SIZE

    tex_depth = ctx.depth_texture(
        (width, height)
    )  # implicit -> dtype='f4', components=1
    fbo_depth = ctx.framebuffer(depth_attachment=tex_depth)
    fbo_depth.use()
    fbo_depth.clear(depth=1.0)
    pcd.get_va_from(ctx, program).render(
        mode=moderngl.POINTS, vertices=pcd._points.shape[0]
    )
    ctx.finish()
    depth_from_dbo = np.frombuffer(tex_depth.read(), dtype=np.dtype("f4")).reshape(
        (width, height)[::-1]
    )
    depth_from_dbo = np.flip(depth_from_dbo, axis=0)
    return depth_utils.create_depth_image(depth_from_dbo, filter=filter)


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
    ROTATION = np.pi / 4  # 45 degrees
    # around x axis
    ROTATION_M = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(ROTATION), -np.sin(ROTATION)],
            [0.0, np.sin(ROTATION), np.cos(ROTATION)],
        ],
        dtype=np.float32,
    )
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


if __name__ == "__main__":
    test_obtain_point_ids()
    test_obtain_depth()
