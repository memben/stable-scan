import moderngl
import numpy as np
from moderngl_window.opengl.vao import VAO
from PIL import Image


class PointCloud:
    EMPTY = 2**32 - 1

    def __init__(
        self, points: np.ndarray, colors: np.ndarray = None, point_size: float = 1.0
    ) -> None:
        # to provide a uniform camera experience
        self._points = normalize(points)
        self._point_size = point_size
        if colors is None:
            colors = np.random.rand(points.shape[0], 3).astype(np.float32)
        self.colors = colors
        self.vao = None

    @property
    def point_size(self) -> float:
        return self._point_size

    def get_vao(self) -> VAO:
        if self.vao is None:
            self.vao = self._create_pc()
        return self.vao

    def get_va_from(
        self, ctx: moderngl.Context, program: moderngl.Program
    ) -> moderngl.VertexArray:
        """Create a vertex array from the points and colors."""
        vbo_points = ctx.buffer(self._points.astype("f4").tobytes())
        vbo_colors = ctx.buffer(self.colors.astype("f4").tobytes())
        va = ctx.vertex_array(
            program, [(vbo_points, "3f", "in_position"), (vbo_colors, "3f", "in_color")]
        )
        return va

    def _create_pc(self) -> VAO:
        vao = VAO(mode=moderngl.POINTS)
        vbo = self._points.astype("f4").tobytes()
        vao.buffer(vbo, "3f", "in_position")
        vbo = self.colors.astype("f4").tobytes()
        vao.buffer(vbo, "3f", "in_color")
        return vao

    def flag(self, ids: np.array) -> None:
        """Flag all points with ids in the ids set."""
        self.colors[ids] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.vao = None

    def retexture(self, texture: Image, ids: np.ndarray) -> None:
        """Given a texture and a 2D array of ids, retexture the point cloud."""
        for y in range(texture.height):
            for x in range(texture.width):
                color = texture.getpixel((x, y))
                color = np.array(color, dtype=np.float32) / 255.0
                id = ids[y, x]
                if id == PointCloud.EMPTY:
                    continue
                self.colors[id] = color
        self.vao = None

    def filter(self, u_ids: np.array) -> None:
        """Discard all points except those with ids in the ids array."""
        self._points = self._points[u_ids]
        self.colors = self.colors[u_ids]
        self.vao = None


# TODO(memben): Slighly shifts the point cloud one pixel to the bottom and right.
def normalize(points: np.ndarray) -> np.ndarray:
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    ranges = max_coords - min_coords
    scaling_factor = 2.0 / np.max(ranges)
    normalized_points = (points - min_coords) * scaling_factor - 1.0
    return normalized_points


def flatten_and_filter(ids: np.ndarray) -> np.ndarray:
    """Given a 2D array of ids, flatten it and remove all empty ids."""
    ids = ids.flatten()
    ids = ids[ids != PointCloud.EMPTY]
    return ids
