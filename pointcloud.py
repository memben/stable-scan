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
        self._colors = colors
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
        vbo_colors = ctx.buffer(self._colors.astype("f4").tobytes())
        va = ctx.vertex_array(
            program, [(vbo_points, "3f", "in_position"), (vbo_colors, "3f", "in_color")]
        )
        return va

    def _create_pc(self) -> VAO:
        vao = VAO(mode=moderngl.POINTS)
        vbo = self._points.astype("f4").tobytes()
        vao.buffer(vbo, "3f", "in_position")
        vbo = self._colors.astype("f4").tobytes()
        vao.buffer(vbo, "3f", "in_color")
        return vao

    def set_color(self, ids: np.array, colors: np.array) -> None:
        """Set the color of all points with ids in the ids set."""
        self._colors[ids] = colors
        self.vao = None

    def set_pcd(self, points: np.ndarray, colors: np.ndarray):
        self._points = normalize(points)
        self._colors = colors
        self.vao = None

    def filter(self, u_ids: np.array) -> None:
        """Discard all points except those with ids in the ids array."""
        self._points = self._points[u_ids]
        self._colors = self._colors[u_ids]
        self.vao = None
        print(self.vao)


class SDPointCloud:
    """A point cloud wrapper that provides additional retexturing functionality for StableScan."""

    def __init__(self, pcd: PointCloud, debug=False) -> None:
        self.original_points = pcd._points.copy()
        self.original_colors = pcd._colors.copy()
        self.pcd = pcd
        self.debug = debug
        self.retextured_points = set()

    @property
    def retextured_point_ids(self) -> np.ndarray:
        return np.array(list(self.retextured_points))

    def retexture(self, texture: Image, ids: np.ndarray) -> None:
        """Given a texture and a 2D array of ids, retexture the point cloud."""
        assert texture.width == ids.shape[1]
        assert texture.height == ids.shape[0]
        retexture_ids = []
        retexture_colors = []
        for y in range(texture.height):
            for x in range(texture.width):
                id = ids[y, x]
                if id == PointCloud.EMPTY:
                    continue
                if id in self.retextured_points:
                    continue
                color = texture.getpixel((x, y))
                color = np.array(color, dtype=np.float32) / 255.0
                retexture_ids.append(id)
                retexture_colors.append(color)        
        retexture_ids = np.array(retexture_ids)
        retexture_colors = np.array(retexture_colors)
        self.pcd.set_color(retexture_ids, retexture_colors)
        self.retextured_points.update(retexture_ids)

    def flag(self, ids: np.ndarray) -> None:
        """Flag all points with ids in the ids set."""
        self.pcd.set_color(ids, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    def filter(self, ids: np.ndarray) -> None:
        """Discard all points except those with ids in the ids array."""
        self.pcd.filter(ids)

    def save(self, filename: str) -> None:
        """Save the change on the point cloud to two .npy files."""
        ids = np.array(list(self.retextured_points))
        colors = self.pcd._colors[ids]
        np.save(filename + "_ids.npy", ids)
        np.save(filename + "_colors.npy", colors)

    def load(self, filename: str) -> None:
        """Load two .npy files and retexture the point cloud."""
        self.reset()
        ids = np.load(filename + "_ids.npy")
        colors = np.load(filename + "_colors.npy")
        self.pcd.set_color(ids, colors)
        self.retextured_points.update(ids)

    def reset(self) -> None:
        """Reset the point cloud to its original state."""
        self.pcd.set_pcd(self.original_points.copy(), self.original_colors.copy())
        self.retextured_points.clear()

    def mask_retextured(self, ids: np.ndarray) -> np.ndarray:
        """Given a 2D ids array, mask seen ids with 1, and unseen ids with 0."""
        mask = np.zeros_like(ids, dtype=np.uint8)
        debug = np.zeros((*ids.shape, 3), dtype=np.uint8)
        mask_count = 0
        for x in range(ids.shape[1]):
            for y in range(ids.shape[0]):
                id = ids[y, x]
                if id == PointCloud.EMPTY:
                    debug[y, x] = [0, 0, 0] 
                    mask[y, x] = 1
                    continue
                if id in self.retextured_points:
                    debug[y, x] = [0, 255, 0]
                    mask[y, x] = 0
                    mask_count += 1
                else: 
                    debug[y, x] = [255, 0, 0]
                    mask[y, x] = 1
        if self.debug:
            print(f"Keeping the color of {mask_count} points.")    
            Image.fromarray(debug, mode="RGB").show()
        return mask
        


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
