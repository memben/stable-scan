from pathlib import Path

import moderngl
import numpy as np

import depth_utils
import point_cloud_rendering_utils as pcru
import pointcloud
from base_viewer import CameraWindow


class PointCloudViewer(CameraWindow):
    resource_dir = (Path(__file__).parent / "shaders").resolve()

    def __init__(
        self,
        pcd: pointcloud.PointCloud,
        callbacks: dict[callable],
        debug: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # self.wnd.mouse_exclusivity = True
        self.pcd = pcd
        self.callbacks = callbacks
        self.debug = debug
        self.prog = self.load_program("point_color.glsl")
        self.fbo = None

    # TODO(memben): figure out why it does not perfrom like point cloud renderer
    def render(self, time: float, frametime: float):
        # activate the context

        self.ctx.clear(1.0, 1.0, 1.0, 1.0)
        self.ctx.enable_only(moderngl.PROGRAM_POINT_SIZE | moderngl.DEPTH_TEST)
        self.ctx.multisample = False

        mvp = self.camera.projection.matrix * self.camera.matrix
        self.prog["mvp"].write(mvp)
        self.prog["point_size"].value = self.pcd._point_size
        self.pcd.vao.render(self.prog)

    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        if action != self.wnd.keys.ACTION_PRESS:
            return

        mvp = self.camera.projection.matrix * self.camera.matrix
        _, __, width, height = self.ctx.viewport

        # Retexture the point cloud
        if key == self.wnd.keys.R:
            self.callbacks["retexture"](self.ctx, mvp)

        elif key == self.wnd.keys.O:
            self.callbacks["save"]()
        elif key == self.wnd.keys.L:
            self.callbacks["load"]()

        elif key == self.wnd.keys.X and self.debug:
            self.callbacks["retexture_only"]()
        elif key == self.wnd.keys.N and self.debug:
            self.prog = self.load_program("point_color.glsl")
            self.callbacks["reset"]()
        elif key == self.wnd.keys.B and self.debug:
            self.callbacks["blend"]()
        # Show the indices of the points
        elif key == self.wnd.keys.I and self.debug:
            pcru.obtain_point_ids(self.ctx, self.pcd, mvp, width, height, debug=True)
            self.prog = self.load_program("point_id.glsl")

        # Show the effect of the applied filters, red are the points that were removed
        elif key == self.wnd.keys.F and self.debug:
            params = (self.ctx, self.pcd, mvp, width, height)
            depth_image = pcru.create_depth_image(*params, filter=False)
            depth_image_filtered = pcru.create_depth_image(*params, filter=True)
            raw_ids = pcru.obtain_point_ids(*params)
            depth_image.show("Depth Image")
            depth_image_filtered.show("Depth Image Filtered")
            ids, ids_removed = depth_utils.filter_ids(
                raw_ids, depth_image_filtered, depth_image, debug=True
            )
            ids = pointcloud.flatten_and_filter(ids)
            ids_removed = pointcloud.flatten_and_filter(ids_removed)
            self.callbacks["flag"](ids_removed)
            self.callbacks["filter"](np.concatenate((ids, ids_removed)))

        elif key == self.wnd.keys.UP:
            self.pcd._point_size += 1.0
        elif key == self.wnd.keys.DOWN:
            self.pcd._point_size -= 1.0


if __name__ == "__main__":
    app = PointCloudViewer(size=(768, 512))
    app.run()
