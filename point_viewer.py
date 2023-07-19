from pathlib import Path

import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image

import depth_utils
import point_cloud_rendering_utils as pcru
import pointcloud
from base_viewer import CameraWindow


class PointCloudViewer(CameraWindow):
    resource_dir = (Path(__file__).parent / "shaders").resolve()

    def __init__(
        self,
        pcd: pointcloud.PointCloud,
        retexture_callback: callable,
        debug_callbacks: dict[callable],
        debug: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # self.wnd.mouse_exclusivity = True
        self.pcd = pcd
        self._retexture_callback = retexture_callback
        self._debug_callbacks = debug_callbacks
        self.debug = debug
        self.prog = self.load_program("point_color.glsl")
        self.fbo = None

        self._buffer_width = self.ctx.viewport[2]
        self._buffer_height = self.ctx.viewport[3]

    # TODO(memben): figure out why it does not perfrom like point cloud renderer
    def render(self, time: float, frametime: float):
        self._buffer_width = self.ctx.viewport[2]
        self._buffer_height = self.ctx.viewport[3]

        self.ctx.clear(1.0, 1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.multisample = False

        projection = self.camera.projection.matrix
        camera_matrix = self.camera.matrix
        mvp = projection * camera_matrix
        self.prog["mvp"].write(mvp)
        self.prog["point_size"].value = self.pcd._point_size
        self.pcd.get_vao().render(self.prog)

    def _process_screen(self, debug=False):
        width = self._buffer_width
        height = self._buffer_height

        mvp = self.camera.projection.matrix * self.camera.matrix
        ids = pcru.obtain_point_ids(self.ctx, self.pcd, mvp, width, height, debug=debug)
        screen_image = pcru.create_screen_image(self.ctx, width, height)
        screen_rend_image = pcru.render_screen_image(
            self.ctx, self.pcd, mvp, width, height, debug=debug
        )

        depth_image = pcru.create_depth_image(
            self.ctx, self.pcd, mvp, width, height, debug=debug
        )
        depth_image_unfiltered = pcru.create_depth_image(
            self.ctx,
            self.pcd,
            mvp,
            width,
            height,
            filter=False,
            debug=debug,
        )
        return ids, screen_image, depth_image, depth_image_unfiltered

    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        if action != self.wnd.keys.ACTION_PRESS:
            return

        # Retexture the point cloud
        if key == self.wnd.keys.R:
            (
                ids,
                screen_image,
                depth_image,
                depth_image_unfiltered,
            ) = self._process_screen(debug=self.debug)

            ids, _ = depth_utils.filter_ids(ids, depth_image, depth_image_unfiltered)

            from view_control import ScreenCapture

            width, height = self._buffer_width, self._buffer_height
            self._retexture_callback(
                ScreenCapture(screen_image, depth_image, width, height, ids)
            )

        # Show the indices of the points
        elif key == self.wnd.keys.I and self.debug:
            mvp = self.camera.projection.matrix * self.camera.matrix
            pcru.obtain_point_ids(self.ctx, self.pcd, mvp, width, height, debug=True)
            self.prog = self.load_program("point_id.glsl")

        # Show the effect of the applied filters, red are the points that were removed
        elif key == self.wnd.keys.F and self.debug:
            (
                ids,
                screen_image,
                depth_image,
                depth_image_unfiltered,
            ) = self._process_screen()
            depth_image.show(title="Filtered Depth Image")
            depth_image_unfiltered.show(title="Unfiltered Depth Image")
            ids, ids_removed = depth_utils.filter_ids(
                ids, depth_image, depth_image_unfiltered, debug=True
            )
            ids = pointcloud.flatten_and_filter(ids)
            ids_removed = pointcloud.flatten_and_filter(ids_removed)
            self._debug_callbacks["flag"](ids_removed)
            self._debug_callbacks["filter"](np.concatenate((ids, ids_removed)))
        elif key == self.wnd.keys.O:
            self._debug_callbacks["save"]()
        elif key == self.wnd.keys.L:
            self._debug_callbacks["load"]()
        elif key == self.wnd.keys.X:
            self._debug_callbacks["exclusive_apply"]()
        elif key == self.wnd.keys.N:
            self.prog = self.load_program("point_color.glsl")
            self._debug_callbacks["reset"]()
        elif key == self.wnd.keys.B:
            self._debug_callbacks["blend"]()
        elif key == self.wnd.keys.UP:
            self.pcd._point_size += 1.0
        elif key == self.wnd.keys.DOWN:
            self.pcd._point_size -= 1.0


if __name__ == "__main__":
    app = PointCloudViewer(size=(768, 512))
    app.run()
