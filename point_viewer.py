from pathlib import Path

import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image

import depth_utils
import gen_control
import pcd_io
import point_cloud_rendering_utils as pcru
import pointcloud
from base_viewer import CameraWindow


class PointCloudViewer(CameraWindow):
    resource_dir = (Path(__file__).parent / "shaders").resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.wnd.mouse_exclusivity = True
        self.prog = self.load_program("point_color.glsl")
        self.fbo = None
        # hard coded for now
        self.pcd = pcd_io.read_pcd("../room-scan.las")

    def add_geometry(self, pcd: pointcloud.PointCloud):
        self.pcd = pcd

    def render(self, time: float, frametime: float):
        self.ctx.enable_only(
            moderngl.CULL_FACE | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE
        )
        projection = self.camera.projection.matrix
        camera_matrix = self.camera.matrix
        mvp = projection * camera_matrix
        self.prog["mvp"].write(mvp)
        self.prog["point_size"].value = self.pcd._point_size
        self.pcd.get_vao().render(self.prog)

    def _process_screen(self):
        mvp = self.camera.projection.matrix * self.camera.matrix
        ids = pcru.obtain_point_ids(
            self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height
        )
        screen_image = pcru.create_screen_image(
            self.ctx.screen, self.wnd.width, self.wnd.height
        )
        depth_image = pcru.create_depth_image(
            self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height
        )
        depth_image_unfiltered = pcru.create_depth_image(
            self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height, filter=False
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
            ) = self._process_screen()
            ids, _ = depth_utils.filter_ids(ids, depth_image, depth_image_unfiltered)
            result = gen_control.generate_img(screen_image, depth_image)
            if result is None:
                return
            self.pcd.retexture(result, ids)
            result.save("result.png")
            np.save("ids.npy", ids)
        # Show the indices of the points
        elif key == self.wnd.keys.I:
            mvp = self.camera.projection.matrix * self.camera.matrix
            pcru.obtain_point_ids(
                self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height, debug=True
            )
            self.prog = self.load_program("point_id.glsl")
        # Show the depth image
        elif key == self.wnd.keys.E:
            mvp = self.camera.projection.matrix * self.camera.matrix
            depth_image = pcru.create_depth_image(
                self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height
            )
            depth_image_unfiltered = pcru.create_depth_image(
                self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height, filter=False
            )
            depth_image.show(title="Filtered Depth Image")
            depth_image_unfiltered.show(title="Unfiltered Depth Image")
        # Show the effect of the applied filters, red are the points that were removed
        elif key == self.wnd.keys.F:
            (
                ids,
                screen_image,
                depth_image,
                depth_image_unfiltered,
            ) = self._process_screen()
            ids, ids_removed = depth_utils.filter_ids(
                ids, depth_image, depth_image_unfiltered, debug=True
            )
            ids = pointcloud.flatten_and_filter(ids)
            ids_removed = pointcloud.flatten_and_filter(ids_removed)
            self.pcd.flag(ids_removed)
            self.pcd.filter(np.concatenate((ids, ids_removed)))
        # Show the texture applied to the point cloud
        elif key == self.wnd.keys.M:
            result = Image.open("result.png")
            ids = np.load("ids.npy")
            self.pcd.retexture(result, ids)
        # Show the texture applied to the point cloud without any other points
        elif key == self.wnd.keys.X:
            result = Image.open("result.png")
            ids = np.load("ids.npy")
            self.pcd.retexture(result, ids)
            ids = pointcloud.flatten_and_filter(ids)
            self.pcd.filter(ids)
        elif key == self.wnd.keys.UP:
            self.pcd._point_size += 1.0
        elif key == self.wnd.keys.DOWN:
            self.pcd._point_size -= 1.0


if __name__ == "__main__":
    app = PointCloudViewer(size=(768, 512))
    app.run()
