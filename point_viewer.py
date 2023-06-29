from pathlib import Path

import moderngl
import moderngl_window as mglw
import numpy as np
from moderngl_window.opengl.vao import VAO
from PIL import Image

import depth_utils
import gen_control
import pcd_io
import point_cloud_rendering_utils as pcru
from base_viewer import CameraWindow
from pointcloud import PointCloud


class PointCloudViewer(CameraWindow):
    COLOR = 0
    INDEX = 1
    DEPTH = 2

    title = "Point Cloud Viewer"
    window_size = (512, 512)
    aspect_ratio = 1.0
    resource_dir = (Path(__file__).parent / "shaders").resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = self.COLOR
        self.prog = self.load_program("point_color.glsl")
        self.fbo = None
        # hard coded for now
        self.pcd = pcd_io.read_pcd("../room-scan.las")

    def render(self, time: float, frametime: float):
        self.ctx.enable_only(
            moderngl.CULL_FACE | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE
        )
        projection = self.camera.projection.matrix
        camera_matrix = self.camera.matrix
        mvp = projection * camera_matrix
        self.prog["mvp"].write(mvp)
        self.prog["point_size"].value = self.pcd.point_size
        self.pcd.get_vao().render(self.prog)

    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        if action != self.wnd.keys.ACTION_PRESS:
            return
        if key == self.wnd.keys.I:
            self.mode = self.INDEX
            mvp = self.camera.projection.matrix * self.camera.matrix
            pcru.obtain_point_ids(
                self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height, debug=True
            )
            self.prog = self.load_program("point_id.glsl")
        elif key == self.wnd.keys.E:
            self.mode = self.DEPTH
            mvp = self.camera.projection.matrix * self.camera.matrix
            depth_image = pcru.create_depth_image(
                self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height, debug=True
            )
            depth_image.show(title="Filtered Depth Image")
        elif key == self.wnd.keys.C:
            self.mode = self.COLOR
            self.prog = self.load_program("point_color.glsl")
        elif key == self.wnd.keys.F:
            mvp = self.camera.projection.matrix * self.camera.matrix
            ids = pcru.obtain_point_ids(
                self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height
            )
            depth_image = pcru.create_depth_image(
                self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height
            )
            depth_image_unfiltered = pcru.create_depth_image(
                self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height, filter=False
            )
            ids, ids_removed = depth_utils.filter_ids(
                ids, depth_image, depth_image_unfiltered, debug=True
            )
            u_ids = np.unique(ids.flatten())
            u_ids = u_ids[u_ids != PointCloud.EMPTY]
            self.pcd.flag(ids_removed)
            self.pcd.filter(set(u_ids) | ids_removed)
        elif key == self.wnd.keys.R:
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
            ids, _ = depth_utils.filter_ids(ids, depth_image, depth_image_unfiltered)
            result = gen_control.generate_img(screen_image, depth_image)
            if result is None:
                return
            self.pcd.retexture(result, ids)
            result.save("result.png")
            np.save("ids.npy", ids)
        elif key == self.wnd.keys.M:
            result = Image.open("result.png")
            ids = np.load("ids.npy")
            self.pcd.retexture(result, ids)
        elif key == self.wnd.keys.X:
            result = Image.open("result.png")
            ids = np.load("ids.npy")
            self.pcd.exclusive_retexture(result, ids)
        elif key == self.wnd.keys.UP:
            self.pcd.point_size += 1.0
        elif key == self.wnd.keys.DOWN:
            self.pcd.point_size -= 1.0


if __name__ == "__main__":
    mglw.run_window_config(PointCloudViewer)
