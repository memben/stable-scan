from collections import defaultdict
from dataclasses import dataclass

import moderngl
import numpy as np
from moderngl_window.timers.clock import Timer
from PIL import Image

import depth_utils
import point_cloud_rendering_utils as pcru
from point_viewer import PointCloudViewer
from pointcloud import SDPointCloud


@dataclass
class ScreenCapture:
    color_image: Image
    depth_image: Image
    width: int
    height: int
    ids: np.ndarray


class ViewControl:
    """Controls a moderngl_window viewer"""

    def __init__(
        self,
        sd_pcd: SDPointCloud,
        width: int,
        height: int,
        retexture_callback: callable,
        retexture_width: int,
        retexture_height: int,
        debug: bool = False,
    ):
        self.sd_pcd = sd_pcd
        self.debug = debug

        callbacks = defaultdict(lambda: lambda: print("Action not defined"))

        callbacks["retexture"] = lambda ctx, mvp: retexture_callback(
            self.create_screen_capture(
                ctx, mvp, retexture_width, retexture_height, debug
            )
        )
        callbacks["load"] = lambda: self.sd_pcd.load("retexture")
        callbacks["save"] = lambda: self.sd_pcd.save("retexture")

        if debug:
            callbacks["reset"] = lambda: self.sd_pcd.reset()
            callbacks["retexture_only"] = lambda: self.sd_pcd.filter(
                self.sd_pcd.retextured_point_ids
            )
            callbacks["blend"] = lambda: self.sd_pcd.flag(
                self.sd_pcd.retextured_point_ids
            )
            callbacks["flag"] = lambda ids: self.sd_pcd.flag(ids)
            callbacks["filter"] = lambda ids: self.sd_pcd.filter(ids)

        self.viewer = PointCloudViewer(
            self.sd_pcd.pcd,
            callbacks,
            title="StableScan",
            size=(width, height),
            debug=debug,
        )

    def run(self):
        """Runs a custom render loop"""
        timer = Timer()
        timer.start()
        while not self.viewer.wnd.is_closing:
            self.viewer.step(timer)
        self.viewer.wnd.destroy()

    def create_screen_capture(
        self,
        ctx: moderngl.Framebuffer,
        mvp: np.ndarray,
        width: int,
        height: int,
        debug: bool = False,
    ) -> ScreenCapture:
        params = (ctx, self.sd_pcd.pcd, mvp, width, height)
        screen_image = pcru.render_pointcloud(*params, debug=debug)
        depth_image = pcru.create_depth_image(*params, filter=False, debug=debug)
        depth_image_filtered = pcru.create_depth_image(
            *params, filter=True, debug=debug
        )
        raw_ids = pcru.obtain_point_ids(*params, debug=debug)
        ids, _ = depth_utils.filter_ids(
            raw_ids, depth_image_filtered, depth_image, debug=debug
        )
        return ScreenCapture(screen_image, depth_image_filtered, width, height, ids)
