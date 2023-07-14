from dataclasses import dataclass

import moderngl_window
import numpy as np
from moderngl_window import activate_context, get_local_window_cls
from moderngl_window.timers.clock import Timer
from PIL import Image

from point_viewer import PointCloudViewer
from pointcloud import SDPointCloud


@dataclass
class ScreenCapture:
    color_image: Image
    depth_image: Image
    ids: np.ndarray


class ViewControl:
    """Controls a moderngl_window viewer"""

    def __init__(
        self,
        sd_pcd: SDPointCloud,
        width: int,
        height: int,
        retexture_callback: callable,
        debug: bool,
    ):
        self.sd_pcd = sd_pcd
        debug_callbacks = {
            "flag": lambda ids: self.sd_pcd.flag(ids),
            "filter": lambda ids: self.sd_pcd.filter(ids),
            "load": lambda: print("load"),
            "exclusive_apply": lambda: print("exclusive_apply"),
            "save": lambda: print("save"),
            "reset": lambda: self.sd_pcd.reset(),
        }

        self.viewer = PointCloudViewer(
            self.sd_pcd.pcd,
            retexture_callback,
            debug_callbacks,
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
