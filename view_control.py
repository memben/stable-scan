
import moderngl_window
import numpy as np
from moderngl_window import activate_context, get_local_window_cls
from moderngl_window.timers.clock import Timer

from point_viewer import PointCloudViewer
from pointcloud import PointCloud


class ViewControl:
    """Controls a moderngl_window viewer"""
    def __init__(self, pcd: PointCloud, width:int, height: int, debug: bool):   
        self.pcd = pcd
        self.viewer = PointCloudViewer(title="StableScan", size=(width, height))
        self.viewer.add_geometry(pcd)

    def run(self):
        """Runs a custom render loop"""
        timer = Timer()
        timer.start()
        while not self.viewer.wnd.is_closing:
            self.viewer.step(timer)
        self.viewer.wnd.destroy()

