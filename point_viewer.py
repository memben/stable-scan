
from pathlib import Path
from moderngl_window.opengl.vao import VAO

import moderngl
import pcd_io
import moderngl_window as mglw
import numpy as np
from pointcloud import PointCloud
from base_viewer import CameraWindow


class PointCloudViewer(CameraWindow):
    COLOR = 0
    INDEX = 1

    title = "Point Cloud Viewer"
    resource_dir = (Path(__file__).parent / 'programs').resolve()
    # lock curser to window

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = self.COLOR
        self.prog = self.load_program('point_color.glsl')
        # hard coded for now
        self.pcd = pcd_io.read_pcd('../room-scan.las')
     
    def render(self, time: float, frametime: float):
        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        projection = self.camera.projection.matrix
        camera_matrix = self.camera.matrix
        mvp = projection * camera_matrix
        self.prog['mvp'].write(mvp)
        self.prog['point_size'].value = 5.0
        self.pcd.vao.render(self.prog)
        
            
    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        if key == self.wnd.keys.I:
            self.mode = self.INDEX
            self.prog = self.load_program('point_id.glsl')
            
if __name__ == '__main__':
    mglw.run_window_config(PointCloudViewer)