
from pathlib import Path
from moderngl_window.opengl.vao import VAO

import moderngl
import pcd_io
import moderngl_window as mglw
import numpy as np
import point_cloud_rendering_utils as pcru
from pointcloud import PointCloud
from base_viewer import CameraWindow



class PointCloudViewer(CameraWindow):
    COLOR = 0
    INDEX = 1
    DEPTH = 2

    title = "Point Cloud Viewer"
    resource_dir = (Path(__file__).parent / 'shaders').resolve()
    # lock curser to window

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = self.COLOR
        self.prog = self.load_program('point_color.glsl')
        self.fbo = None
        # hard coded for now
        self.pcd = pcd_io.read_pcd('../room-scan.las')
     
    def render(self, time: float, frametime: float):

        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        projection = self.camera.projection.matrix
        camera_matrix = self.camera.matrix
        mvp = projection * camera_matrix
        self.prog['mvp'].write(mvp)
        self.prog['point_size'].value = self.pcd.point_size
        self.pcd.get_vao().render(self.prog)
        
            
    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        if action != self.wnd.keys.ACTION_PRESS:
            return
        if key == self.wnd.keys.I:
            self.mode = self.INDEX
            mvp = self.camera.projection.matrix * self.camera.matrix
            pcru.obtain_point_ids(self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height, debug=True)
            self.prog = self.load_program('point_id.glsl')
        elif key == self.wnd.keys.E:
            self.mode = self.DEPTH
            mvp = self.camera.projection.matrix * self.camera.matrix
            depth_image = pcru.obtain_depth_image(self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height, debug=True)
            depth_image.show(title="Filtered Depth Image")
        elif key == self.wnd.keys.C:
            self.mode = self.COLOR
            self.prog = self.load_program('point_color.glsl')
        elif key == self.wnd.keys.R:
            mvp = self.camera.projection.matrix * self.camera.matrix
            ids = pcru.obtain_point_ids(self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height)
            depth_image = pcru.obtain_depth_image(self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height)
            
            
if __name__ == '__main__':
    mglw.run_window_config(PointCloudViewer)