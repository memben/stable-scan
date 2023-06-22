
from pathlib import Path
from moderngl_window.opengl.vao import VAO

import moderngl
import moderngl_window as mglw
import numpy as np
import pointcloud
from base_viewer import CameraWindow


class PointCloudViewer(CameraWindow):
    title = "Point Cloud Viewer"
    resource_dir = (Path(__file__).parent / 'programs').resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prog = self.load_program('point_color.glsl')
        # points = np.random.rand(1000, 3).astype(np.float32)
        # colors = np.random.rand(1000, 3).astype(np.float32)
        # self.points = pointcloud.create_pc(points, colors)
        self.points = pointcloud.read_pcd('../room-scan.las')

    def render(self, time: float, frametime: float):
        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        projection = self.camera.projection.matrix
        camera_matrix = self.camera.matrix
        mvp = projection * camera_matrix
        self.prog['mvp'].write(mvp)
        self.prog['point_size'].value = 5.0
        self.points.render(self.prog)
            
    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        if key == self.wnd.keys.SPACE and action == self.wnd.keys.ACTION_PRESS:
            self.prog = self.load_program('point_debugger.glsl')
            print("SPACE")




if __name__ == '__main__':
    mglw.run_window_config(PointCloudViewer)