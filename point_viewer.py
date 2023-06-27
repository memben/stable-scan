
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
        self.prog['point_size'].value = 2.0
        self.pcd.get_vao().render(self.prog)
        
            
    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        if action != self.wnd.keys.ACTION_PRESS:
            return
        if key == self.wnd.keys.I:
            self.mode = self.INDEX
            mvp = self.camera.projection.matrix * self.camera.matrix
            pcru.obtain_point_ids(self.ctx, self.pcd, mvp, self.wnd.width, self.wnd.height, debug=True)
            # self.prog = self.load_program('point_id.glsl')
        elif key == self.wnd.keys.R:
            self.mode = self.DEPTH
            if self.fbo is None:
                self.fbo = self.ctx.framebuffer(self.ctx.renderbuffer(self.wnd.size), 
                                                self.ctx.depth_renderbuffer(self.wnd.size, components=32))
            self.fbo.use()
            depth_data = self.fbo.read(components=1, alignment=1)
            depth_data = np.frombuffer(depth_data, dtype=np.uint8)  # Read as bytes
            depth_data = depth_data.reshape((self.wnd.height, self.wnd.width, 3))  # Reshape considering 3 bytes per depth value

            # Combine the bytes to get the 24 bit depth values
            depth_data_24bit = depth_data[:,:,0] + depth_data[:,:,1] * 256 + depth_data[:,:,2] * 256**2

            # Convert the 24 bit depth values to float
            depth_data_float = depth_data_24bit / (2**24 - 1)  # Normalizing
            print(depth_data_float)

            
if __name__ == '__main__':
    mglw.run_window_config(PointCloudViewer)