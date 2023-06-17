
from pathlib import Path
from pyrr import Matrix44, Vector3

import moderngl
import moderngl_window as mglw
from moderngl_window.opengl.vao import VAO
import numpy as np
import pointcloud
from moderngl_window import geometry
from base_viewer import CameraWindow


class PointCloudViewer(CameraWindow):
    title = "Point Cloud Viewer"
    resource_dir = (Path(__file__).parent / 'programs').resolve()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prog = self.load_program('point_color.glsl')
        points = np.random.rand(1000, 3).astype(np.float32)
        colors = np.random.rand(1000, 3).astype(np.float32)
        self.points = pointcloud.create_pc(points, colors)

    def setup(self, pointcloud: VAO, shader: str='point_color.glsl'):
        self.prog = self.load_program(shader)
        self.pointcloud = pointcloud


    def render(self, time: float, frametime: float):
        self.ctx.enable_only(moderngl.CULL_FACE | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)

        rotation = Matrix44.from_eulers((time, time, time), dtype='f4')
        translation = Matrix44.from_translation((0.0, 0.0, -3.5), dtype='f4')
        modelview = translation * rotation
        # projection = self.camera.projection.matrix
        # camera_matrix = self.camera.matrix
        # mvp = projection * camera_matrix * modelview
        self.prog['m_proj'].write(self.camera.projection.matrix)
        self.prog['m_model'].write(modelview)
        self.prog['m_camera'].write(self.camera.matrix)

        self.points.render(self.prog)


if __name__ == '__main__':
    mglw.run_window_config(PointCloudViewer)