import moderngl
from moderngl_window.opengl.vao import VAO

def create_pc(points, colors) -> VAO:
    """Create a pointcloud VAO

    Args:
        points (numpy.ndarray): Array of points
        color (numpy.ndarray(3)): RGB color

    Returns:
        A :py:class:`moderngl_window.opengl.vao.VAO` instance
    """
    vao = VAO(mode=moderngl.POINTS)
    vbo = points.astype('f4').tobytes()
    vao.buffer(vbo, '3f', 'in_position')
    vbo = colors.astype('f4').tobytes()
    vao.buffer(vbo, '3f', 'in_color')
    return vao