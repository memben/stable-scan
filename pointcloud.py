import moderngl
import laspy
import numpy as np
from moderngl_window.opengl.vao import VAO

def normalize(points):
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0) 
    ranges = max_coords - min_coords
    scaling_factor = 2.0 / np.max(ranges)
    normalized_points = (points - min_coords) * scaling_factor - 1.0
    return normalized_points

def create_pc(points, colors) -> VAO:
    vao = VAO(mode=moderngl.POINTS)
    vbo = points.astype('f4').tobytes()
    vao.buffer(vbo, '3f', 'in_position')
    vbo = colors.astype('f4').tobytes()
    vao.buffer(vbo, '3f', 'in_color')
    return vao

def extract_las(*filenames: str):
    point_data_list = []
    point_color_list = []
    if not filenames:
        raise ValueError('No files provided for extraction.')
    for filename in filenames:
        if not filename.endswith('.las'):
            raise ValueError('Only .las files are supported.')
        with laspy.open(filename) as fh:
            las = fh.read()
            point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
            point_color = np.stack([las.red, las.green, las.blue], axis=0).transpose((1, 0)) 
            point_data_list.append(point_data)
            point_color_list.append(point_color)
    return np.concatenate(point_data_list, axis=0), np.concatenate(point_color_list, axis=0)

def read_pcd(*filenames: str):
    '''
    Read .las files and return a point cloud
    '''
    point_data, point_color = extract_las(*filenames)
    point_data = normalize(point_data)
    point_color = point_color / (2**16 - 1)
    return create_pc(point_data, point_color)