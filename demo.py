import numpy as np
from PIL import Image
import point_id_renderer

def continues_points_to_id():
    width = 512
    height = 512
    n_points = width * height
    MVP = np.eye(4, dtype=np.float32)
    
    points = np.zeros((n_points, 3), dtype=np.float32)
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            i = (y-1) * width + (x-1)
            xn = x / width * 2 - 1
            yn = y / height * 2 - 1
            points[i] = (xn, yn, 0)
    ids = point_id_renderer.points_to_id(points, MVP, width, height, debug=True)
    return ids

if __name__ == '__main__':
    print(continues_points_to_id())