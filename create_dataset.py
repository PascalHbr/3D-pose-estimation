import open3d as o3d
import numpy as np
import copy
from PIL import Image
import os
from tqdm import tqdm
import argparse


def get_R(angles):
    x, y, z = angles.astype(np.float32)
    
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R


def main(arg):
    # Create image directory
    if not os.path.exists("images"):
        os.makedirs("images")
    # Remove existing labels
    if os.path.exists("labels.txt"):
        os.remove("labels.txt")

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(arg.file, True)

    for i in tqdm(range(arg.n_samples)):
        # Make copy
        mesh_rotated = copy.deepcopy(mesh)
        
        # Make rotation matrix
        rot_x = np.random.uniform(-np.pi, np.pi)
        rot_y = np.random.uniform(-np.pi, np.pi)
        rot_z = np.random.uniform(-np.pi, np.pi)
        R = get_R(np.array([rot_x, rot_y, rot_z]))
        mesh_rotated.rotate(R, center=(0, 0, 0))

        # create image
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=256, height=256)
        vis.add_geometry(mesh_rotated)
        img = vis.capture_screen_float_buffer(True)
        pil_img = Image.fromarray((255*np.asarray(img)).astype('uint8'), 'RGB')

        # save image and update labels
        pil_img.save(f"images/img_{str(i).zfill(5)}.png", "PNG")
        with open("labels.txt", "a+") as f:
            f.write(f"{i}: {rot_x} {rot_y} {rot_z} \n")    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, type=str)
    parser.add_argument('--n_samples', default=5000, type=int)

    arg = parser.parse_args()
    main(arg)
