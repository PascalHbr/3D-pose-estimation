from model import PoseModel
from utils import load_model, set_random_seed
import torch
from PIL import Image
from utils import get_R_from_uv, get_R_from_angles
import numpy as np
import open3d as o3d
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import argparse


def main(arg):
    # Set random seeds
    set_random_seed()

    # Load model
    model = PoseModel().eval()
    device = torch.device("cpu")
    if arg.example:
        load_model(model, device=device, load_name="saved_model_example.pt")
    else:
        load_model(model, device=device, load_name=arg.load_name)

    # Load test image
    img = Image.open(arg.input)
    if not arg.rgb:
        img = img.point(lambda p: 255 if p > 254 else 0)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transforms = Compose([
                        Resize(224),
                        ToTensor(),
                        Normalize(mean=mean, std=std)
            ])
    img = transforms(img)

    # Make prediction
    mesh = o3d.io.read_triangle_mesh("WOLF.OBJ")
    prediction = model(img.unsqueeze(0))
    if arg.euler:
        R = get_R_from_angles(prediction[0].detach().numpy())
    else:
        R = get_R_from_uv(u=prediction[:, :3], v=prediction[:, 3:]).detach().numpy()[0]
    mesh_rotated = mesh.rotate(R, center=(0, 0, 0))
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=256, height=256)
    vis.add_geometry(mesh_rotated)
    img = vis.capture_screen_float_buffer(True)
    pil_img = Image.fromarray((255*np.asarray(img)).astype('uint8'), 'RGB')
    if arg.save_img:
        pil_img.save(f"prediction.png", "PNG")
    if arg.save_matrix:
        np.savetxt("prediction.txt", R)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="test_image.png", type=str)
    parser.add_argument('--load_name', default="saved_model.pt", type=str)
    parser.add_argument('--save_img', action='store_true', default=False)
    parser.add_argument('--save_matrix', action='store_true', default=False)
    parser.add_argument('--euler', action='store_true', default=False)
    parser.add_argument('--rgb', action='store_true', default=False)
    parser.add_argument('--example', action='store_true', default=False)

    arg = parser.parse_args()
    main(arg)

