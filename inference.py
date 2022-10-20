from model import TurbineModel
from utils import load_model, set_random_seed, mean_angle_loss, get_6D_representations
import torch
from PIL import Image
from utils import get_R_from_uv
import numpy as np
import open3d as o3d
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, Resize
import matplotlib.pyplot as plt

# Set random seeds
set_random_seed()

# Load model
binary = True
model = TurbineModel().eval()
device = torch.device("cpu")
load_model(model, device=device, binary=binary)

# Load test image
img = Image.open("test_image.png")
if binary:
    img = img.point(lambda p: 255 if p > 254 else 0)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transforms = Compose([
                    Resize(224),
                    ToTensor(),
                    Normalize(mean=mean, std=std)
        ])
img = transforms(img)
plt.imshow(img.permute(1, 2, 0).numpy())
plt.axis('off')
plt.savefig("test_image_binary.png", bbox_inches='tight')
# plt.imshow(img.permute(1, 2, 0).numpy())
# plt.show()

# Check metric
# angles = np.array([-2.2949084375876354, 1.7754964609885313, 2.147607186783767])
# labels = torch.Tensor(get_6D_representations(angles)).unsqueeze(0)

# Make prediction
mesh = o3d.io.read_triangle_mesh("turbine.obj")
prediction = model(img.unsqueeze(0))
# loss = custom_loss(prediction, labels)
# print(loss)
R = get_R_from_uv(u=prediction[:, :3], v=prediction[:, 3:]).detach().numpy()[0]
mesh_rotated = mesh.rotate(R, center=(0, 0, 0))
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=256, height=256)
vis.add_geometry(mesh_rotated)
img = vis.capture_screen_float_buffer(True)
pil_img = Image.fromarray((255*np.asarray(img)).astype('uint8'), 'RGB')
if binary:
    pil_img.save(f"test_image_prediction_binary.png", "PNG")
else:
    pil_img.save(f"test_image_prediction.png", "PNG")

