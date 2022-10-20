from torchvision.models import resnet50
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import set_random_seed


class TurbineModel(nn.Module):
    def __init__(self, num_classes=6):
        super(TurbineModel, self).__init__()
        self.num_classes = num_classes
        self.resnet = resnet50(pretrained=True)

        # Replace classification head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, eps=1e-8):
        x = self.relu(self.resnet(x))
        x = self.fc2(x)
        # if self.num_classes == 6:
        #     e1 = F.normalize(x[:, :3] + eps)
        #     e2 = F.normalize(x[:, 3:] - torch.einsum('bi,bi->b', e1, x[:, 3:]).unsqueeze(-1) * e1 + eps)
        #     x = torch.cat([e1, e2], dim=1)
        return x


if __name__ == "__main__":
    set_random_seed()
    img = torch.randn(4, 3, 224, 224)

    model = TurbineModel()
    out = model(img)
    print(out.shape)