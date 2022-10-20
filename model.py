from torchvision.models import resnet50
import torch.nn as nn
import torch
from utils import set_random_seed


class PoseModel(nn.Module):
    def __init__(self, num_classes=6):
        super(PoseModel, self).__init__()
        self.num_classes = num_classes
        self.resnet = resnet50(pretrained=True)

        # Replace classification head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.resnet(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    set_random_seed()
    img = torch.randn(4, 3, 224, 224)

    model = PoseModel()
    out = model(img)
    print(out.shape)