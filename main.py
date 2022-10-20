from tqdm import tqdm
import torch
from utils import set_random_seed, save_model, mean_angle_loss
from dataset import TurbineDataset
from model import PoseDataset
import torch.nn as nn
import torch.optim as optim
from config import parse_args


def main(arg):
    # Set random seeds
    set_random_seed()

    # Set device
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    # Load dataset
    dataset = TurbineDataset(euler=arg.euler, rgb=arg.rgb)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=arg.bn, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=arg.bn, shuffle=False)

    # Initialize model
    num_classes = 3 if arg.euler else 6
    model = PoseDataset(num_classes=num_classes).to(device)

    # Define hyperparameters
    lr = arg.lr
    epochs = arg.epochs

    # Definde loss function and optimizer
    criterion = mean_angle_loss
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    best_loss = 1e10

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Train loop
        model.train()
        running_loss = 0.0
        running_metric = 0.0

        for step, (images, labels, images_org) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels, euler=arg.euler)
            metric = mean_angle_loss(outputs, labels, euler=arg.euler)

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * images.size(0)
            running_metric += metric.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        epoch_metric = running_metric / len(train_dataset)
        print('Train Loss: {:.4f}'.format(epoch_loss))
        print('Train Metric: {:.4f}'.format(epoch_metric))

        # Test loop
        model.eval()
        running_loss = 0.0
        running_metric = 0.0

        for step, (images, labels, images_org) in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                images = images.to(device)
                labels = labels.to(device)

                # forward
                outputs = model(images)
                loss = criterion(outputs, labels, euler=arg.euler)
                metric = mean_angle_loss(outputs, labels, euler=arg.euler)

                # statistics
                running_loss += loss.item() * images.size(0)
                running_metric += metric.item() * images.size(0)

        epoch_loss = running_loss / len(test_dataset)
        epoch_metric = running_metric / len(test_dataset)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_model(model, arg.save_name)

        print('Test Loss: {:.4f}'.format(epoch_loss))
        print('Test Metric: {:.4f}'.format(epoch_metric))


if __name__ == "__main__":
    arg = parse_args()
    main(arg)