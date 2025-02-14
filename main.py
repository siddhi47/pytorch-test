import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import torch.optim as optim
from clearml import Task, OutputModel
from argparse import ArgumentParser
import hydra


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@hydra.main(config_path=".", config_name="hydra_config.yaml")
def main(cfg):
    EPOCH = cfg.model.epoch
    task = Task.init(
        project_name="examples",
        task_name="CIFAR10 PyTorch",
        output_uri="s3://cifar/models/",
    )
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = cfg.model.batch_size

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    logger = Task.current_task().get_logger()
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=cfg.model.lr, momentum=cfg.model.momentum
    )

    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                logger.report_scalar("loss", "train", running_loss / 2000, iteration=i)
                running_loss = 0.0

    predictions = []
    actuals = []
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted)
        actuals.extend(labels)

    confusion_matrix = sklearn.metrics.confusion_matrix(actuals, predictions)
    logger.report_matrix("confusion matrix", "test", confusion_matrix, iteration=0)

    PATH = "./cifar_net.pth"
    torch.save(net.state_dict(), PATH)

    output_model = OutputModel(
        task=task,
        framework="pytorch",
    )
    output_model.set_upload_destination("s3://cifar/models/")


if __name__ == "__main__":
    main()
