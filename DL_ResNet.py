# -*- coding: utf-8 -*-
"""ResNet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/131_HMPzy2h0ps1wnYuHTF_Gy_mpLAPzW
"""

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

if device == "cuda":
    num_workers = 4
else:
    num_workers = 1

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

if __name__ == "__main__":
    # load the image folder dataset
    train_dataset = torchvision.datasets.ImageFolder(
        root=f"{os.getcwd()}/spectrograms", transform=transform
    )
    test_dataset = torchvision.datasets.ImageFolder(
        root=f"{os.getcwd()}/spectrograms_val", transform=transform
    )

    # print(train_dataset.class_to_idx)

    batch_size = 64
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # from torchvision.utils import make_grid

    # for images, _ in train_loader:
    #     plt.figure(figsize=(16,8))
    #     plt.axis('off')
    #     plt.imshow(make_grid(images, nrow=8).permute((1, 2, 0)))
    #     break

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, downsample=None):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=stride, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
                nn.BatchNorm2d(out_channels),
            )
            self.downsample = downsample
            self.relu = nn.ReLU()
            self.out_channels = out_channels

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

    class Resnet(nn.Module):
        def __init__(self, block, layers, num_classes=13):
            super(Resnet, self).__init__()
            self.in_channels = 64
            self.conv = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self.make_layer(block, 64, layers[0], 1)
            self.layer2 = self.make_layer(block, 128, layers[1], 2)
            self.layer3 = self.make_layer(block, 256, layers[2], 2)
            self.layer4 = self.make_layer(block, 512, layers[3], 2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

        def make_layer(self, block, out_channels, blocks, stride=1):
            downsample = None
            if (stride != 1) or (self.in_channels != out_channels):
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            layers = []
            layers.append(block(self.in_channels, out_channels, stride, downsample))
            self.in_channels = out_channels
            for i in range(1, blocks):
                layers.append(block(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv(x)
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    model = Resnet(ResidualBlock, [3, 4, 6, 3]).to(device)
    next(model.parameters()).is_cuda

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    def train(model, dataloaders, criterion, optimizer, lr_scheduler, num_epochs=45):
        since = time.time()
        val_acc_history = []
        loss_history = []
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            print("-" * 10)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # print('input shape:', inputs.shape)
                    # print('label shape:', labels.shape)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(
                        phase == "train"
                    ):  # Forward. Track history if only in train

                        if (
                            phase == "train"
                        ):  # Backward + optimize only if in training phase
                            outputs = model(inputs)
                            # print('output shape:', outputs.shape)
                            loss = criterion(outputs, labels)

                            _, preds = torch.max(outputs, 1)
                            loss.backward()
                            optimizer.step()

                        if phase == "val":
                            outputs = model(inputs)
                            # print('output shape:', outputs.shape)
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                loss_history.append(epoch_loss)

                if phase == "val":
                    lr_scheduler.step(epoch_loss)

                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == "val":
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history, loss_history

    model, val_acc_history, loss_history = train(
        model,
        {"train": train_loader, "val": test_loader},
        criterion,
        optimizer,
        lr_scheduler,
    )

    plt.figure(figsize=(16, 8))
    val_acc_history_value = [i.item() for i in val_acc_history]
    plt.plot(val_acc_history_value, label="Validation Accuracy")
    plt.legend()
    plt.savefig("val_acc_history_RN.png")

    plt.figure(figsize=(16, 8))
    plt.plot(loss_history, label="Loss")
    plt.legend()
    plt.savefig("loss_history_RN.png")

    torch.save(model.state_dict(), "checkpoint_resnet_final.pth")

    def calculate_metrics(model, test_loader):
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print(len(model(inputs)))
                # print(len(labels))
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        return y_true, y_pred

    y_true, y_pred = calculate_metrics(model, test_loader)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes,
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig("conf_RN.png")
