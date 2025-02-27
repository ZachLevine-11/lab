import torch
from torch import nn as nn
from torch import optim
from RetinaScanDataset import make_train_test_split_retina
from matplotlib import pyplot as plt
from piqa import SSIM

class both(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size = 3, stride = 2)
        self.bn1 = nn.BatchNorm2d(9)
        self.conv1a = nn.Conv2d(in_channels=9, out_channels=27, kernel_size = 3, stride = 2)
        self.bn1a = nn.BatchNorm2d(27)
        self.conv1b = nn.Conv2d(in_channels=27, out_channels=32, kernel_size = 3, stride = 2)
        self.bn1b = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride = 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2a = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride = 2)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride = 1)
        self.bn2b = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride = 1, padding = "same")
        self.bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.ConvTranspose2d(in_channels=1, out_channels=64, kernel_size=2, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4a = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=1)
        self.bn4a = nn.BatchNorm2d(64)
        self.conv4b = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.bn4b = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(in_channels=32, out_channels=27, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(27)
        self.conv5a = nn.ConvTranspose2d(in_channels=27, out_channels=9, kernel_size=3, stride=2)
        self.bn5a = nn.BatchNorm2d(9)
        self.conv5b = nn.ConvTranspose2d(in_channels=9, out_channels=3, kernel_size=3, stride=2) ##to allign the two latent spaces
        self.bn5b = nn.BatchNorm2d(3)
        self.bn4ab = nn.BatchNorm2d(1)

    def forward(self, x = None, onlyEmbeddings = False, onlyDecode = False, z = None, latent_shape = None, orig_shape = None):
        if not onlyDecode:
            x = self.conv1(x)
            x = self.relu(x)
            x = self.bn1(x)
            x = self.conv1a(x)
            x = self.relu(x)
            x = self.bn1a(x)
            x = self.conv1b(x)
            x = self.relu(x)
            x = self.bn1b(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = self.bn2(x)
            x = self.conv2a(x)
            x = self.relu(x)
            x = self.bn2a(x)
            x = self.conv2b(x)
            x = self.relu(x)
            x = self.bn2b(x)
            x = self.conv3(x)
            x = self.relu(x)
            ##upsample to match the two latent dimensions
            x = nn.Upsample((latent_shape[0], latent_shape[1]), mode="bicubic")(x)
            x = self.bn3(x)
            x = x.reshape(x.shape[0], -1)
            if onlyEmbeddings:
                return x
            else:
                x = x.reshape(x.shape[0], 1, latent_shape[0], latent_shape[1])
        else:
            x = z.reshape(z.shape[0], 1,  latent_shape[0], latent_shape[1])
        x = self.bn4ab(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = self.conv4a(x)
        x = self.relu(x)
        x = self.bn4a(x)
        x = self.conv4b(x)
        x = self.relu(x)
        x = self.bn4b(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.conv5a(x)
        x = self.relu(x)
        x = self.bn5a(x)
        x = self.conv5b(x)
        x = self.bn5b(x)
        x = nn.Upsample((orig_shape[0], orig_shape[1]), mode="bicubic")(x)
        return x

def combinedLoss(x,y, device):
    return nn.MSELoss()(x,y)

def train_loop(steps, epochs, model, ds_train, train_loader, ds_test, test_loader, device, do_eval = False, lr = 1e-2, latent_shape = None, orig_shape = None):
    print("Using " + str(device))
    torch.cuda.empty_cache()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    i = 0
    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            if i == steps:
                return model
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features, latent_shape = latent_shape, orig_shape = orig_shape)
            train_loss = combinedLoss(outputs, batch_features, device)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
            i += 1
            print(loss/i, i)
            eval_loss_total = 0
            if do_eval and i % 10 == 0:
                j = 0
                with torch.no_grad():
                    for test_features in test_loader:
                        test_features = test_features.to(device)
                        outputs = model(test_features, latent_shape = latent_shape, orig_shape = orig_shape)
                        eval_loss = combinedLoss(outputs, test_features, device)
                        eval_loss_total += eval_loss.item()
                        if j % 2 == 0:
                            predicted = outputs.detach().cpu()[0, :, :].permute(1, 2, 0)
                            plt.imshow(predicted)
                            plt.show()
                            plt.imshow(test_features.detach().cpu()[0, :, :].permute(1, 2, 0))
                            plt.show()
                        j += 1
                print("Epoch: ", epoch)
                print("Average per-batch training loss: ", loss/i)
                print("Average per-batch evaluation loss: ", eval_loss_total/j)
    return model
