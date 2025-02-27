import torch
from torch import optim
from torch import nn
from matplotlib import pyplot as plt
from RetinaScanDataset import make_train_test_split

ds_train, RetinaScanLoader_train, ds_test, RetinaScanLoader_test =  make_train_test_split()

class AE_linear(nn.Module):
    def __init__(self, imgsize, mainsize):
        super().__init__()
        self.imgsize = imgsize
        self.mainsize = mainsize
        self.encoder_hidden_layer = nn.Linear(in_features= 3*self.imgsize**2, out_features=self.mainsize)
        self.encoder_output_layer = nn.Linear(in_features=self.mainsize, out_features=self.mainsize)
        self.decoder_hidden_layer = nn.Linear(in_features=self.mainsize, out_features=self.mainsize)
        self.decoder_output_layer = nn.Linear(in_features=self.mainsize, out_features=3*self.imgsize**2)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.encoder_hidden_layer(x)
        x = torch.relu(x)
        x = self.encoder_output_layer(x)
        x = torch.relu(x)
        x = self.decoder_hidden_layer(x)
        x = torch.relu(x)
        x = self.decoder_output_layer(x)
        x = torch.relu(x)
        x = x.reshape(x.shape[0], 3, self.imgsize, self.imgsize)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ", device)
imsize = ds_train.imgsize
model = AE_linear(imgsize = 17, mainsize = 960).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
train_loader = RetinaScanLoader_train


epochs = 100
for epoch in range(epochs):
    i = 0
    for batch_features in train_loader:
        batch_features = batch_features.to(device)
        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features)
        train_loss.backward()
        optimizer.step()
        print(train_loss)
        i += 1
        if i % 2 == 0:
            plt.imshow(outputs.detach()[1, :].permute(1, 2, 0).cpu().numpy())
            plt.show()
            plt.imshow(batch_features.detach()[1, :].view(3, imsize, imsize).permute(1, 2, 0).cpu().numpy())
            plt.show()
