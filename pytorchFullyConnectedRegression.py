import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

class Pytorch_Regressor(pl.LightningModule):
    def __init__(self):
        super(Pytorch_Regressor, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(1, 5),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(5, 1),
            torch.nn.LeakyReLU()
        )

    def forward(self, x):

        return self.seq(x)

    def loss(self, y, y_gt):

        loss = torch.nn.MSELoss(reduction="sum")
        return loss(torch.squeeze(y), y_gt)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001) #torch.optim.Adam(self.parameters())#
        return optimizer

    def training_step(self, trainBatch):
        
        x, y = trainBatch
        loss = self.loss(self.forward(x[:, None]), y)
        return loss

class RegressorDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return self.x.shape[0]

class Pytorch_Regressor_DataModule(pl.LightningDataModule):

    def __init__(self, x, y):
        self.d = RegressorDataset(x, y)


    def train_dataloader(self):
        return DataLoader(self.d, self.d.x.shape[0])