import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

class Pytorch_FullyConnected(pl.LightningModule):
    def __init__(self, isClassifier=False, numClasses=None):
        self.isClassifier = isClassifier

        super(Pytorch_FullyConnected, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1) if not self.isClassifier else torch.nn.Linear(10, numClasses), 
            torch.nn.ReLU() 
        )
        if(self.isClassifier):
            self.seq = torch.nn.Sequential(self.seq, torch.nn.LogSoftmax(dim=1))

    def forward(self, x):

        return self.seq(x)

    def loss(self, y, y_gt):
        if(not self.isClassifier):
            loss = torch.nn.MSELoss(reduction="sum")
        else:
            loss = torch.nn.NLLLoss()
        return loss(y, y_gt.long())

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.0001) #torch.optim.Adam(self.parameters())#
        return optimizer

    def training_step(self, trainBatch):
        
        x, y = trainBatch
        loss = self.loss(self.forward(x), y)
        return loss

class SimpleDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)

    def __getitem__(self, index):
        return (self.x[index, :], self.y[index])

    def __len__(self):
        return self.x.shape[0]

class Pytorch_Simple_DataModule(pl.LightningDataModule):

    def __init__(self, x, y):
        self.d = SimpleDataset(x, y)


    def train_dataloader(self):
        return DataLoader(self.d, 10)