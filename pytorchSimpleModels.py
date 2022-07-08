import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from importlib_metadata import Pair

class Pytorch_FullyConnectedClassifier(pl.LightningModule):
    def __init__(self, layerDimensions : list[Pair[int]]):


        super(Pytorch_FullyConnectedClassifier, self).__init__()


        self.seq = torch.nn.Sequential(
            torch.nn.Linear(layerDimensions[0][0], layerDimensions[0][1]),
            torch.nn.ReLU())
        for i in range(1, len(layerDimensions)):
            self.seq = torch.nn.Sequential(self.seq, 
                torch.nn.Linear(layerDimensions[i][0], layerDimensions[i][1]),
                torch.nn.ReLU())
        
        self.seq = torch.nn.Sequential(self.seq, torch.nn.LogSoftmax(dim=1))

    def forward(self, x):
        
        return self.seq(x)

    def loss(self, y, y_gt):

        loss = torch.nn.NLLLoss()
        return loss(y, y_gt.long())

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters())
        optimizer = torch.optim.SGD(self.parameters(), lr=10)
        #optimizer = torch.optim.SGD
        return optimizer

    def training_step(self, trainBatch):
        
        x, y = trainBatch
        loss = self.loss(self.forward(x), y)
        return loss

class Pytorch_FullyConnectedRegressor(pl.LightningModule):
    def __init__(self, layerDimensions : list[Pair[int]], lr):


        super(Pytorch_FullyConnectedRegressor, self).__init__()


        self.seq = torch.nn.Sequential(
            torch.nn.Linear(layerDimensions[0][0], layerDimensions[0][1]),
            torch.nn.ReLU())
        for i in range(1, len(layerDimensions)):
            self.seq = torch.nn.Sequential(self.seq, 
                torch.nn.Linear(layerDimensions[i][0], layerDimensions[i][1]),
                torch.nn.ReLU())
        self.lr = lr
        
    def forward(self, x):
        
        return self.seq(x if len(x.shape) > 1 else x[:, None]).squeeze()

    def loss(self, y, y_gt):

        loss = torch.nn.MSELoss()
        return loss(y, y_gt)

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters())
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD
        return optimizer

    def training_step(self, trainBatch):
        
        x, y = trainBatch
        loss = self.loss(self.forward(x), y)
        return loss

class Pytorch_LinearRegression(pl.LightningModule):
    def __init__(self, inputDim, lr):
        super(Pytorch_LinearRegression, self).__init__()

        self.lr = lr
        self.lin = torch.nn.Linear(inputDim, 1)
        torch.nn.init.constant_(self.lin.weight, 0.0)
        
    def forward(self, x):
        
        return self.lin(x if len(x.shape) > 1 else x[:, None]).squeeze()

    def loss(self, y, y_gt):

        loss = torch.nn.MSELoss()
        return loss(y, y_gt)

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters())
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD
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
        return (self.x[index, ...], self.y[index])

    def __len__(self):
        return self.x.shape[0]

class Pytorch_Simple_DataModule(pl.LightningDataModule):

    def __init__(self, x, y, batchSize):
        self.d = SimpleDataset(x, y)
        self.batchSize = batchSize

    def train_dataloader(self):
        return DataLoader(self.d, self.batchSize)