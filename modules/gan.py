import torch
from torch import nn, save, load


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    n_feats = 784
    out_feats = 1

    self.hidden0 = nn.Sequential(
      nn.Linear(n_feats, 1024),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )

    self.hidden1 = nn.Sequential(
      nn.Linear(1024, 512),
      nn.BatchNorm1d(512),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )

    self.hidden2 = nn.Sequential(
      nn.Linear(512, 256),
      nn.BatchNorm1d(256),
      nn.LeakyReLU(0.2),
      nn.Dropout(0.3)
    )

    self.out = nn.Sequential(
      nn.Linear(256, out_feats),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
    return x


class Generator(nn.Module):
  def __init__(self, n_feats):
    super(Generator, self).__init__()
    out_feats = 784

    self.hidden0 = nn.Sequential(
      nn.Linear(n_feats, 256),
      nn.BatchNorm1d(256),
      nn.LeakyReLU(0.2)
    )

    self.hidden1 = nn.Sequential(
      nn.Linear(256, 512),
      nn.BatchNorm1d(512),
      nn.LeakyReLU(0.2)
    )

    self.hidden2 = nn.Sequential(
      nn.Linear(512, 1024),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(0.2)
    )

    self.out = nn.Sequential(
      nn.Linear(1024, out_feats),
      nn.Tanh()
    )

  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    return self.out(x)

  def save(self, path):
    """
    This method saves the model at the given path

    Args: path (string): filepath of the model to be saved
    """
    save(self.state_dict(), path)


  def load(self, path):
    """
    This method loads a (saved) model at the given path

    Args: path (string): filepath of the saved model
    """

    self.load_state_dict(load(path))
