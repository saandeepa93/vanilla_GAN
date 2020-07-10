import torch
import matplotlib.pyplot as plt
import numpy as np
from sys import exit as e

from modules.gan import Generator
import modules.util as util

def plot_2d(generator):
  size_x = 12
  size_y = 10
  fig, axs = plt.subplots(size_x, size_y, figsize=(12, 9))
  m = 0.0
  for i in np.arange(size_x):
    n = 0.0
    for j in np.arange(size_y):
      print(m, n)
      fake_data = generator(torch.from_numpy(np.array([m, n])).type(torch.float)).view(1, 28, 28)
      axs[i, j].imshow(fake_data[0].detach(), cmap=plt.cm.gray)
      n+=0.1
    m+=0.1
  plt.show()

def test_gan(config):
  generator = Generator(config['hypers']['z'])
  generator.load(config['paths']['model'])
  noise = torch.randn(1, config['hypers']['z'])
  img = generator(noise).view(1, 28, 28)
  plt.imshow(img[0].detach())
  plt.show()