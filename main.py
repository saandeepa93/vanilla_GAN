import click
from sys import exit as e


from modules.train import train_gan
from modules.test import test_gan
import modules.util as util


config = util.getconfig()

@click.command()
def train():
  train_gan(config)

@click.command()
def test():
  test_gan(config)

@click.group()
def main():
  pass

if __name__ == '__main__':
  main.add_command(train)
  main.add_command(test)
  main()