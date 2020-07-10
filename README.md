# Vanilla GAN for MNIST dataset

## Installation

Install `pip ` packages

```bash
pip install -r requirements.txt
```

## Models

The trained model is present in the folder `./models/final/generator.pt`.

## Dataset

Create a directory called `dataset` in the root folder. Pytorch dataset class will download the MNIST dataset while training

## Training
```bash
python main.py train
```

## Testing
```bash
python main.py test
```

## Configurations
Any config setting can be found in `./config.yaml` file.