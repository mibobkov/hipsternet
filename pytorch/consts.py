import torch
# NUM_LAYERS = 8
BATCH_SIZE = 128
NUM_CLASSES = 10
IMAGE_DIMENSIONS = 32*32
CHANNELS=3
WIDTH = 256
# WIDTH = 256
CONV_WIDTH = 8
KERNEL_SIZE = 5
device = torch.device("cuda:0")
criterion = torch.nn.CrossEntropyLoss()