import numpy as np
from dataset import get_cityscapes_loader
from evalLauncher import load_my_state_dict
import torch
from erfnet import ERFNet
from temperature_scaling3 import ModelWithTemperature
from torch.utils.data import DataLoader, Subset

NUM_CLASSES = 20

# *********************************************************************************************************************

def split_loader(loader, num_splits=5):
    dataset = loader.dataset
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)  # Shuffle the indices to ensure randomness

    split_size = dataset_size // num_splits
    split_loaders = []

    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_splits - 1 else dataset_size
        split_indices = indices[start_idx:end_idx]
        split_subset = Subset(dataset, split_indices)
        split_loader = DataLoader(split_subset, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers)
        split_loaders.append(split_loader)

    return split_loaders


def main():

    model = ERFNet(NUM_CLASSES)

    state_dict = torch.load("./trained_models/erfnet_pretrained.pth", map_location=lambda storage, loc: storage)

    loader = get_cityscapes_loader(datadir='./Dataset/Cityscapes', batch_size=10, subset='val')

    loaders = split_loader(loader, num_splits=10)

    # carica nel modello lo state dictionary creato
    model = load_my_state_dict(model, state_dict)

    model = ModelWithTemperature(model)

    for l in loaders:
        model.set_temperature(l)


if __name__ == '__main__':
    main()
