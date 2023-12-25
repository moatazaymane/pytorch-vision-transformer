import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn
import numpy


class VitDataset(Dataset):

    def __init__(self, images, targets, patch_size, num_classes):

        super().__init__()
        self.images = images
        self.targets = targets
        self.width = images.shape[1]
        self.height = images.shape[2]
        self.patch_size = patch_size
        self.num_classes = num_classes

    def __len__(self):

        return self.images.shape[0]

    def __getitem__(self, index):

        pixel_values = torch.tensor(self.images[index])
        pixel_values = pixel_values.transpose(0, -1)
        target = torch.tensor(self.targets[index])
        pixel_values = pixel_values.to(dtype=torch.float)
        target = target.to(dtype=torch.float)
        # pixel_values = pixel_values.unsqueeze(dim=0)
        pixel_values = fn.normalize(pixel_values, pixel_values.mean([1, 2]), pixel_values.std([1, 2]))

        # flattened patches: size (64, 16) cifar (4x4 patches)

        patches_list = []
        # avg_pixels = torch.mean(pixel_values, dim=0) # averaging data accross the channel dimension

        for i in range(self.height // self.patch_size):

            pa = pixel_values[:, i * self.patch_size:(i + 1) * self.patch_size, :]

            for j in range(self.width // self.patch_size):
                patch = pa[:, :, j * self.patch_size: (j + 1) * self.patch_size].numpy()
                patches_list.append(patch)

        item = {'pixel_values': pixel_values,
                'flattened_patches': torch.tensor(numpy.array(patches_list)).transpose(0, 1).flatten(-2, -1),
                'target': target}
        return item
