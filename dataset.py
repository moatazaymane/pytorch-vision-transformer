import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn
import numpy as np
import torchvision.transforms.v2 as v2


def global_contrast_normalization(or_image: np.array, s: int, lmbda: int, epsilon: float) -> np.array:

      image = or_image.copy()
      image = torch.tensor(image).type(torch.float64)

      assert image.shape[-1] == 3

      h, w = image.shape[0], image.shape[1]


      mean, std = 0, 0

      for image_channel in range(3):

        for i in range(h):

          for j in range(w):

            mean += image[:, i,j,image_channel]

      mean/=3*h*w

      for image_channel in range(3):

        for i in range(h):

          for j in range(w):

            std += (image[i, j, image_channel] - mean)**2

      std /= 3*h*w

      for image_channel in range(3):

        for i in range(h):

          for j in range(w):

            image[i, j, image_channel] = s*(image[i, j, image_channel])/max(epsilon, lmbda + std)

      return torch.tensor(image)


def apply_transfom_dsn(training_data: np.array) -> torch.tensor:

    data = torch.tensor(training_data).transpose(0,-1)

    compose = v2.Compose([v2.Pad(4), v2.RandomVerticalFlip(p=torch.rand(1).numpy().tolist()[0]), v2.RandomCrop((32, 32))])

    return compose(data).transpose(0, -1)



class VitDataset(Dataset):

    def __init__(self, images, targets, patch_size, num_classes, preprocess = False, transform = False):

        super().__init__()
        self.images = images
        self.targets = targets
        self.width = images.shape[1]
        self.height = images.shape[2]
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):

        return self.images.shape[0]

    def __getitem__(self, index):

        pixel_values = global_contrast_normalization(torch.tensor(self.images[index]))  if self.preprocess else self.images[index]
        pixel_values = apply_transfom_dsn(pixel_values)

        pixel_values = pixel_values.transpose(0, -1)
        target = torch.tensor(self.targets[index])
        pixel_values = pixel_values.to(dtype=torch.float)
        target = target.to(dtype=torch.float)
        # pixel_values = pixel_values.unsqueeze(dim=0)
        #pixel_values = fn.normalize(pixel_values, pixel_values.mean([1, 2]), pixel_values.std([1, 2]))

        # flattened patches: size (64, 16) cifar (4x4 patches)

        patches_list = []
        # avg_pixels = torch.mean(pixel_values, dim=0) # averaging data accross the channel dimension

        for i in range(self.height // self.patch_size):

            pa = pixel_values[:, i * self.patch_size:(i + 1) * self.patch_size, :]

            for j in range(self.width // self.patch_size):
                patch = pa[:, :, j * self.patch_size: (j + 1) * self.patch_size].numpy()
                patches_list.append(patch)

        item = {'pixel_values': pixel_values,
                'flattened_patches': torch.tensor(np.array(patches_list)).transpose(0, 1).flatten(-2, -1),
                'target': target}
        return item
