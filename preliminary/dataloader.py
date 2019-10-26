# Thank the authors of pytorch-generative-model-collections and examples of pytorch.
# The github address is https://github.com/znxlwm/pytorch-generative-model-collections
# and https://github.com/pytorch/examples/blob/master/mnist/main.py respectively.
# Our code is widely adapted from their repositories.

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.utils.data as data
import numpy as np
import torch


class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)


def get_dataset(indices,raw_loader):
    images, labels = [], []
    for idx in indices:
        image, label = raw_loader[idx]
        images.append(image)
        labels.append(label)

    images = torch.stack(images, 0)  # shape [100, 1, 28, 28]
    labels = torch.from_numpy(np.array(labels, dtype=np.int64)).squeeze()  # torch.Size([100])
    return images, labels


def dataloader(dataset, input_size, batch_size, num_labels, split='train'):

    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    if dataset == 'mnist':
        training_set = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)

        indices = np.arange(len(training_set))
        np.random.shuffle(indices)
        mask = np.zeros(indices.shape[0], dtype=np.bool)
        labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
        for i in range(10):
            mask[np.where(labels == i)[0][: num_labels // 10]] = True
        labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
        print('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0])

        labeled_set = get_dataset(labeled_indices, training_set)
        unlabeled_set = get_dataset(unlabeled_indices, training_set)

        labeled_set = MyDataset(labeled_set[0], labeled_set[1])
        unlabeled_set = MyDataset(unlabeled_set[0], unlabeled_set[1])

        labeled_loader = DataLoader(labeled_set,
            batch_size=batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_set,
            batch_size=batch_size, shuffle=True)

        test_loader = DataLoader(
            datasets.MNIST('data/mnist', train=False, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)

    return labeled_loader, unlabeled_loader, test_loader