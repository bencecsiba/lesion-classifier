import os
import torch
from torchvision import datasets, transforms
import tqdm
import math
import matplotlib.pyplot as plt
from torchsampler import ImbalancedDatasetSampler


def compute_mean_std():

    cache_file = "mean_std.pt"

    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)

        return d["mean"], d["std"]
    
    ds = datasets.ImageFolder(
        '/Users/bence/Documents/Personal Projects/Skin/Data/Images', transform=transforms.Compose([transforms.ToTensor()])
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=0
    )

    print(dl)

    mean = 0.0
    for images, _ in dl:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)

    print(mean)

    var = 0.0
    npix = 0
    for images, _ in dl:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()

    std = torch.sqrt(var / (npix / 3))

    # Cache results to avoid redoing the computation

    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std

def get_data_loaders(batch_size: int = 32, valid_size: float = 0.2, num_workers: int  = 1, **kwargs):

    data_loaders = {'train': None, 'valid': None, 'test': None}
    base_path = '../Data/Images'

    mean, std = compute_mean_std()
    print(f"Mean: {mean}, Std: {std}")

    # Define Transforms

    data_transforms = {
        "train": transforms.Compose(

            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandAugment(
                    num_ops = 2,
                    magnitude = 2,
                    interpolation = transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        ),

        "valid": transforms.Compose(

            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        ),

        "test": transforms.Compose(

            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        )
    }

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path + '/' + "train",
        transform = data_transforms['train']
        
    )

    valid_data = datasets.ImageFolder(
        base_path + '/' + "train",
        transform = data_transforms['valid']
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler  = torch.utils.data.SubsetRandomSampler(valid_idx)

    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        #sampler=train_sampler,
        sampler=ImbalancedDatasetSampler(train_data),
        num_workers=num_workers,
        **kwargs # For example: GPU optimizations.
    )

    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
    )

    # Create the test data loader
    test_data = datasets.ImageFolder(
        base_path + '/' + "test",        
        transform = data_transforms['test']   
    )

    data_loaders["test"] = torch.utils.data.DataLoader(        
        test_data,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = False
    )

    return data_loaders

def visualize_one_batch(data_loaders, max_n: int = 5):

    dataiter  = iter(data_loaders['train'])
    images, labels  = next(dataiter)

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # Get class names from the train data loader
    class_names  = data_loaders['train'].dataset.classes# YOUR CODE HERE

    # Convert from BGR (the format used by pytorch) to RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(class_names[labels[idx].item()])