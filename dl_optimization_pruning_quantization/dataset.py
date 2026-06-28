import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# Default selection of 5 classes (animals, which are highly confused)
SELECTED_CLASSES = ['bird', 'cat', 'deer', 'dog', 'frog']
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create mapping from original label to new label (0-4)
ORIGINAL_IDX_MAP = {CIFAR10_CLASSES.index(cls): idx for idx, cls in enumerate(SELECTED_CLASSES)}

class CIFAR5(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.cifar10 = datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
        # Filter indices to only include the 5 selected classes
        self.indices = [i for i, (_, label) in enumerate(self.cifar10) if label in ORIGINAL_IDX_MAP]
        
    def __getitem__(self, index):
        img, label = self.cifar10[self.indices[index]]
        # Map original label to 0-4
        new_label = ORIGINAL_IDX_MAP[label]
        return img, new_label
        
    def __len__(self):
        return len(self.indices)

def get_dataloaders(data_dir='./data', batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR5(root=data_dir, train=True, transform=transform_train, download=True)
    test_dataset = CIFAR5(root=data_dir, train=False, transform=transform_test, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
