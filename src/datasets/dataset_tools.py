import torch
from torchvision import datasets, transforms


def get_mnist_data():
    normalize = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    
    mnist_train = datasets.MNIST("../input/", train=True, transform=normalize)
    mnist_test = datasets.MNIST("../input/", train=False, transform=normalize)
    return mnist_train, mnist_test


def get_data_splits(
        train_data, test_data, 
        train_size, 
        retain_size, forget_size, 
        val_size, test_size, 
        seed):    
    generator = torch.Generator().manual_seed(seed)
        
    train, val = torch.utils.data.random_split(
        train_data, 
        [train_size, val_size],
        generator=generator
    )
    
    retain, forget = torch.utils.data.random_split(
        train, 
        [retain_size, forget_size],
        generator=generator
    )
    
    assert len(train.indices) == train_size
    assert len(val.indices) == val_size
    assert len(test_data) == test_size

    assert len(retain.indices) == retain_size
    assert len(forget.indices) == forget_size
    
    return train, retain, forget, val, test_data


def get_dataloaders(
        train_data, test_data,
        train_size, 
        retain_size, forget_size, 
        val_size, test_size, 
        seed,
        batch_size, 
        num_workers):
    train, retain, forget, val, test = get_data_splits(
        train_data, test_data,
        train_size, 
        retain_size, forget_size, 
        val_size, test_size, 
        seed
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )

    retain_dataloader = torch.utils.data.DataLoader(
        retain, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )

    forget_dataloader = torch.utils.data.DataLoader(
        forget, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    val_dataloader = torch.utils.data.DataLoader(
        val, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    test_dataloader = torch.utils.data.DataLoader(
        test, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_dataloader, retain_dataloader, forget_dataloader, val_dataloader, test_dataloader
