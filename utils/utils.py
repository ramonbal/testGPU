import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.models import *

def get_data(slice=1, train=True):
#    full_dataset = torchvision.datasets.MNIST(root=".",
    full_dataset = torchvision.datasets.Flowers102(root=".",
                                              split="train" if train else "test", 
                                              transform=transforms.Compose([
                                                # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                # torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                                # torchvision.transforms.RandomAffine(degrees=30, shear=10),
                                                # Resize image and normalize pixels using the provided mean and standard deviation
                                                torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1.transforms(),
                                                # torchvision.transforms.Resize((256, 256)),
                                                # img2t
                                              ]),
                                              download=True)
    #  equiv to slicing with [::slice] 
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset

def collate_fn(batch):  
    return [torch.stack([x[0] for x in batch]),  torch.tensor([x[1] for x in batch])]

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=4,  collate_fn=collate_fn,
                                         )
    return loader


def make(config, device="cuda"):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    #model = ConvNet(config.kernels, config.classes).to(device)
    model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V2).to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.classes).to(device)
        
    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer