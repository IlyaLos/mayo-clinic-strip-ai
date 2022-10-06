from torchvision import transforms


train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.ColorJitter(brightness=0.2, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.ColorJitter(brightness=0.2, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
