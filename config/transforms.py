from torchvision import transforms



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

composed_data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        normalize
    ]),
    'validation': transforms.Compose([
        transforms.Scale(256),  
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),  
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
}


def data_transforms(phase):
    return composed_data_transforms[phase]