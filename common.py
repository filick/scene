from data import transforms


#################  Trainsforms ##################

img_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def basic_train_transform(img_size):
    return transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.ToTensor(), 
                img_normalize])


def basic_validate_transform(img_size):
    return transforms.Compose([
                transforms.Resize(img_size),  
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                img_normalize])