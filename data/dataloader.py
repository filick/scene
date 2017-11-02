from torch.utils.data import DataLoader
import random


class MultiTransformWrapper(DataLoader):


    def __init__(self, dataloader, transforms, shuffle=True):
        self.dataloader = dataloader
        self.transforms = transforms
        self.shuffle = shuffle
        self.current_transform = -1

        if len(transforms) <= 0:
            raise ValueError("Transforms must be a list or tupple with size over zero.")


    def __iter__(self):
        if hasattr(self.dataloader.dataset, 'transform'):
            self.current_transform += 1
            self.current_transform %= len(self.transforms)
            if self.current_transform == 0 and self.shuffle:
                random.shuffle(self.transforms)
            self.dataloader.dataset.transform = self.transforms[self.current_transform]
            return iter(self.dataloader)
        else:
            raise RuntimeError("Cannot wrap the dataloader into multi-transform, because dataset has no transfrom attribute.")


    def __len__(self):
        return len(self.dataloader.batch_sampler)