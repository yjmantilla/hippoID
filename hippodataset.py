import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms

def flatten(l):
    return [item for sublist in l for item in sublist]

class HippoDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder='data\hippopotamus.v1-trainversion.coco', split='train',transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        """
        self.split = split.lower()

        assert self.split in {'train', 'test','valid'}

        self.data_folder = data_folder
        self.transform = transform

        # Read data files
        with open(os.path.join(data_folder, self.split, '_annotations.coco.json'), 'r') as j:
            self.json = json.load(j)
        self.images = [os.path.join(data_folder,self.split,x['file_name']) for x in self.json['images']]
        self.boxes = []
        self.labels = []
        for i in range(len(self.images)):
            this_image_boxes = [x['bbox'] for x in self.json['annotations'] if x['image_id']==i]
            def foo(x):
                return x-1
            this_image_labels = [foo(x['category_id']) for x in self.json['annotations'] if x['image_id']==i]

            self.boxes.append(this_image_boxes)
            self.labels.append(this_image_labels)
        assert len(self.images)==len(self.boxes)
        assert len(self.images)==len(self.labels)
        self.classes = set(flatten(self.labels))
        self.metadata = self.json['categories']
    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Read objects in this image (bounding boxes, labels, difficulties)
        boxes = torch.FloatTensor(self.boxes[i])  # (n_objects, 4)
        labels = torch.LongTensor(self.labels[i])  # (n_objects)

        # Apply transformations
        #image, boxes, labels = transform

        return image, boxes, labels

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels # tensor (N, 3, 300, 300), 3 lists of N tensors each
    
if __name__ =='__main__':
    daset = HippoDataset()
    daset[2]