import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms
import cv2

import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

def collate(batch):
    image = [b['image'] for b in batch]  # w, h
    label = [b['label'] for b in batch]
    id = [b['id'] for b in batch]
    adj_s = [b['adj_s'] for b in batch]
    return {'image': image, 'label': label, 'id': id, 'adj_s': adj_s}



def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)

    trnsfrms_val = transforms.Compose(
                    [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = mean, std = std)
                    ]
                )

    return trnsfrms_val

class GraphDataset(data.Dataset):
    """input and label image dataset"""

    def __init__(self, WSI_path, num_feats, target_patch_size=-1):
        super(GraphDataset, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.WSI_path = WSI_path
        self.num_feats = num_feats

        self.classdict = {'PS': 0, 'PL': 1}
        self._up_kwargs = {'mode': 'bilinear'}

    def __getitem__(self, index):
        sample = {}
        sample['label'] = 0
        sample['id'] = self.WSI_path.split('/')[-1].split('_')[0]

        feature_path = rf'{self.WSI_path}/features.pt'

        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location=lambda storage, loc: storage)
        else:
            print(feature_path + ' not exists')
            features = torch.zeros(1, self.num_feats)

        adj_s_path = rf'{self.WSI_path}/adj_s.pt'

        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location=lambda storage, loc: storage)
        else:
            print(adj_s_path + ' not exists')
            adj_s = torch.ones(features.shape[0], features.shape[0])


        sample['image'] = features
        sample['adj_s'] = adj_s

        return sample


    def __len__(self):
        return 1