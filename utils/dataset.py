import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]
    
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

    def __init__(self, root, ids, num_feats, target_patch_size=-1):
        super(GraphDataset, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.ids = ids
        self.num_feats = num_feats

        self.classdict = {'PS': 0, 'PL': 1}

        self._up_kwargs = {'mode': 'bilinear'}

    def __getitem__(self, index):
        sample = {}
        info = self.ids[index].replace('\n', '')
        file_name, label = info.split('\t')[0], info.split('\t')[1]

        file_path = self.root
        sample['label'] = self.classdict[label]
        sample['id'] = file_name

        feature_path = os.path.join(file_path, file_name, 'features.pt')

        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location=lambda storage, loc: storage)
        else:
            print(feature_path + ' not exists')
            features = torch.zeros(1, self.num_feats)

        adj_s_path = os.path.join(file_path, file_name, 'adj_s.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location=lambda storage, loc: storage)
        else:
            print(adj_s_path + ' not exists')
            adj_s = torch.ones(features.shape[0], features.shape[0])

        sample['image'] = features
        sample['adj_s'] = adj_s     #adj_s.to(torch.double)

        return sample


    def __len__(self):
        return len(self.ids)