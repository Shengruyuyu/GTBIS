import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as VF
from torchvision import transforms
from tqdm import tqdm
import argparse, os, glob
import numpy as np
from PIL import Image
import timm
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img


class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        img = img.resize((224, 224))
        sample = {'input': img}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']

        img = VF.to_tensor(img)
        img = VF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return {'input': img}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def save_coords(txt_file, csv_file_path):
    for path in csv_file_path:
        newIndex = path.rfind("-")
        if newIndex != -1:
            x_y = path[newIndex + 1:][:-5]
            x, y = x_y.split('_')[0], x_y.split('_')[1]
        txt_file.writelines(str(x) + '\t' + str(y) + '\n')
    txt_file.close()


def adj_matrix(csv_file_path):
    total = len(csv_file_path)
    adj_s = np.zeros((total, total))

    for i in range(total - 1):
        path_i = csv_file_path[i]
        newIndex = path_i.rfind("-")
        if newIndex != -1:
            x_y = path_i[newIndex + 1:][:-5]
            x_i, y_i = x_y.split('_')[0], x_y.split('_')[1]

        for j in range(i + 1, total):
            # sptial
            path_j = csv_file_path[j]
            jjnewIndex = path_j.rfind("-")
            if jjnewIndex != -1:
                jjx_y = path_j[jjnewIndex + 1:][:-5]
                x_j, y_j = jjx_y.split('_')[0], jjx_y.split('_')[1]
            if abs(int(x_i) - int(x_j)) <= 1 and abs(int(y_i) - int(y_j)) <= 1:
                adj_s[i][j] = 1
                adj_s[j][i] = 1

    adj_s = torch.from_numpy(adj_s)
    adj_s = adj_s.cuda()

    return adj_s


def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                     transform=Compose([
                                         ToTensor()
                                     ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def compute_feats(args, bags_list, i_classifier, save_path=None):
    num_bags = len(bags_list)
    for i in tqdm(range(0, num_bags)):
        feats_list = []
        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        file_name = bags_list[i].split('/')[-1]

        dataloader, bag_size = bag_dataset(args, csv_file_path)
        print('{} files to be processed: {}'.format(len(csv_file_path), file_name))

        if os.path.isdir(os.path.join(save_path, file_name)) or len(csv_file_path) < 1:
            print('alreday exists')
            continue
        with torch.no_grad():
            lenDataloader = len(dataloader)
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                patches = patches.type_as(next(i_classifier.parameters()))

                feats = i_classifier(patches).half().cpu().detach()
                # feats = feats.cpu().numpy()
                feats_list.extend(feats)
                print(f"{i}/{num_bags}, {iteration}/{lenDataloader}")

        os.makedirs(os.path.join(save_path, file_name), exist_ok=True)

        txt_file = open(os.path.join(save_path, file_name, 'c_idx.txt'), "w+")
        save_coords(txt_file, csv_file_path)
        # save node features
        output = torch.stack(feats_list, dim=0).cuda()
        torch.save(output, os.path.join(save_path,  file_name, 'features.pt'))
        # save adjacent matrix
        adj_s = adj_matrix(csv_file_path)
        torch.save(adj_s, os.path.join(save_path, file_name, 'adj_s.pt'))

        print('\r Computed: {}/{}'.format(i + 1, num_bags))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_feats', default=1024, type=int, help='Feature size')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size of dataloader')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--dataset', default=rf"results/patches", type=str,
                        help='path to patches')

    parser.add_argument('--checkpoint', default="vit_large_patch16_224.dinov2.uni_mass100k", type=str,
                        help='path to the pretrained weights')
    parser.add_argument('--output', default=rf"results/features", type=str, help='path to the output graph folder')
    args = parser.parse_args()

    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, "pytorch_model.bin"), map_location="cpu"), strict=True)

    os.makedirs(args.output, exist_ok=True)

    bagPaths = []
    for bagName in os.listdir(args.dataset):
        bagPath = args.dataset + '/' + bagName
        bagPaths.append(bagPath)

    compute_feats(args, bagPaths, model.cuda(), args.output)


if __name__ == '__main__':
    main()
