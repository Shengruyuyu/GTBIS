import numpy as np
import os
dll_path = os.path.abspath("./openslide-win64-20171122/bin")
os.add_dll_directory(dll_path)
import openslide
import cv2
import pandas as pd
import torch
import networkx as nx
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from models.GraphTransformer_1node import Classifier
from dataset_heatmap import GraphDataset, collate
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib.pyplot as plt

import os
from pickletools import uint8

import torch
import numpy as np
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import openslide
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from typing import List
# Please set your Hugging Face API token


def create_overlay_image(original, result, patch_size, alpha=0.3):
    # Resize result to match the original image size
    original = cv2.resize(original, (patch_size, patch_size))
    result_resized = cv2.resize(result, (original.shape[1], original.shape[0]))
    overlay = (alpha * original + (1 - alpha) * result_resized * 255).astype(np.uint8)
    return overlay


def load_and_preprocess_image(image_path: str) -> Image.Image:
    with open(image_path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    return img

def process_images(images: List[Image.Image], background_threshold: float = 0.5, larger_pca_as_fg: bool = True) -> List[np.ndarray]:
    imgs_tensor = torch.stack([transform(img).to(device) for img in images])

    with torch.no_grad():
        intermediate_features = model.forward_intermediates(imgs_tensor, intermediates_only=True)
        features = intermediate_features[-1].permute(0, 2, 3, 1).reshape(-1, 1024).cpu()

    # pca_features = scaler.fit_transform(pca.fit_transform(features))

    pca_features = pca.fit_transform(features)

    # Prepare the result
    result_img = np.zeros((imgs_tensor.size(0) * 196, n))  # Assuming 14x14 patches
    # result_img[fg_indices] = normalized_features
    result_img = pca_features
    imgs_tensor = imgs_tensor.cpu()

    transformed_imgs = []
    for i, img in enumerate(imgs_tensor):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = (img_np * 255).astype(np.uint8)
        transformed_imgs.append(img_np)

    results = [result_img.reshape(imgs_tensor.size(0), 14, 14, n)[i] for i in range(len(images))]

    return results, transformed_imgs, pca_features

def smooth_heatmap(heatmap, kernel_size=5, sigma=2):
    """
    将类激活图平滑处理。

    参数：
    heatmap (numpy.ndarray): 原始激活图的2D数组。
    kernel_size (int): 高斯核的大小，应为奇数（例如15, 25等）。
    sigma (float): 高斯核的标准差，越高则模糊越强。

    返回:
    smooth_heatmap (numpy.ndarray): 平滑后的激活图。
    """

    heatmap_normalized = np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)))
    smooth_heatmap = cv2.GaussianBlur(heatmap_normalized, (kernel_size, kernel_size), sigma)
    smooth_heatmap = smooth_heatmap.astype(np.float32) / 255.0
    return smooth_heatmap


def preparefeatureLabel(batch_graph, batch_adjs, num_feat):
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
    max_node_num = 0

    for i in range(batch_size):
        max_node_num = max(max_node_num, batch_graph[i].shape[0])

    masks = torch.zeros(batch_size, max_node_num)
    adjs = torch.zeros(batch_size, max_node_num, max_node_num)
    batch_node_feat = torch.zeros(batch_size, max_node_num, num_feat)

    for i in range(batch_size):
        cur_node_num = batch_graph[i].shape[0]
        # node attribute feature
        tmp_node_fea = batch_graph[i]
        batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

        # adjs
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]

        # masks
        masks[i, 0:cur_node_num] = 1

    node_feat = batch_node_feat.cuda()
    adjs = adjs.cuda()
    masks = masks.cuda()
    return node_feat, adjs, masks

slide_name = '  '
slide_path = rf'./dataset/svs/{slide_name}.svs'
feature_path = rf'./results/features/{slide_name}_5xTiles'
patches_path = rf'./results/patches/{slide_name}_5xTiles'

patches_name = list(os.listdir(patches_path))
model = Classifier(1, 1024).cuda()
ckpt = torch.load('./checkpoints/bestModel.pth')
model.load_state_dict(ckpt)

LUCNEC_data = GraphDataset(feature_path, 1024)
LUCNEC_data_loader = DataLoader(LUCNEC_data, 1, shuffle=False, drop_last=False, collate_fn=collate)

model.eval()
for step, (slide) in enumerate(LUCNEC_data_loader):
    ig = IntegratedGradients(model)
    input = (slide['image'][0].unsqueeze(0), slide['adj_s'][0].unsqueeze(0))
    image = [torch.zeros(slide['image'][0].shape[0], slide['image'][0].shape[1]).unsqueeze(0)]
    adj_s = [torch.zeros(slide['adj_s'][0].shape[0], slide['adj_s'][0].shape[1]).unsqueeze(0)]
    baseline = (image[0], adj_s[0])
    attributions = ig.attribute(input, baseline, n_steps=100)

    patches_attribution = attributions[0]
    patches_attribution = patches_attribution.squeeze(0)
    patches_attribution = patches_attribution.cpu().detach().numpy()
    patches_attribution = np.mean(patches_attribution, axis=1)


    edges_attribution = attributions[1]
    edges_attribution = edges_attribution.squeeze(0)
    edges_attribution = edges_attribution.cpu().detach().numpy()
    model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0,
                              dynamic_img_size=True)
    model.load_state_dict(
        torch.load(os.path.join("./vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"),
                   map_location="cpu"), strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    n = 3
    pca = PCA(n_components=n)
    scaler = MinMaxScaler(clip=True)

    image_path = patches_path
    image_paths = list(os.listdir(image_path))
    image_paths = [rf'{image_path}/{path}' for path in image_paths]

    slide = openslide.OpenSlide(rf'{slide_path}')
    width, height = slide.dimensions

    slide_thumbnail = slide.get_thumbnail((width / 8, height / 8))
    patch_size = 112
    images = [load_and_preprocess_image(path) for path in image_paths]
    heatmap = np.ones((slide_thumbnail.size[1]+100, slide_thumbnail.size[0]+100, 3)) * 150
    batch_size = 32
    index = 0

    original_images = []
    weighted_images = []
    images_name = []
    for i in range(0, len(images), batch_size):
        print(rf'Step {i}')
        batch_images = images[i:i + batch_size]
        results, transformed_imgs, pca_features = process_images(batch_images, larger_pca_as_fg=False)
        for j, (image, result) in enumerate(zip(transformed_imgs, results)):
            image_name = image_paths[index].split('/')[-1]
            index = index + 1
            images_name.append(image_name)
            original_images.append(image)
            weighted_images.append(result)

    weighted_images = np.array(weighted_images)

    weighted_images[:, :, :, 0] = (weighted_images[:, :, :, 0] - weighted_images[:, :, :, 0].min()) / (
                weighted_images[:, :, :, 0].max() - weighted_images[:, :, :, 0].min())
    weighted_images[:, :, :, 1] = (weighted_images[:, :, :, 1] - weighted_images[:, :, :, 1].min()) / (
                weighted_images[:, :, :, 1].max() - weighted_images[:, :, :, 1].min())
    weighted_images[:, :, :, 2] = (weighted_images[:, :, :, 2] - weighted_images[:, :, :, 2].min()) / (
                weighted_images[:, :, :, 2].max() - weighted_images[:, :, :, 2].min())

    weights = np.mean(weighted_images, axis=3)
    patches_attribution = patches_attribution.reshape([weights.shape[0], 1, 1])
    patches_attribution = (patches_attribution - patches_attribution.min()) / (patches_attribution.max() - patches_attribution.min())
    weights_ = weights*patches_attribution
    weights_ = (weights_ - weights_.min()) / (weights_.max() - weights_.min())
    colored_weights = plt.cm.magma(weights_)
    colored_weights_ = colored_weights[:, :, :, :3]
    for i in range(len(original_images)):
        overlay = create_overlay_image(original_images[i], colored_weights_[i], patch_size)
        image_name = images_name[i]
        x, y = map(int,
                   (image_name.split('-')[-1].split('_')[0], image_name.split('-')[-1].split('_')[1].split('.')[0]))
        heatmap[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size, :] = overlay

    heatmap = heatmap.astype(np.uint8)
    patches_weight_from_sry = pd.read_csv(rf'.attention/AttentionPatches.csv')
    patches_weight_from_sry = np.array(patches_weight_from_sry.iloc[:, 1].to_list())
    patches_weight_from_sry = patches_weight_from_sry.reshape([weights.shape[0], 1, 1])
    weights_ = weights * patches_weight_from_sry
    weights_ = (weights_ - weights_.min()) / (weights_.max() - weights_.min())
    colored_weights = plt.cm.CMRmap(weights_)
    colored_weights_ = colored_weights[:, :, :, :3]
    for i in range(len(original_images)):
        overlay = create_overlay_image(original_images[i], colored_weights_[i], patch_size)
        image_name = images_name[i]
        x, y = map(int,
                   (image_name.split('-')[-1].split('_')[0], image_name.split('-')[-1].split('_')[1].split('.')[0]))
        heatmap[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size, :] = overlay

    heatmap = heatmap.astype(np.uint8)
    plt.imsave(rf'./{slide_name}_heatmap.pdf', heatmap)