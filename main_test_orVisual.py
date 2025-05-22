#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function
import os
import torch
from utils.dataset import GraphDataset
from utils.helper import Evaluator, collate
from utils.option import Options

from models.GraphTransformer_1node import Classifier

args = Options().parse()
torch.cuda.synchronize()
torch.backends.cudnn.deterministic = True

n_class = 2

test = True
graphcam = False

batch_size = 1
log_interval_local = 6
num_feats = 1024
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

k = 1  # 第几折
task_name = f"GTP{k}"
print(task_name)
resume = rf"results/saved_models/GTP_Model{k}.pth"

data_path = "results/features"
val_set = rf"dataset/train_val_txt/val_{k}.txt"

valName = val_set.rsplit('/', 1)[1]
valName = valName.split('.')[0]

save_path = f"results/evaluate_results/fold_{k}/"
os.makedirs(save_path, exist_ok=True)

##### Load datasets
print("preparing datasets and dataloaders......")


ids_val = open(val_set).readlines()
dataset_val = GraphDataset(os.path.join(data_path, ""), ids_val, num_feats)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=0,
                                             collate_fn=collate, shuffle=False, pin_memory=True)
total_val_num = len(dataloader_val) * batch_size

##### creating models #############
print("creating models......")

model = Classifier(n_class, num_feats)
model.load_state_dict(torch.load(resume))

print('loaded model{}'.format(resume))


if torch.cuda.is_available():
    model = model.to(device)

evaluator = Evaluator(n_class, num_feats)

with torch.no_grad():
    model.eval()
    print("===================================evaluating...")

    total = 0.
    batch_idx = 0

    sampleNamesList = []
    trueLabelsList = []
    predProbsList = []
    predLabelsList = []
    for i_batch, sample_batched in enumerate(dataloader_val):
        preds, labels, _, probs = evaluator.eval_test(sample_batched, model, graphcam)

        labelsList = labels.tolist()
        if isinstance(labelsList, int):
            labelsList = [labelsList]
        if test and not graphcam:

            sampleNames = [item.rsplit("_", 1)[0] for item in sample_batched["id"]]
            sampleNamesList.extend(sampleNames)

            trueLabelsList.extend(labelsList)

            predProbs = (probs.squeeze()).tolist()
            predProbsList.append(predProbs)
            predLabelsList.append(preds.item())

            labels = [labels]
            total += len(labels)

            evaluator.metrics.update(labels, preds)

        if (i_batch + 1) % log_interval_local == 0:
            print('[%d/%d] val agg acc: %.3f' % (total, total_val_num, evaluator.get_scores()))
            evaluator.plot_cm()

    print('[%d/%d] val agg acc: %.3f' % (total_val_num, total_val_num, evaluator.get_scores()))
    cm = evaluator.plot_cm()
    if test and not graphcam:
        savedName = val_set.rsplit('/', 1)[1]
        savedName = savedName.split('.')[0]
        evaluator.save_cm(cm, save_path + f"Confusion_matrix({savedName}).png")
        evaluator.save_CSV(sampleNamesList, trueLabelsList, predProbsList, predLabelsList, save_path + f"predResults({savedName}).csv")
        evaluator.plot_ROC(trueLabelsList, predProbsList, save_path + f"ROC({savedName}).png")
        evaluator.plot_originROC(trueLabelsList, predProbsList, save_path + f"ROC({savedName}).png")
        evaluator.plot_PR(trueLabelsList, predProbsList, save_path + f"PR({savedName}).png")

    val_acc = evaluator.get_scores()
    print("===========================================")
    print("The last val_acc", val_acc)


