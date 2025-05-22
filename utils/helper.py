#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils.metrics import ConfusionMatrix
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
from scipy import interp
import seaborn as sn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.font_manager import FontProperties
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lifelines.statistics import logrank_test
torch.backends.cudnn.deterministic = True


def collate(batch):
    image = [b['image'] for b in batch]  # w, h
    label = [b['label'] for b in batch]
    id = [b['id'] for b in batch]
    adj_s = [b['adj_s'] for b in batch]
    return {'image': image, 'label': label, 'id': id, 'adj_s': adj_s}


def preparefeatureLabel(batch_graph, batch_label, batch_adjs, num_feat):
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
    max_node_num = 0

    for i in range(batch_size):
        labels[i] = batch_label[i]
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
    labels = labels.cuda()
    adjs = adjs.cuda()
    masks = masks.cuda()

    return node_feat, labels, adjs, masks


class Trainer(object):
    def __init__(self, n_class, num_feats):
        self.metrics = ConfusionMatrix(n_class)
        self.num_feats = num_feats

    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()

    def plot_cm(self):
        self.metrics.plotcm()

    def train(self, sample, model):
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'],
                                                             self.num_feats)
        pidList = sample['id']
        pred, labels, loss, probs = model.forward(node_feat, labels, adjs, masks, pidList)

        return pred, labels, loss, probs


class Evaluator(object):
    def __init__(self, n_class, num_feats):
        self.metrics = ConfusionMatrix(n_class)
        self.num_feats = num_feats

    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()

    def plot_cm(self):
        return self.metrics.plotcm()

    def eval_test(self, sample, model, graphcam_flag=False):        # 注意这个是test
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'],
                                                             self.num_feats)
        pidList = sample['id']
        if not graphcam_flag:
            with torch.no_grad():
                pred, labels, loss, probs = model.forward(node_feat, labels, adjs, masks, pidList)
        else:
            torch.set_grad_enabled(True)
            pred, labels, loss, probs = model.forward(node_feat, labels, adjs, masks, pidList, graphcam_flag=graphcam_flag)
        return pred, labels, loss, probs

    def calculate_accuracy(self, cm):
        correct_samples = np.trace(cm)
        total_samples = np.sum(cm)
        accuracy = correct_samples / total_samples
        return accuracy

    def save_cm(self, cm, save_path):
        class_names = ['PS', 'PL']

        plt.figure(figsize=(6, 4))
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        ax = sn.heatmap(df_cm, annot=True, fmt='.20g', cmap='Blues')

        correctNum = np.trace(cm)
        allNum = np.sum(cm)
        acc = correctNum / allNum
        # acc = round(acc, 3)
        acc = ('%.3f' % acc)
        print("acc: ", acc)
        ax.set_title(f'Confusion matrix(Acc={acc})')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()


    def save_CSV(self, sampleNames, trueLabels, predProbs, predLabels, save_path):
        trueLabelNames = ["LCNEC" if temp == 1 else "SCLC" for temp in trueLabels]
        predLabelNames = ["LCNEC" if temp == 1 else "SCLC" for temp in predLabels]
        predCorrects = ["YES" if trueLabel == predLabel else "NO" for (trueLabel, predLabel) in zip(trueLabelNames, predLabelNames)]

        df = pd.DataFrame({
            "sampleName": sampleNames,
            "predProb(LCNEC)": predProbs,
            "predLabel": predLabelNames,
            "trueLabel": trueLabelNames,
            "predCorrect": predCorrects
        })

        df.to_csv(save_path, index=False)

        precision = precision_score(trueLabels, predLabels, average='binary')
        recall = recall_score(trueLabels, predLabels, average='binary')
        f1Score = f1_score(trueLabels, predLabels, average='binary')
        print("===================== Precision: {:.3f}, Recall: {:.3f}, F1-score: {:.3f}".format(precision, recall, f1Score))

    def plot_ROC(self, trueLabels, predProbs, save_path):
        plt.figure(figsize=(20, 20), dpi=300)
        y_true = np.array(trueLabels)
        y_score = np.array(predProbs)
        B = 1000
        auc_values = []
        for i in range(B):
            indices = np.random.choice(range(len(y_true)), len(y_true), replace=True)
            y_true_bootstrap = y_true[indices]
            y_score_bootstrap = y_score[indices]
            auc_value = roc_auc_score(y_true_bootstrap, y_score_bootstrap)
            auc_values.append(auc_value)
        alpha = 0.95
        p = ((1.0 - alpha) / 2.0) * 100
        confidence_lower = max(0.0, np.percentile(auc_values, p))
        p = (alpha + ((1.0 - alpha) / 2.0)) * 100
        confidence_upper = min(1.0, np.percentile(auc_values, p))
        print("================AUC: {:.3f} ({:.1f} %CI {:.3f} to {:.3f}) ".format(roc_auc_score(y_true, y_score), alpha * 100, confidence_lower,confidence_upper))

        fpr, tpr, thresholds = roc_curve(trueLabels, predProbs, pos_label=1)
        AUROC_value = auc(fpr, tpr)
        print("================AUROC={:.3f}".format(AUROC_value))
        plt.plot(fpr, tpr, lw=6, label='AUROC={:.3f}({:.3f}-{:.3f})'.format(AUROC_value, confidence_lower, confidence_upper), color="#ff9999")
        # plt.fill_between(fpr, tpr, color='gray', alpha=0.4, label='Confidence Interval (95% CI = {:.3f}-{:.3f})'.format(confidence_lower, confidence_upper))  # 给ROC曲线也加上置信范围
        plt.plot([0, 1], [0, 1], '--', lw=5, color='gray')
        plt.axis('square')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xticks(fontsize=35, weight='bold')
        plt.yticks(fontsize=35, weight='bold')
        plt.xlabel('1 - Specificity', fontsize=30, weight='bold')
        plt.ylabel('Sensitivity', fontsize=30, weight='bold')
        # plt.title('ROC Curve', fontsize=50, weight='bold')

        legend_font = FontProperties(weight='bold', size=50)
        plt.legend(loc='lower right', prop=legend_font, bbox_to_anchor=(0.875, 0.250, 0.125, 0.125))

        plt.savefig(save_path, format="png", dpi=300)
        plt.close()

    def plot_originROC(self, trueLabels, predProbs, save_path):
        plt.figure(figsize=(20, 20), dpi=300)

        fpr, tpr, thresholds = roc_curve(trueLabels, predProbs, pos_label=1)
        AUROC_value = auc(fpr, tpr)
        print("================AUROC={:.3f}".format(AUROC_value))
        plt.plot(fpr, tpr, lw=6, label='AUROC={:.3f}'.format(AUROC_value), color="#ff9999")
        plt.plot([0, 1], [0, 1], '--', lw=5, color='gray')
        plt.axis('square')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xticks(fontsize=35, weight='bold')
        plt.yticks(fontsize=35, weight='bold')
        plt.xlabel('1 - Specificity', fontsize=30, weight='bold')
        plt.ylabel('Sensitivity', fontsize=30, weight='bold')
        # plt.title('ROC Curve', fontsize=50, weight='bold')

        legend_font = FontProperties(weight='bold', size=50)
        plt.legend(loc='lower right', prop=legend_font, bbox_to_anchor=(0.875, 0.250, 0.125, 0.125))

        plt.savefig(save_path, format="png", dpi=300)
        plt.close()


    def plot_mutiROC(self, labelsList, probsList, save_path):

        classes = ["Small", "Large"]
        colorNames = ["#ff796c", "#448ee4"]
        plt.figure(figsize=(20, 20), dpi=100)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(labelsList[i], probsList[i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(classes)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        #画macro ROC
        plt.plot(fpr["macro"], tpr["macro"], label='Macro (AUROC={:.3f})'.format(roc_auc["macro"]), linestyle='--', linewidth=6, color='black')

        for (className, colorName, trueLabels, probs) in zip(classes, colorNames, labelsList, probsList):
            fpr, tpr, thresholds = roc_curve(trueLabels, probs, pos_label=1)

            plt.plot(fpr, tpr, lw=5, label='{} (AUROC={:.3f})'.format(className, auc(fpr, tpr)), color=colorName)
            plt.plot([0, 1], [0, 1], '--', lw=5, color='gray')
            plt.axis('square')
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xticks(fontsize=35, weight='bold')
            plt.yticks(fontsize=35, weight='bold')
            plt.xlabel('1 - Specificity', fontsize=40, weight='bold')
            plt.ylabel('Sensitivity', fontsize=40, weight='bold')
            plt.title('ROC Curve', fontsize=50, weight='bold')

            legend_font = FontProperties(weight='bold', size=40)
            plt.legend(loc='lower right', prop=legend_font, bbox_to_anchor=(0.875, 0.125, 0.125, 0.125))

            plt.savefig(save_path, format="png", dpi=300)

        plt.close()

    def plot_PR(self, trueLabels, predProbs, save_path):
        plt.figure(figsize=(20, 20), dpi=300)
        precision, recall, thresholds = precision_recall_curve(trueLabels, predProbs, pos_label=1)
        AUPRC_value = average_precision_score(trueLabels, predProbs, pos_label=1)
        plt.plot(recall, precision, lw=6, label='AUPRC={:.3f}'.format(AUPRC_value), color="#33ccff")
        plt.axis('square')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xticks(fontsize=35, weight='bold')
        plt.yticks(fontsize=35, weight='bold')
        plt.xlabel('Recall', fontsize=30, weight='bold')
        plt.ylabel('Precision', fontsize=30, weight='bold')
        # plt.title('ROC Curve', fontsize=50, weight='bold')

        legend_font = FontProperties(weight='bold', size=50)
        plt.legend(loc='lower left', prop=legend_font, bbox_to_anchor=(-0.010, 0.125, 0.125, 0.125))

        plt.savefig(save_path, format="png", dpi=300)
        plt.close()

    def plotSurvival(self, csvPath, epoch, survivalType, treatWay):
        XH_LSPath = csvPath + "epoch" + str(epoch) + '/XH_LS_25/predResults.csv'
        BZ_LSPath = csvPath + "epoch" + str(epoch) + '/BZ_LS_11/predResults.csv'
        XH_LSData = pd.read_csv(XH_LSPath)
        BZ_LSData = pd.read_csv(BZ_LSPath)

        AllLSData = pd.concat([XH_LSData, BZ_LSData], axis=0, ignore_index=True)
        AllLSPredData = AllLSData.drop(columns=["trueLabel", "predCorrect"])
        AllLSPredData = AllLSPredData[AllLSPredData["sampleName"] != "LS-4 - 2023-05-15 17.17.26"]

        clinicalData = pd.read_csv("./data/clinicalData/LS_clinicalData.csv")
        AllClinicalData = clinicalData.drop(clinicalData.columns[[1, 3, 8, 9, 10, 15]], axis=1)

        allPredRes = AllLSPredData.merge(AllClinicalData, left_on="sampleName", right_on="sName", how="left")
        allPredRes.drop(columns="sName", inplace=True)

        df = allPredRes[allPredRes[f"{treatWay}"] == 1]

        item = survivalType
        df = df.dropna(subset=[f'{item}'])
        T = df[f'{item}.time']
        E = df[f'{item}']

        results = logrank_test(df.loc[df['predLabel'] == 'LCNEC', f'{item}.time'],
                               df.loc[df['predLabel'] == 'SCLC', f'{item}.time'],
                               event_observed_A=df.loc[df['predLabel'] == 'LCNEC', f'{item}'],
                               event_observed_B=df.loc[df['predLabel'] == 'SCLC', f'{item}'])
        # results.print_summary()


class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

        print('Focal Loss:')
        print('    Alpha = {}'.format(self.alpha))
        print('    Gamma = {}'.format(self.gamma))

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                      preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    在optimizer中会设置一个基础学习率base lr,
    当multiplier>1时,预热机制会在total_epoch内把学习率从base lr逐渐增加到multiplier*base lr,再接着开始正常的scheduler
    当multiplier==1.0时,预热机制会在total_epoch内把学习率从0逐渐增加到base lr,再接着开始正常的scheduler
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler and (not self.finished):
                self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                self.finished = True
            # !这是很关键的一个环节，需要直接返回新的base-lr
            return [base_lr for base_lr in self.after_scheduler.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        print('warmuping...')
        if self.last_epoch <= self.total_epoch:
            warmup_lr = None
            if self.multiplier == 1.0:
                warmup_lr = [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
            else:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                             self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
