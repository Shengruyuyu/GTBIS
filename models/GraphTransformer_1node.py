import sys
import os
import torch
import random
import numpy as np
import torch.nn.functional as F


from .ViT_1node import *
from .gcn import GCNBlock

from torch_geometric.nn import GCNConv, DenseGraphConv, dense_mincut_pool
from torch.nn import Linear


class Classifier(nn.Module):
    def __init__(self, n_class, num_feats):
        super(Classifier, self).__init__()

        self.embed_dim = 64
        self.num_layers = 3
        self.node_cluster_num = 100
        # self.node_cluster_num = 75
        self.num_feats = num_feats
        self.transformer = VisionTransformer(num_classes=n_class, embed_dim=self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.criterion = nn.BCEWithLogitsLoss()

        self.bn = 1
        self.add_self = 1
        self.normalize_embedding = 1
        self.conv1 = GCNBlock(self.num_feats, self.embed_dim, self.bn, self.add_self, self.normalize_embedding, 0., 0)
        self.pool1 = Linear(self.embed_dim, self.node_cluster_num)


    def forward(self,node_feat,labels,adj,mask, pidList, is_print=False, graphcam_flag=False):
        # node_feat, labels = self.PrepareFeatureLabel(batch_graph)
        cls_loss=node_feat.new_zeros(self.num_layers)
        rank_loss=node_feat.new_zeros(self.num_layers-1)
        X=node_feat
        X=mask.unsqueeze(2)*X
        X = self.conv1(X, adj, mask)
        features_GCN = X
        s = self.pool1(X)

        if graphcam_flag:
            pid = pidList[0]
            graphcamPath = "results/" + pid + '/'
            os.makedirs(graphcamPath, exist_ok=True)
            s_matrix = torch.argmax(s[0], dim=1)
            from os import path
            torch.save(s_matrix, graphcamPath + 's_matrix.pt')
            torch.save(s[0], graphcamPath + 's_matrix_ori.pt')

            if path.exists(graphcamPath + 'att_1.pt'):
                os.remove(graphcamPath + 'att_1.pt')
                os.remove(graphcamPath + 'att_2.pt')
                os.remove(graphcamPath + 'att_3.pt')

        pName = pidList[0]
        X, adj, mc1, o1 = dense_mincut_pool(X, adj, s, mask)
        b, _, _ = X.shape
        cls_token = self.cls_token.repeat(b, 1, 1)  # (1, 1, 64)
        X = torch.cat([cls_token, X], dim=1)
        out = self.transformer(X)
        features_GCN = features_GCN.tolist()
        labels = labels.unsqueeze(1)
        labels = labels.float()
        probs = F.sigmoid(out)
        # loss
        loss = self.criterion(out, labels)
        loss = loss + mc1 + o1
        threshold = 0.5
        pred = [int(1) if item >= threshold else int(0) for item in probs]
        pred = (torch.Tensor(pred)).to(torch.int64)
        labels = (labels.squeeze()).to(torch.int64)
        if graphcam_flag:
            print('GraphCAM enabled')
            # p = F.softmax(out)
            p = F.sigmoid(out)
            torch.save(p, graphcamPath + 'prob.pt')
            one_hot = np.zeros((1, out.size()[-1]), dtype=np.float32)
            one_hot[0, 0] = out[0][0]
            one_hot_vector = one_hot

            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * out)
            self.transformer.zero_grad()
            one_hot.backward(retain_graph=True)

            kwargs = {"alpha": 1}
            cam = self.transformer.relprop(torch.tensor(one_hot_vector).to(X.device), method="transformer_attribution", is_ablation=False,
                                        start_layer=0, **kwargs)

            torch.save(cam, graphcamPath + 'cam_0.pt')

        # return pred,labels,loss,probs, features_GCN
        return pred,labels,loss,probs
