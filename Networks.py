from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import math
import numpy as np
from sklearn import svm
from sklearn.neighbors import KernelDensity

def cal_weight(x):
    w = 1 / (x + 0.00001)
   # w = (w - w.min())/(w.max() -　w.min()) + 1
    return np.array(w)

class Model(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=7, pretrained=True, drop_rate=0.0):
        super(Model, self).__init__()
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(num_classes)
        self.kde = KernelDensity(bandwidth=0.2, kernel='gaussian')

        if backbone == 'resnet18':
            self.feature = nn.Sequential(*list(models.resnet18(pretrained).children())[:-1], nn.Flatten(), nn.Dropout(drop_rate))
            self.fc = nn.Linear(512, num_classes, bias = False)
        elif backbone == 'resnet50':
            self.feature = nn.Sequential(*list(models.resnet50(pretrained).children())[:-1], nn.Flatten())#, nn.Dropout(drop_rate), nn.Linear(2048, 512))
            self.fc = nn.Linear(2048, num_classes, bias = False)

        elif backbone == 'mobilenet_v2':
            self.feature = nn.Sequential(*list(models.mobilenet_v2(pretrained).children())[:-1], nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop_rate), nn.Linear(1280, 512), nn.Dropout(drop_rate)) 
            self.fc = nn.Linear(512, num_classes, bias = False)

        else:
            raise ValueError('Backbone Error!')

    def forward(self, x, targets, idx, mode = 'train', task = 'source'):
        fea = self.feature(x)
        out = self.bn(self.fc(fea))
        if mode == 'train':
              if task == 'target':  ####calculate affinity loss for all source training samples while calculate affinity loss for only confident target samples.
                  idx = (idx == 1).nonzero().squeeze()
                  fea = torch.index_select(fea, 0, idx)
                  targets = torch.index_select(targets, 0, idx)
              c_num = len(set(targets.cpu().numpy()))
              if c_num < self.num_classes:
                  return [out]
              batch = fea.shape[0]
              targets_1 = targets.unsqueeze(1)
              one_hot = torch.zeros(targets_1.shape[0], self.num_classes).cuda().scatter_(1, targets_1, 1)
              features2, inds_softlabel = self.split_feature_makeLD(fea, targets) ### inds_softlabel = [[***],..., [***]] store each class index
              cos_dot_product_matrix = fea.mm(fea.t())
              feature_norm = torch.norm(fea, dim=1).unsqueeze(1)
              feature_norm_product = feature_norm.mm(feature_norm.t())
              cos_similarity = cos_dot_product_matrix / feature_norm_product  ### B * B
              cos_matrixs = [cos_similarity[:, ind[:, 0]] for ind in inds_softlabel]
              cos_matrixs_mean = [torch.mean(cos_m, dim=1, keepdim=True) for cos_m in cos_matrixs]  ### C * B
              cos_mean = torch.cat(tuple(cos_matrixs_mean), dim=1)   ## B * C

              volume = np.zeros(shape = (len(features2)))
              for idx in range(len(features2)):
                  f = features2[idx].cpu().detach().numpy()
                  self.kde.fit(f)
                  v = np.sum(self.kde.score_samples(f)) #np.sum(self.kde.score_samples(f))
                  volume[idx] = v
              weight = cal_weight(volume)

              w = torch.from_numpy(np.array(weight)).unsqueeze_(0).repeat(batch, 1).cuda()
              similarity1 = torch.sum(cos_mean * one_hot * w, dim=0)   ###intra-class
              similarity2 = torch.sum(cos_mean * (1-one_hot) * w, dim=1)/(one_hot.size(1)-1)   ### inter_class
              affinity_loss = (1+similarity2.mean()+(1-similarity1.mean()))/2

              return [out, affinity_loss]

        else:
          return out, fea.cpu()

        
    def split_feature_makeLD(self, x, target):
        x_parts = []
        inds = []
        for c in range(self.num_classes):
            ind = (target == c).nonzero()
            inds.append(ind)
            x_parts.append(x[ind[:, 0], :])
        return x_parts, inds


