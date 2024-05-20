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

    def forward(self, x, targets, idx, mode = 'train', task = 'source', epoch=0):
        fea = self.feature(x)
        out = self.bn(self.fc(fea))
        batch = fea.shape[0]
        if mode == 'train':
              if task == 'source':
                    features = self.split_feature_makeLD(fea, targets)
                    weight = self.volume(features)
                    w = torch.from_numpy(np.array(weight)).cuda()            
                    inter_loss = torch.tensor(0.).cuda()            
                    for i in range(7):
                        fea = features[i]
                        if len(fea) != 0:
                            fea_= remove_element(features, i).cuda()
                            inter_loss += mmd_loss(fea, fea_) * w[i]
                    affinity_loss = -1 * inter_loss
                    return [out, affinity_loss] 

              if task == 'target':  ####calculate affinity loss for all source and confident target samples.
                    idx = (idx == 1).nonzero().squeeze()
                    fea = torch.index_select(fea, 0, idx)
                    targets = torch.index_select(targets, 0, idx)

                    source_features = self.split_feature_makeLD(fea[:batch//2], targets[:batch//2])
                    target_features = self.split_feature_makeLD(fea[batch//2:], targets[batch//2:])
                    features = self.split_feature_makeLD(fea, targets)
                    weight = self.volume(features)
                    w = torch.from_numpy(np.array(weight)).cuda()       
                    intra_loss, inter_loss = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()             
                    for i in range(7):
                        fea_s = source_features[i]
                        fea_t = target_features[i]
                        if len(fea_s) != 0 and len(fea_t) != 0:
                            intra_loss += mmd_loss(fea_s, fea_t) * w[i]

                        fea = features[i]
                        if len(fea) != 0:
                            fea_= remove_element(features, i).cuda()
                            inter_loss += mmd_loss(fea, fea_) * w[i]

                    affinity_loss = intra_loss - inter_loss
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
        return x_parts

    def volume(self, features):
        volume = np.zeros(shape = (len(features)))
        for idx in range(len(features)):
            f = features[idx].cpu().detach().numpy()
            if len(f) != 0:
                self.kde.fit(f)
                v = np.sum(1/(self.kde.score_samples(f)))
            else:
                v = 0.00001
            volume[idx] = v
        # volume = np.exp(volume) / np.sum(np.exp(volume))
        weight = cal_weight(volume)
        return weight
    
def mmd_loss(source_features, target_features):
    # sigma =1.0 # MMD的高斯核宽度
    num_samples = min(source_features.size(0), target_features.size(0))
    matched_source = source_features[torch.randperm(source_features.size(0))[:num_samples]]
    matched_target = target_features[torch.randperm(target_features.size(0))[:num_samples]]
    
    # 计算源域和目标域的核矩阵
    kernels = compute_kernel_matrix(matched_source, matched_target)
    XX = kernels[:num_samples, :num_samples]
    YY = kernels[num_samples:, num_samples:]
    XY = kernels[:num_samples, num_samples:]
    YX = kernels[num_samples:, :num_samples]
    loss = torch.mean(XX + YY - XY -YX)    
    return loss

def compute_kernel_matrix(source, target, kernel_mul=2.0, kernel_num=5):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    bandwidth=5
    # bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples) ##bandwidth=0.5
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)#/len(kernel_val)

def remove_element(matrix, index):
    dim = matrix[0].shape[-1]
    matrix1 = matrix[:index]
    matrix2 = matrix[index+1:]
    data = []
    for item in matrix1:
        data.extend(item)
    for item in matrix2:
        data.extend(item)
    result = torch.cat(data, 0).reshape(-1, dim)
    return result

