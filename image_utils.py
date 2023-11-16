import cv2
import numpy as np
import torch.utils.data as data
from torchvision import transforms, utils
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import itertools
from sklearn.cluster import SpectralClustering
import torch.nn.functional as F

def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var**0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def color2gray(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_img_3d = image_array.copy()
    gray_img_3d[:, :, 0] = gray
    gray_img_3d[:, :, 1] = gray
    gray_img_3d[:, :, 2] = gray
    return gray_img_3d

def crop(image_array):
    image = cv2.resize(image_array, (256, 256))
    w = np.random.randint(0, 32)
    h = np.random.randint(0, 32)
    image = image[w:w+224,h:h+224,:]
    return image

def rotation(image_array):
    h,w = image_array.shape[:2]
    center = (w//2, h//2)
    angle = np.random.randint(-20, 20)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image_array, M, (w, h))
    return rotated

def Label_extend(args, predB, dis, label_dict):   ## joint distribution
    class_num = dis.shape[1]
    ctf_num = predB.shape[1]
    temps = []
    for key in label_dict.keys():
        ctfs = label_dict[key]
        temp = torch.zeros(size = (class_num,), dtype = torch.float).cuda()
        for ctf in ctfs:
            temp[ctf] = 1
        temps.append(temp)

    pred_extend = torch.zeros(size=(predB.shape[0], class_num), dtype=torch.float).cuda()
    for idx in range(ctf_num):
        temp = temps[idx].repeat(predB.shape[0], 1)
        predb = torch.unsqueeze(predB[:, idx], 1).repeat(1, class_num)
        pred_extend += temp * predb * dis
    # return F.softmax(pred_extend, dim = 1)
    return pred_extend

def make_confucion_matrix(preds, targets, class_name=None):
    if isinstance(preds, list):
        preds = torch.cat(preds, dim = 0)
    if isinstance(targets, list):
        targets = torch.cat(targets, dim = 0)

    font = {'family': 'Nimbus Roman',
            'style': 'normal',
            'size': 20}
    preds = torch.flatten(preds)
    targets = torch.flatten(targets)
    if class_name is None or type(class_name) != list:
        class_name = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
    cmtx = confusion_matrix(targets, preds, labels=list(range(len(class_name))), normalize='true')
    mean_acc = []
    for i in range(cmtx.shape[0]):
        mean_acc.append(cmtx[i, i])
    return mean_acc, cmtx


def plot_cmatrix(preds, targets, data2, stage, backbone):
    font = {'family': 'DejaVu Sans',
            'style': 'normal',
            'size': 20}

    class_name = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
    figure = plt.figure(figsize=(10,10))
    _, cmtx= make_confucion_matrix(preds, targets, class_name=None)


    plt.imshow(cmtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Consufion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_name))
    plt.xticks(tick_marks, class_name, rotation=45, fontsize=15)
    plt.yticks(tick_marks, class_name, fontsize=15)

    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        # color = 'white' if cmtx[i, j] > threshold else 'black'
        plt.text(
            j,
            i,
            format(cmtx[i, j], '.2f') ,#if cmtx[i, j] != 0 else '.',
            horizontalalignment='center',
            fontsize=20,
            # color=color,
            fontdict=font
        )

    plt.tight_layout()
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    if not os.path.exists('./' + data2 + backbone):
        os.mkdir('./' + data2 + backbone)
    save_path = os.path.join('./' + data2 + backbone, str(stage) + '.png')
    # print(save_path)
    plt.savefig(save_path)


def save_failure(preds, targets, imgs):
    print('save failure cases.................')
    path = './fail_imgs'
    if not os.path.exists(path):
        os.mkdir(path)

    batch = preds.shape[0]
    index = np.arange(batch)
    preds = preds.numpy()
    targets = targets.numpy()
    fail_index = index[preds != targets]
    print('%s failure cases..'%str(len(fail_index)))
    for i in fail_index:
        true_label = targets[i]
        pred_label = preds[i]
        cv2.imwrite(os.path.join(path, str(true_label) + '_' + str(pred_label) + '_' + str(i) + '.png'), cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))
        # utils.save_image(imgs[i], os.path.join(path, str(true_label) + '_' + str(pred_label) + '_' + str(i) + '.png'))

def save_threshold_samples(stage, p, num):
    fid = open('./threshold_record6.txt', 'a')
    fid.write(str(stage) + ' ')
    for idx in p:
        fid.write(str(idx) + ' ')
    for idx in num:
        fid.write(str(idx) + ' ') 
    fid.write('\n')   
    fid.close()    


def pseudo_accuracy(preds, targets):
    mean_acc, _ = make_confucion_matrix(targets, preds)
    return mean_acc
    
if __name__ == "__main__":
    plot_cmatrix()