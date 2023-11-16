import torch.utils.data as data
import cv2
import pandas as pd
import os
import image_utils as util
import random
import glob
import numpy as np
import random
import shutil



"0: surprise, 1: fear, 2: disgust, 3: happy 4: sad  5: angry 6: neutral 7:attempt"

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, strong_transform =None, basic_aug=False, ratio=1):
        self.phase = phase
        self.transform = transform
        self.strong_transform = strong_transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:,
                     LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        ###shuffle dataset
        seed = np.random.seed(2000)
        np.random.shuffle(file_names)
        seed = np.random.seed(2000)
        np.random.shuffle(self.label)
                
        self.file_paths = []
        # use raf-db aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        distribute = np.array(self.label)

        self.label_dis = [np.sum(distribute == 0), np.sum(distribute == 1), np.sum(distribute == 2),
                          np.sum(distribute == 3), \
                          np.sum(distribute == 4), np.sum(distribute == 5), np.sum(distribute == 6)]
        print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (
        self.label_dis[0], self.label_dis[1], self.label_dis[2], self.label_dis[3], \
        self.label_dis[4], self.label_dis[5], self.label_dis[6]))

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape=len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        img = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 2)
                img = self.aug_func[index](img)

        if self.transform is not None:
            img = self.transform(img)
            
        if self.strong_transform is not None:
            img_aug = self.strong_transform(img)
            return img, img_aug, label
        else:
            return img, label


class FER(data.Dataset):
    def __init__(self, path, phase, transform=None, strong_transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.strong_transform = strong_transform

        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        self.file_paths, self.label = [], []
        if self.phase == 'train':
            files = glob.glob(os.path.join(path, 'train/*/*.jpg'))
            files += glob.glob(os.path.join(path, 'val/*/*.jpg'))
            seed = np.random.seed(2000)
            np.random.shuffle(files)
        else:
            files = glob.glob(os.path.join(path, 'test/*/*.jpg'))
        for file in files:
            self.file_paths.append(file)
            self.label.append(int(file.split('/')[-2]))
        distribute = np.array(self.label)
        self.label_dis = [ np.sum(distribute == 0),  np.sum(distribute == 1),  np.sum(distribute == 2),  np.sum(distribute == 3),  \
                      np.sum(distribute == 4),  np.sum(distribute == 5),  np.sum(distribute == 6)]
        print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (self.label_dis[0], self.label_dis[1], self.label_dis[2],self.label_dis[3],\
                                                                          self.label_dis[4],self.label_dis[5],self.label_dis[6]))

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape = len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            img = self.transform(image)

        if self.strong_transform is not None:
            img_aug = self.strong_transform(image)
            return img, img_aug, label

        else:
            return img, label
                      