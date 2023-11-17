# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import logging
import PIL.Image
import torch
from torch.utils import data
# -*- coding: utf-8 -*-
import numpy as np


# Configuration Class
class Config(object):

    MEAN_AND_STD = {'mean_rgb':np.array([0.485,0.456,0.406]),
                    'std_rgb':np.array([0.229,0.224,0.225])}

    # save visual feature map in this path
    SAVE_FEATURE_MAP = ''

    # set the size of test image
    SCALE_SIZE = 256

    # set your optimizer ['adam','sgd','rmsprop']
    OPTIMIZERS = 'adam'

    # initial learning rate
    LR = 0.0001

    # stepsize to decay learning rate (>0 means this is enabled)
    STEP_SIZE = -1

    # learning rate decay
    GAMMA = 0.1

    # weight decay (default: 5e-04)
    WEIGHT_DECAY = 5e-04


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


config = Config()


class MyData(data.Dataset):  # inherit
    """
    load data in a folder
    """

    def __init__(self, root, DF, transform=True):
        super(MyData, self).__init__()
        self.root = root
        self._transform = transform
        self.scale_size = config.SCALE_SIZE

        self.DF = pd.DataFrame(columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin',
                                        'xmax', 'ymax', 'width', 'height','discFlag','rater'])
        for spilt in DF:
            DF_all = pd.read_csv(root + '/' + 'Glaucoma_multirater_' + spilt + '.csv', encoding='gbk')

            DF_this = DF_all.loc[DF_all['rater'] == 0]      # Final Label
            DF_this = DF_this.reset_index(drop=True)
            DF_this = DF_this.drop('Unnamed: 0', 1)
            self.DF = pd.concat([self.DF, DF_this])

        self.DF.index = range(0, len(self.DF))


    def __len__(self):
        return len(self.DF)

    def __getitem__(self, index):
        img_Name = self.DF.loc[index, 'imgName']

        """ Get the images """
        fullPathName = os.path.join(self.root, img_Name)
        fullPathName = fullPathName.replace('\\', '/')  # image path

        img = PIL.Image.open(fullPathName).convert('RGB')  # read image
        img = img.resize((self.scale_size, self.scale_size))
        img = np.array(img)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)    # add additional channel in dim 2 (channel)

        img_ori = img

        """ Get the six raters masks """
        masks = []    # 用list装multi mask
        data_path = self.root
        for n in range(1,7):     # n:1-6
            # # load rater 1-6 label recurrently

            maskName = self.DF.loc[index, 'maskName'].replace('FinalLabel','Rater'+str(n))
            fullPathName = os.path.join(data_path, maskName)
            fullPathName = fullPathName.replace('\\', '/')
            #0,150,255
            Mask = PIL.Image.open(fullPathName).convert('L')
            Mask = Mask.resize((self.scale_size, self.scale_size))
            Mask = np.array(Mask)
            mask = Mask.copy()
            mask[(mask > 200) & (mask < 255)] = 255
            mask[(mask >= 75) & (mask <= 200)] = 150
            mask[(mask < 75)] = 0
            mask[mask == 255] = 2
            mask[mask == 150] = 1


            # Mask = Mask.transpose((2, 0, 1))
            mask = torch.from_numpy(mask)
            masks.append(mask)


        if self._transform:
            img_ori, img, masks = self.transform(img_ori, img, masks)
            return {'image': img, 'image_ori': img_ori, 'mask': masks, 'name': img_Name.split('.')[0]}
        else:
            return {'image': img, 'image_ori': img_ori, 'mask': masks, 'name': img_Name.split('.')[0]}


    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img_o, img, lbl):
        if img.max() > 1:
            img = img.astype(np.float64) / 255.0
        img -= config.MEAN_AND_STD['mean_rgb']
        img /= config.MEAN_AND_STD['std_rgb']
        img = img.transpose(2, 0, 1)  # to verify
        img = torch.from_numpy(img)

        if img.max() > 1:
            img_o = img_o.astype(np.float64) / 255.0
        img_o = img_o.transpose(2, 0, 1)  # to verify
        img_o = torch.from_numpy(img_o)

        return img_o, img, lbl
