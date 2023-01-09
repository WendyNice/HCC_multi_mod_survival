import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import csv
from opts import parse_opts
from utils import OsJoin
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torchvision import transforms as transforms
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from scipy.ndimage.interpolation import zoom
import os
import torchio as tio
from augmentation import augmentation_func


opt = parse_opts()
data_type = opt.data_type
# print('opt.data_root_path', opt.data_root_path)
csv_dir = OsJoin(opt.data_root_path, 'csv', data_type)

# def default_loader(path):
#     img_pil = sitk.ReadImage(path)
#     img_arr = sitk.GetArrayFromImage(img_pil)
#     img_arr_cleaned = np.nan_to_num(img_arr)    # Replace NaN with zero and infinity with large finite numbers.
#     img_arr_cleaned = img_arr_cleaned /img_arr_cleaned.max()
#     img_pil = torch.from_numpy(img_arr_cleaned)
#     return img_pil


def npy_loader_train(path):
    # print('path', path)
    img_nii = sitk.ReadImage(path)
    img_npy = sitk.GetArrayFromImage(img_nii)
    # print('img_npy', img_npy.shape)
    # img_npy = np.load(path)
    # print('img_npy', img_npy.shape)
    # img_npy = img_npy[:3, :, :, :]
    # plt.imshow(img_npy[0, 10, :, :])
    # plt.show()
    img_npy = img_npy / np.max(img_npy)
    # img_npy = np.expand_dims(img_npy, axis=0)
    # img_ori_save = sitk.GetImageFromArray(img_npy)
    # print('img_npy', img_npy.shape)

    img_pil = torch.from_numpy(img_npy)
    img_pil = augmentation_func(img_pil)
    # print('img_pil', img_pil.shape)
    # plt.imshow(img_pil[0, 50, :, :])
    # plt.show()
    # img_save = sitk.GetImageFromArray(img_pil)
    # path_save = '/home/amax/Wendy/nnUNet/ouput_kaggle/aug_img'
    # sitk.WriteImage(img_save, os.path.join(path_save, path.split('/')[-1] + '.nii.gz'))
    # sitk.WriteImage(img_ori_save, os.path.join(path_save, path.split('/')[-1] + 'ori.nii.gz'))
    return img_pil


def npy_loader(path):
    img_nii = sitk.ReadImage(path)
    img_npy = sitk.GetArrayFromImage(img_nii)
    # img_npy = img_npy[:3, :, :, :]
    # img_npy = np.load(path)
    # print('img_npy', img_npy.shape)
    img_npy = img_npy / np.max(img_npy)
    # img_npy = np.expand_dims(img_npy, axis=0)
    img_pil = torch.from_numpy(img_npy)
    return img_pil


class TrainSet(Dataset):

    def __init__(self, fold_id, loader=npy_loader_train):
        print('csv_dir', csv_dir)
        with open(csv_dir + '/train_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_train = [OsJoin(opt.data_root_path, row[0]) for row in reader]
        with open(csv_dir + '/train_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_train = [row[1] for row in reader]
        self.image = file_train
        self.label = [int(i) for i in label_train]
        self.loader = loader
        # print('self.image', len(self.image))
        # print('self.label', len(self.label), np.sum(self.label))

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        # print('fn', fn)
        # print('label', label)
        return img, label

    def __len__(self):
        return len(self.image)

class ValidSet(Dataset):
    def __init__(self, fold_id, loader = npy_loader):
        with open(csv_dir + '/val_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_valid = [OsJoin(opt.data_root_path, row[0]) for row in reader]
        with open(csv_dir + '/val_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_valid = [row[1] for row in reader]
        self.image = file_valid
        self.label = [int(i) for i in label_valid]
        self.loader = loader
        # print('self.image', self.image)
        # print('self.label', self.label)

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.image)

class TestSet(Dataset):

    def __init__(self, fold_id, loader=npy_loader):
        with open(csv_dir + '/test_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_test = [OsJoin(opt.data_root_path, row[0]) for row in reader]
        with open(csv_dir + '/test_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_test = [row[1] for row in reader]
        self.image = file_test
        self.label = [int(i) for i in label_test]
        self.loader = loader

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.image)


class AllSet(Dataset):

    def __init__(self, fold_id, loader=npy_loader):
        with open(csv_dir + '/All_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_test = [OsJoin(opt.data_root_path, row[0]) for row in reader]
        with open(csv_dir + '/All_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_test = [row[1] for row in reader]
        self.image = file_test
        self.label = [int(i) for i in label_test]
        self.loader = loader

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.image)


class ExtraValidSet(Dataset):

    def __init__(self, fold_id, loader=npy_loader):
        with open(csv_dir + '/extra_valid_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_test = [OsJoin(opt.data_root_path, row[0]) for row in reader]
        with open(csv_dir + '/extra_valid_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_test = [row[1] for row in reader]
        self.image = file_test
        self.label = [int(i) for i in label_test]
        self.loader = loader

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.image)


class ExtraAllValidSet(Dataset):

    def __init__(self, fold_id, loader=npy_loader):
        with open(csv_dir + '/extra_all_valid_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            file_test = [OsJoin(opt.data_root_path, row[0]) for row in reader]
        with open(csv_dir + '/extra_all_valid_fold%s.csv'%str(fold_id), 'r') as csvfile:
            reader = csv.reader(csvfile)
            label_test = [row[1] for row in reader]
        self.image = file_test
        self.label = [int(i) for i in label_test]
        self.loader = loader

    def __getitem__(self, index):
        fn = self.image[index]
        img = self.loader(fn)
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.image)

