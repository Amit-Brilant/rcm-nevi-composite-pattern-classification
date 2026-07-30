from itertools import count
from random import shuffle
from re import M
import torch
import numpy as np
import pandas as pd

import os
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

import albumentations as A
import kornia.augmentation as K
import cv2

SEED = 42
all_classes = [ 'clod', 'mesh', 'ring', 'none', 'clod_mesh', 'clod_ring', 'mesh_ring', 'clod_mesh_ring' ]
class_to_id = {   
                    'none':         0,
                    'ring':         1,
                    'mesh':         2,
                    'mesh_ring':    2,
                    'clod_mesh':    3,
                    'clod_mesh_ring':   3,
                    'clod':         4,
                    'clod_ring':    4,       
}
id_to_class = {
    0: 'none',
    1: 'ring',
    2: 'mesh',
    3: 'clod_mesh',
    4: 'clod'
}

class_to_conf_mat = {
    'none':                         class_to_id['none'],
    'ring':                         class_to_id['ring'],
    'mesh /\nmesh_ring':            class_to_id['mesh_ring'],
    'clod_mesh /\nclod_mesh_ring':  class_to_id['clod_mesh'],
    'clod /\nclod_ring':            class_to_id['clod_ring'],
}
num_classes = 5
tile_width, tile_height = 224, 224


class TilesData:

    def __init__(self, test_paths, data_paths, train_size=0.8, seed=SEED):
        if not isinstance(test_paths, list):
            test_paths = [test_paths]

        if not isinstance(data_paths, list):
            test_paths = [data_paths]

        self.splits_df = pd.DataFrame()
        self.test_paths = test_paths
        self.data_paths = data_paths
        self.train_size = train_size
        self.seed = seed

    def load_data(self, max_data=650):
        x_test, y_test = [], []
        x_data, y_data = [], []
        test_mosaics = []
        MAX_DATA = max_data
        counts = np.zeros( num_classes )

        # iterate over the test data first
        for path in self.test_paths:
            mosaics = [ m for m in os.listdir(path) 
                                                if 'COMP' in m 
                                                or 'AI' in m
                                                or 'CON' in m
                                                or 'PRG' in m ]

            for mosaic_folder in mosaics:
                test_mosaics.append(mosaic_folder)
                for cls in all_classes:
                    class_path = os.path.join(path, mosaic_folder, cls)
                    for tile in os.listdir(class_path):
                        tile_path = os.path.join(class_path, tile)

                        label = class_to_id[cls]     # get the id of the class
                        x_test.append( tile_path )
                        y_test.append( label )
                        self.splits_df = self.splits_df.append({ 'phase': 'test', 'split': 0, 'x': tile_path, 'y': label, 'tile': tile }, ignore_index=True)


        # iterate over the training data
        for path in self.data_paths:
            mosaics = [ m for m in os.listdir(path) 
                                                if ('COMP' in m 
                                                    or 'AI' in m
                                                    or 'CON' in m
                                                    or 'PRG' in m)
                                                    and m not in test_mosaics ]

            for mosaic_folder in mosaics:
                for cls in all_classes:
                    class_path = os.path.join(path, mosaic_folder, cls)
                    if not os.path.exists(class_path):
                        continue
                    for tile in os.listdir(class_path):
                        tile_path = os.path.join(class_path, tile)

                        label = class_to_id[cls]     # get the id of the class

                        counts[label] += 1
                        if cls == 'mesh' and counts[label] > MAX_DATA:
                            continue

                        x_data.append( tile_path )
                        y_data.append( label )

        # validate there is no overlap between test and training
        for x in x_test:
            assert( x not in x_data )
        for x in x_data:
            assert( x not in x_test )

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_data, y_data = np.array(x_data), np.array(y_data)

        splits = StratifiedShuffleSplit(n_splits=1, train_size=int(len(x_data)*self.train_size), random_state=self.seed)

        for train_index, test_index in splits.split(x_data, y_data):
            x_train, y_train = x_data[train_index], y_data[train_index]
            x_valid, y_valid = x_data[test_index], y_data[test_index]

        for phase, x_list, y_list in zip(['train', 'valid'], [x_train, x_valid], [y_train, y_valid]):

            for x, y in zip(x_list, y_list):
                tile_name = x.split('/')[-1]
                self.splits_df = self.splits_df.append({ 'phase': 'test', 'x': x, 'y': y, 'tile': tile_name }, ignore_index=True)

    
        assert(self.splits_df.duplicated(['x']).any() == False)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

    def save_df_as_csv(self, name, path):
        self.splits_df.to_csv(os.path.join(path, name))


def get_transformer(phase):
    if phase == 'train':
        return A.Compose([
            A.Rotate(360, p=1),
            A.Resize(height=tile_height+30, width=tile_width+30),
            A.RandomCrop(height=tile_height, width=tile_width),
            A.Flip(p=0.9),
            # A.ElasticTransform(alpha_affine=20, p=0.8),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(max_holes=14, max_height=20, max_width=20, min_holes=8, min_height=8, min_width=8, p=0.9),
            A.GaussNoise(p=0.7),
            # A.MotionBlur(blur_limit=(3, 10), p=0.7),
            # A.RandomGridShuffle(grid=(3, 3), p=0.8),

            A.Normalize(p=1),
        ])

    if phase == 'mid_test':
        return A.Compose([
            A.Rotate(360, p=1),
            A.Resize(height=tile_height+30, width=tile_width+30),
            A.RandomCrop(height=tile_height, width=tile_width),
            A.Flip(p=0.8),
            A.ElasticTransform(alpha_affine=20, p=0.8),
            A.RandomBrightnessContrast(p=0.8),
            A.CoarseDropout(max_holes=14, max_height=14, max_width=14, min_holes=8, min_height=8, min_width=8, p=0.9),
            A.GaussNoise(p=0.7),
            A.MotionBlur(blur_limit=(3, 10), p=0.7),
            A.RandomGridShuffle(grid=(4, 4), p=0.8),

            A.Normalize(p=1),
        ])

    return A.Compose([
        A.Resize(height=tile_height, width=tile_width),
        A.Normalize(p=1),
    ])


class TilesDataset(Dataset):

    def __init__(self, x_data, y_data, transforms=None, return_name=False):
        self.x = x_data
        self.y = y_data
        self.transforms = transforms
        self.return_name = return_name

    def __getitem__(self, idx):
        img_path = self.x[idx]          #   get image path
        label = self.y[idx]          #   get image labels

        try:
            img = cv2.imread(img_path)  #   read image
        except:
            print(img_path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)             # img = np.rollaxis(img, -1, 0)
        
        if self.return_name:
            return torch.from_numpy(img).float().unsqueeze(0), label, img_path
        return torch.from_numpy(img).float().unsqueeze(0), label

    def __len__(self):
        return len(self.x)

    def get_weights(self):
        weights = torch.zeros( num_classes )     #   0 for each possible label
        for i in range(num_classes):
            weights[i] = np.sum(self.y == i)
        weights = 1 - (weights / sum(weights))
        # return (weights) / sum(weights)
        return weights / sum(weights)

    def export(self, path):
        df = pd.DataFrame()
        for x, y in zip( self.x, self.y ):
            df = df.append({ 'x':x, 'y':y }, ignore_index=True)
        df.to_csv(path)


def get_loaders(test_path, data_path, batch_size=8, num_workers=4, train_size=0.8, return_name=False, seed=SEED, overfit=False):

    data_handler = TilesData(test_path, data_path, train_size=train_size, seed=seed)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data_handler.load_data()

    if overfit:
        indices = []
        counts = np.zeros( num_classes )
        for i in range(len(y_train)):
            if counts[ y_train[i] ] >= 25:
                continue
            indices.append(i)
            counts[ y_train[i] ] += 1

        x_train = x_train[indices]
        y_train = y_train[indices]

        x_valid = x_train
        y_valid = y_train

    

    datasets = {
        phase: TilesDataset(x, y, get_transformer(phase), return_name=return_name)
                for phase, x, y in zip( ['train', 'valid', 'test'], [x_train, x_valid, x_test], [y_train, y_valid, y_test] )
    }

    del x_train, y_train, x_valid, y_valid, x_test, y_test, data_handler

    return {
        phase: DataLoader(
            dataset =       datasets[phase],
            batch_size =    batch_size,
            num_workers =   num_workers,
            shuffle =       phase != 'test',
        ) for phase in ['train', 'valid', 'test']
    }


def test_data():
    test_path = '/home/linuxu/Desktop/testset2'
    main_path = '/home/linuxu/Desktop/miriam'
    all_paths = [ os.path.join(main_path, folder, 'separated_tiles') for folder in os.listdir(main_path) ]

    data_handler = TilesData(test_path, all_paths)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data_handler.load_data()

    for phase, x_phase, y_phase in zip( ['train', 'valid', 'test'], [x_train, x_valid, x_test], [y_train, y_valid, y_test] ):
        print('Validating', phase, 'data')
        for x, y in zip( x_phase, y_phase ):
            class_name = x.split('/')[-2]
            assert class_to_id[class_name] == y, x
            if phase == 'train':
                assert x not in x_valid and x not in x_test, x
            elif phase == 'valid':
                assert x not in x_train and x not in x_test, x
            else:
                assert x not in x_train and x not in x_valid, x
            del x, y
        del phase, x_phase, y_phase

    del x_train, y_train, x_valid, y_valid, x_test, y_test, data_handler


def test_dataset():
    test_path = '/home/linuxu/Desktop/testset2'
    main_path = '/home/linuxu/Desktop/miriam'
    all_paths = [ os.path.join(main_path, folder, 'separated_tiles') for folder in os.listdir(main_path) ]

    data_handler = TilesData(test_path, all_paths)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data_handler.load_data()

    datasets = {
        phase: TilesDataset(x, y, get_transformer(phase), return_name=True)
                for phase, x, y in zip( ['train', 'valid', 'test'], [x_train, x_valid, x_test], [y_train, y_valid, y_test] )
    }

    print('Validating datasets')
    for phase in datasets:
        for i in tqdm( range(len( datasets[phase] )), desc=phase ):
            x, y, tile_path = datasets[phase][i]
            class_name = tile_path.split('/')[-2]
            assert class_to_id[class_name] == y, tile_path
            if phase == 'train':
                assert tile_path not in datasets['valid'].x and tile_path not in datasets['test'].x
            elif phase == 'valid':
                assert tile_path not in datasets['train'].x and tile_path not in datasets['test'].x
            else:
                assert tile_path not in datasets['train'].x and tile_path not in datasets['valid'].x
            del x, y, tile_path
        print('samples', [ sum(datasets[phase].y == i) for i in range(num_classes) ])
        print('wights', datasets[phase].get_weights())
        from sklearn.utils import class_weight
        print(class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(datasets[phase].y),
                                                 y=datasets[phase].y))
    del datasets, x_train, y_train, x_valid, y_valid, x_test, y_test, data_handler


def sanity_test():
    test_data()
    test_dataset()

if __name__ == '__main__':
    sanity_test()