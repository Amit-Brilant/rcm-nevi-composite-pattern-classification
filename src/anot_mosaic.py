from utils import norm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from data import *

data_paths = [
    '/home/linuxu/Desktop/miriam/1st - 1.11.21/separated_tiles',
    '/home/linuxu/Desktop/miriam/2nd - 30.12.21/separated_tiles',
    '/home/linuxu/Desktop/miriam/3rd - 24.1.22/separated_tiles',
    '/home/linuxu/Desktop/miriam/4th - 27.2.22/separated_tiles',
]
save_path = '/home/linuxu/Desktop/project/5_classes/anot_mosaics/'
TILE_HEIGHT, TILE_WIDTH = 100, 100
tiles_counter = [0 for i in range(8)]
REAL_MOSAIC = False


def get_numbers_from_tile(tile_name):
    tile_name = tile_name.split('/')[-1].split('.')[0]
    splited_name = tile_name.split('_')

    h_section, w_section = splited_name[1], splited_name[2]

    y, h = h_section.split('-')
    x, w = w_section.split('-')

    return int(y)-1, int(x)-1, int(h), int(w)


def get_tile_index(tile_name):
    y, x, h, w = get_numbers_from_tile(tile_name)
    return y, x


def get_mosaic_dimentions(tile_name):
    y, x, h, w = get_numbers_from_tile(tile_name)
    return h, w


def find_mosaic_dimentions(mosaic_path):
    for itm in os.listdir(mosaic_path):
        if os.path.isdir(os.path.join(mosaic_path, itm)):
            for tile in os.listdir(os.path.join(mosaic_path, itm)):
                return get_mosaic_dimentions(tile)
        elif os.path.isfile(os.path.join(mosaic_path, itm)):
            return get_mosaic_dimentions(itm)
    return 0, 0


def anot_tile(tile_path, classes):

    alpha = 0.6
    colors = {
        class_to_id['clod']: [255, 0, 0],
        class_to_id['mesh']: [0, 255, 0],
        class_to_id['ring']: [0, 0, 255],
        class_to_id['clod_mesh']: [255, 255, 255],
        class_to_id['none']: [0, 0, 0]
    }

    tile = cv2.imread(tile_path)                      #   read image
    tile = cv2.resize(tile, (TILE_HEIGHT, TILE_WIDTH))
    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

    color_map = np.zeros(tile.shape, np.uint8)

    if classes == class_to_id['none']:
        return tile, tile

    if classes == -1:
        x_tile = tile.copy()
        x_tile = cv2.line(x_tile, pt1=(0, 0), pt2=(TILE_HEIGHT, TILE_WIDTH), color=(0, 0, 0), thickness=2)
        x_tile = cv2.line(x_tile, pt1=(TILE_HEIGHT, 0), pt2=(0, TILE_WIDTH), color=(0, 0, 0), thickness=2)
        x_alpha = 0.6
        x_tile = cv2.addWeighted(x_tile, 1 - x_alpha, tile, x_alpha, 0, x_tile)
        return x_tile, tile

    tile_color = colors[ class_to_id[classes] ]

    color_map = cv2.rectangle(img=color_map, pt1=(0, 0), pt2=(color_map.shape[0], color_map.shape[1]), color=tile_color, thickness=cv2.FILLED)

    color_tile = tile.copy()
    mask = color_map.astype(bool)
    color_tile[mask] = cv2.addWeighted(tile, alpha, color_map, 1 - alpha, 0)[mask]
    color_tile = cv2.cvtColor(color_tile, cv2.COLOR_RGB2BGR)
    return color_tile, tile


def anot_mosaic(mosaic_path, h_mosaic, w_mosaic, h_tile, w_tile):
    mos = np.zeros((h_mosaic*h_tile, w_mosaic*w_tile, 3))
    reg_mos = np.zeros((h_mosaic*h_tile, w_mosaic*w_tile, 3))

    not_tagged_tiles = [ os.path.join(mosaic_path, f) for f in os.listdir(mosaic_path) if os.path.isfile(os.path.join(mosaic_path, f)) ]

    for tile in not_tagged_tiles:
        colored_tile, regular_tile = anot_tile(tile, -1)
        y, x = get_tile_index(tile)
        
        h_start, w_start = y * h_tile, x * w_tile
        h_end, w_end = h_start + h_tile, w_start + w_tile

        mos[ h_start : h_end, w_start : w_end, : ] = colored_tile
        if REAL_MOSAIC:
            reg_mos[ h_start : h_end, w_start : w_end, : ] = regular_tile

    sub_dirs = [ f for f in os.listdir(mosaic_path) if os.path.isdir(os.path.join(mosaic_path, f)) ]
    for cls in sub_dirs:
        class_path = os.path.join(mosaic_path, cls)
        for tile in os.listdir(class_path):
            tile_path = os.path.join(class_path, tile)
            colored_tile, regular_tile = anot_tile(tile_path, cls)

            y, x = get_tile_index(tile)
            
            h_start, w_start = y * h_tile, x * w_tile
            h_end, w_end = h_start + h_tile, w_start + w_tile

            mos[ h_start : h_end, w_start : w_end, : ] = colored_tile
            if REAL_MOSAIC:
                reg_mos[ h_start : h_end, w_start : w_end, : ] = regular_tile


    return mos, reg_mos


if __name__ == '__main__':

    for data_path in data_paths:
        mosaics_only = [ m for m in os.listdir(data_path) if 'COMP' in m ] # filter out folders that are not mosaics

        for i, mosaic_name in enumerate(tqdm(mosaics_only)):
        
            mosaic_path = os.path.join(data_path, mosaic_name)
            mos_h, mos_w  = find_mosaic_dimentions(mosaic_path)
            if mos_h == 0 or mos_w == 0:
                print(mosaic_name)
                continue

            mosaic_anot, reg_mosaic = anot_mosaic(mosaic_path, mos_h, mos_w, TILE_HEIGHT, TILE_WIDTH)

            if not os.path.exists(save_path):
                os.mkdir(save_path)
            mos_save_path = save_path + mosaic_name + '_anot.bmp'
            cv2.imwrite(mos_save_path, mosaic_anot)

            if REAL_MOSAIC:
                mos_save_path = save_path + mosaic_name + '.bmp'
                cv2.imwrite(mos_save_path, reg_mosaic)

        