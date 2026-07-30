from concurrent.futures import thread
from email.policy import default
import PySimpleGUI as sg
import os
import io
import base64
import PIL.Image
import PIL.ImageOps
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import tkinter as tk


def get_screen_size():
    root = tk.Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height

screen_width, screen_height = get_screen_size()


main_path = '/home/linuxu/Desktop/miriam'

classes = np.array([    'clod', 'mesh', 'ring', 'none',
                        'clod_mesh', 'clod_ring', 'mesh_ring',
                        'clod_mesh_ring' ])

classes_to_id = {   
                    'none':         0,
                    'ring':         1,
                    'mesh':         2,
                    'mesh_ring':    2,
                    'clod_mesh':    3,
                    'clod_mesh_ring':   3,
                    'clod':         4,
                    'clod_ring':    4,       
}

class_to_class = {  'none': 'none',
                    'clod': 'clod',
                    'mesh': 'mesh',
                    'ring': 'ring',
                    'clod_mesh': 'clod_mesh',
                    'clod_ring': 'clod',
                    'mesh_ring': 'mesh',
                    'clod_mesh_ring': 'clod_mesh',
}

TILE_HEIGHT, TILE_WIDTH = 200, 200
THRESHOLD = 100
NOT_TAGGED_COLOR = 'black'


######################################################
# generate dataframe

all_mosaics = []


for folder in sorted(os.listdir(main_path)):
    folder_path = os.path.join(main_path, folder, 'separated_tiles')

    mosaics_only = [ m + ' (' + folder.split()[0] + ')' 
                        for m in os.listdir(folder_path) 
                                                if 'COMP' in m 
                                                or 'AI' in m
                                                or 'CON' in m
                                                or 'PRG' in m
                                                ] # filter out folders that are not mosaics

    mosaics_only = sorted(mosaics_only)

    all_mosaics += mosaics_only


###############################################
# annotate mosaic

def get_mosaic_tiles(mosaic_name):
    df = pd.DataFrame()

    for folder in sorted(os.listdir(main_path)):
        folder_path = os.path.join(main_path, folder, 'separated_tiles')

        mosaics_only = [ m for m in os.listdir(folder_path) 
                                                    if 'COMP' in m 
                                                    or 'AI' in m
                                                    or 'CON' in m
                                                    or 'PRG' in m
                                                    ] # filter out folders that are not mosaics

        mosaics_only = sorted(mosaics_only)
        
        for mosaic in mosaics_only:
            if mosaic != mosaic_name:
                continue

            mosaic_path = os.path.join(folder_path, mosaic)

            not_tagged = [ f for f in os.listdir(mosaic_path) if os.path.isfile( os.path.join(mosaic_path, f) ) ] 
            for tile in not_tagged:
                df = df.append({ 'tile': tile, 'mosaic': mosaic, 'class': 'not_tagged', 'path': os.path.join(mosaic_path, tile) }, ignore_index=True)

            tagged_folders = [ f for f in os.listdir(mosaic_path) if os.path.isdir( os.path.join(mosaic_path, f) ) ]
            for sub_dir in tagged_folders:
                if sub_dir not in classes:
                    continue
                sub_dir_path = os.path.join(mosaic_path, sub_dir)
                for tile in os.listdir(sub_dir_path):
                    df = df.append({ 'tile': tile, 'mosaic': mosaic, 'class': class_to_class[sub_dir], 'path': os.path.join(mosaic_path, sub_dir, tile) }, ignore_index=True)
    
    return df


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


def anot_tile(tile_df, color_dict):

    alpha = 0.

    colors = {
        classes_to_id['clod']: [255, 0, 0],
        classes_to_id['mesh']: [0, 255, 0],
        classes_to_id['ring']: [0, 0, 255],
        classes_to_id['clod_mesh']: [255, 255, 255],
        classes_to_id['none']: [0, 0, 0]
    }

    tile_df = {
        'class': tile_df[1],
        'path': tile_df[3],
    }

    tile = cv2.imread(tile_df['path'])                      #   read image
    tile = cv2.resize(tile, (TILE_HEIGHT, TILE_WIDTH))
    tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

    color_map = np.ones(tile.shape, np.uint8)

    if tile_df['class'] == 'not_tagged':
        if not color_dict['not_tagged']:
            return tile, tile
        x_tile = tile.copy()
        color = [0, 0, 0] if NOT_TAGGED_COLOR == 'black' else [255, 255, 255]
        x_tile = cv2.line(x_tile, pt1=(0, 0), pt2=(TILE_HEIGHT, TILE_WIDTH), color=color, thickness=3)
        x_tile = cv2.line(x_tile, pt1=(TILE_HEIGHT, 0), pt2=(0, TILE_WIDTH), color=color, thickness=3)
        x_alpha = 0.
        annot_tile = cv2.addWeighted(x_tile, 1 - x_alpha, tile, x_alpha, 0, x_tile)
        return annot_tile, tile

    if tile_df['class'] == 'none' or not color_dict[ tile_df['class'] ]:
        color_tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        return color_tile, tile

    color_map = cv2.rectangle(img=color_map, pt1=(0, 0), pt2=(color_map.shape[0], color_map.shape[1]), color=colors[ classes_to_id[ tile_df['class'] ] ], thickness=cv2.FILLED)

    color_tile = tile.copy()
    mask = color_map.astype(bool)
    color_tile[mask] = cv2.addWeighted(tile, alpha, color_map, 1 - alpha, 0)[mask]
    color_tile = cv2.cvtColor(color_tile, cv2.COLOR_RGB2BGR)
    return color_tile, tile


def anot_mosaic(df_mos, mosaic_name, color_dict, h_tile, w_tile):

    tiles = df_mos[df_mos['mosaic']==mosaic_name]
    h_mosaic, w_mosaic = get_mosaic_dimentions(tiles.iloc[0]['tile'])
    # print(h_mosaic, w_mosaic)
    anot_mos = np.zeros((h_mosaic*h_tile, w_mosaic*w_tile, 3))
    orig_mos = np.zeros((h_mosaic*h_tile, w_mosaic*w_tile, 3))


    for tile in tiles.itertuples():
        colored_tile, orig_tile = anot_tile(tile, color_dict)
        y, x = get_tile_index(tile[4])
        
        h_start, w_start = y * h_tile, x * w_tile
        h_end, w_end = h_start + h_tile, w_start + w_tile

        anot_mos[ h_start : h_end, w_start : w_end, : ] = colored_tile
        orig_mos[ h_start : h_end, w_start : w_end, : ] = orig_tile

    perc = len(tiles[tiles['class']!='not_tagged'])*100/len(tiles)

    return anot_mos, orig_mos, perc


def create_opacity(orig_mos, anot_mos, opacity):
    return cv2.addWeighted(anot_mos, opacity, orig_mos, 1-opacity, 0)



#####################################################
# GUI

# font = 'Calibri 16'
font = 'Arial 16'


# First the window layout in 2 columns

file_list_column = [
    [
        sg.Text(key='mosaic_details', font='Arial 20', text_color='Black')
    ],
    [
        sg.Checkbox('Clod / Clod-Ring', default=True, key='clod', enable_events=True, font=font),
        sg.Checkbox('Mesh / Mesh-Ring', default=True, key='mesh', enable_events=True, font=font),
        sg.Checkbox('Ring', default=True, key='ring', enable_events=True, font=font),
    ],
    [
        sg.Checkbox('Clod-Mesh / Clod-Mesh-Ring', default=True, key='clod_mesh', enable_events=True, font=font),
    ],
    [
        # sg.Checkbox('Clod-Mesh-Ring', default=True, key='clod_mesh_ring', enable_events=True,),
        sg.Checkbox('Not Tagged', default=True, key='not_tagged', enable_events=True, font=font),
    ],
    [
        sg.Text("Not-Tagged Color:", font=font),
        sg.Radio('Black', font=font, group_id='color', enable_events=True, key='not_tagged_black', default=True),
        sg.Radio('White', font=font, group_id='color', enable_events=True, key='not_tagged_white'),
    ],
    [
        sg.Text("Opacity", font=font),
        sg.Slider(range=(0,1), default_value=0.6, resolution=.1, size=(40,10), orientation='horizontal', enable_events=True, key='opacity')
    ],
    [
        sg.Text("Tile size", font=font),
        # sg.Input(key='tile_size', default_text=TILE_WIDTH, enable_events=True, focus=False, size=(10,10)),
        sg.Slider(range=(50,300), default_value=TILE_WIDTH, resolution=50, size=(40,10), orientation='horizontal', enable_events=True, key='tile_size')
    ],
    [
        sg.Listbox(
            values=all_mosaics, enable_events=True, size=(40, 20), key='mosaic', font=font,
        )
    ],
    [
        sg.Image('~/Desktop/project/5_classes/legend.png')
    ]
]



# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)

prev_state = None

# Run the Event Loop
while True:
    event, values = window.read()
    print(event, values)
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == 'opacity' and values['mosaic']:
        img = create_opacity(anot_mos, orig_mos, values['opacity'])

    if event == 'tile_size':
        TILE_WIDTH = TILE_HEIGHT = int(values['tile_size'])

    if event == 'not_tagged_black':
        NOT_TAGGED_COLOR = 'black'

    if event == 'not_tagged_white':
        NOT_TAGGED_COLOR = 'white'

    if ((event == 'clod' or event == 'mesh' or event == 'ring' or event == 'not_tagged' or 'not_tagged' in event) and values['mosaic']) or event == 'mosaic':
        if prev_state == values:
            continue

        prev_state = values

        if 'anot_mos' in locals():
            del anot_mos
        if 'orig_mos' in locals():
            del orig_mos
        
        name_of_mosaic = values['mosaic'][0].split()[0]
        df_mos = get_mosaic_tiles(name_of_mosaic)
        anot_mos, orig_mos, perc = anot_mosaic(df_mos, name_of_mosaic, values, TILE_HEIGHT, TILE_WIDTH)

        window['mosaic_details'].update(value='{} {:.2f}% tagged'.format(name_of_mosaic, perc))

        img = create_opacity(anot_mos, orig_mos, values['opacity'])

    if 'img' in locals():
        img = PIL.Image.fromarray((img).astype(np.uint8))
        b, g, r = img.split()
        img = PIL.Image.merge("RGB", (r, g, b))
        while img.size[0] >= screen_width-THRESHOLD or img.size[1] >= screen_height-THRESHOLD:
            img = img.resize((int(img.size[0]*0.95), int(img.size[1]*0.95)))
        
    

        with io.BytesIO() as output:
            img.save(output, format="PNG")
            data = output.getvalue()

        im_64 = base64.b64encode(data)
        window["-IMAGE-"].update(data=im_64)

        del img, im_64, data


window.close()

