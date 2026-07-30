import torch
import pandas as pd
import random
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data import class_to_conf_mat



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def create_env(path):
    if not os.path.exists(path):
        os.mkdir(path)
    paths = ['logs', 'models', 'media']
    for p in paths:
        sub_path = os.path.join(path, p)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)


def norm(img):
    img = np.array(img, dtype=np.float32)
    img -= img.min()
    img /= img.max()
    return img


def calc_conf_mat_img(y_true, y_pred, phase=None, norm=None):

    cm = confusion_matrix(y_true, y_pred, labels=[ class_to_conf_mat[cls] for cls in class_to_conf_mat ], normalize=norm)

    fig = plt.figure(figsize=(10,7))
    heatmap_fmt = '.2g' if norm else 'g'
    vmax = 1 if norm else None
    ax = sns.heatmap(cm, annot=True, fmt=heatmap_fmt, vmax=vmax, cmap=sns.color_palette("rocket_r", as_cmap=True), annot_kws={"size": 20})
    ax.set_xlabel('Predicted Labels', fontsize=20)
    ax.set_ylabel('True Labels', fontsize=20)
    ax.xaxis.set_ticklabels(class_to_conf_mat.keys(), rotation=45, fontsize=14)
    ax.yaxis.set_ticklabels(class_to_conf_mat.keys(), rotation=0, fontsize=14)
    if phase is not None:
        plt.title('{} Confusion Matrix'.format(phase.capitalize()), fontsize=25)
    fig.tight_layout()
    plt.close('all')
    return fig