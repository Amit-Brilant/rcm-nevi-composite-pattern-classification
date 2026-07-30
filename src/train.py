import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.optim as optim
import torchmetrics
from kornia.color import bgr_to_grayscale
from scheduler import CosineDecayWithWarmUpScheduler

import os
import sys
from datetime import datetime
from tqdm import tqdm
import wandb


from data import TilesDataset, get_loaders, get_transformer, all_classes, class_to_id, class_to_conf_mat, num_classes, SEED
from utils import *
from model import TilesModel
from cost_sensitive_loss import CostSensitiveRegularizedLoss
from kornia.losses.focal import FocalLoss as KFocalLoss

learning_rate = 1e-6
num_epochs = 100
batch_size = 32
train_size = 0.75
freeze = False
drop_prob = 0.85
model_name = 'convnext_tiny'

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False
# DEBUG = True
overfit = False
# overfit = True

punish_matrix = np.array([
    [0,     1,      2,     2,      2],        # none
    [1,     0,      1,     2,      2],        # ring
    [2,     1,      0,     2,      2],        # mesh (mesh / mesh_ring)
    [2,     2,      2,     0,      1],        # compo (clod_mesh / clod_mesh_ring)
    [2,     2,      2,     1,      0],        # clod / clod_ring
#   none   ring   mesh/   compo    clod/
#               mesh_ring         clod_ring
], dtype=np.float64)



def test_model(model, loader, criterion, augmentation):
    ground_truth = { 'test': [] }
    probability = { 'test': [] }
    losses = { 'test': 0.0 }

    model.eval()
    torch.autograd.set_grad_enabled(False)

    running_loss, samples = 0.0, 0

    # iterate over batches
    for idx, batch in tqdm(    
        enumerate(loader), total=len(loader), desc='test'
    ):
        #   get items from batch and move to cuda
        (images, truth) = batch
        images = images.float().to(device)
        truth = truth.long().to(device)

        outputs = model(images)
        loss = criterion(outputs, truth)

        samples             += images.size(0)
        running_loss        += loss.item()

        ground_truth['test'] += truth.cpu().detach().numpy().tolist()
        probability['test'] += outputs.cpu().detach().numpy().tolist()
    
        del images, truth, outputs, loss
        torch.cuda.empty_cache()

    losses = { 'test': running_loss / len(loader) }
    return ground_truth, probability, losses


def test_time_aug_model(model, loader, criterion, augmentation, num_tta=5):
    from data import TilesDataset
    dataset = TilesDataset(loader.dataset.x, loader.dataset.y, augmentation)
    ground_truth = { 'test': [] }
    probability = { 'test': [] }
    losses = { 'test': 0.0 }

    model.eval()
    torch.autograd.set_grad_enabled(False)

    running_loss, samples = 0.0, 0

    # iterate over batches
    for idx in tqdm(    
        range(len(dataset)), total=len(dataset), desc='test'
    ):
        
        tta_output = torch.zeros(num_classes).to(device)
        single_loss = 0

        for i in range(num_tta):
            (images, truth) = dataset[idx]
            images = images.float().to(device).unsqueeze(0)
            truth = torch.LongTensor([truth]).to(device)

            outputs = model(images)
            loss = criterion(outputs, truth)

            predicted = F.softmax(outputs, dim=1).argmax(dim=1)[0]
            tta_output[ predicted ] += 1
            single_loss += loss.item()

            del images, outputs, loss, predicted
            torch.cuda.empty_cache()

        samples             += 1
        running_loss        += single_loss / num_tta

        ground_truth['test'] += truth.cpu().detach().numpy().tolist()
        probability['test'] += torch.div(tta_output, num_tta).unsqueeze(0).cpu().detach().numpy().tolist()

        del tta_output
        torch.cuda.empty_cache()

    losses = { 'test': running_loss / len(dataset) }
    return ground_truth, probability, losses



def train_valid_one_epoch(model, loaders, criterion, optimizer):
    ground_truth = { 'train': [], 'valid': [] }
    probability = { 'train': [], 'valid': [] }
    losses = { 'train': 0.0, 'valid': 0.0 }

    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        torch.autograd.set_grad_enabled(phase == 'train')
        running_loss, samples = 0.0, 0
        
        # iterate over batches
        for idx, batch in tqdm(    
            enumerate(loaders[phase]), total=len(loaders[phase]), desc=phase
        ):
            #   get items from batch and move to cuda
            (images, truth) = batch
            images = images.float().to(device)
            truth = truth.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, truth)

            if phase == 'train':
                loss.backward()
                optimizer.step()
            
            samples             += images.size(0)
            running_loss        += loss.item()

            ground_truth[phase] += truth.cpu().detach().numpy().tolist()
            probability[phase] += outputs.cpu().detach().numpy().tolist()

            del images, truth, loss, outputs
            torch.cuda.empty_cache()

        losses[phase] = running_loss / len(loaders[phase])
    
    return ground_truth, probability, losses


def calc_metrics(phase, ground_truth, probability, num_classes):
    ground_truth = torch.tensor(ground_truth)
    probability = torch.tensor(probability)
    predicted = F.softmax(probability, dim=1).argmax(dim=1)

    met_dict = dict()
    met_dict['accuracy'] = torchmetrics.functional.accuracy(probability, ground_truth, average='micro', num_classes=num_classes)
    met_dict['f1'] = torchmetrics.functional.f1_score(probability, ground_truth, average='micro', num_classes=num_classes)
    met_dict['conf_mat'] = calc_conf_mat_img(ground_truth, predicted, phase)
    met_dict['conf_mat_norm_true'] = calc_conf_mat_img(ground_truth, predicted, phase, norm='true')
    met_dict['conf_mat_norm_pred'] = calc_conf_mat_img(ground_truth, predicted, phase, norm='pred')
    met_dict['conf_mat_norm_all'] = calc_conf_mat_img(ground_truth, predicted, phase, norm='all')

    del ground_truth, probability, predicted

    return met_dict


def calc_metrics_new(phase, ground_truth, probability, num_classes, punish_mat):
    ground_truth = torch.tensor(ground_truth)
    probability = torch.tensor(probability)
    predicted = F.softmax(probability, dim=1).argmax(dim=1)

    cm = confusion_matrix(ground_truth, predicted, labels=[ class_to_conf_mat[cls] for cls in class_to_conf_mat ])
    accuracy, recall, precision, f1_score = [], [], [], []

    for cls in range(num_classes):
        eps = 1e-8
        tp, tn, fp, fn = 0, 0, 0, 0

        class_samples = torch.sum(ground_truth == cls)
        tp = cm[ cls, cls ]

        for i in range(num_classes):
            if i == cls:
                continue
            for j in range(num_classes):
                if j == cls:
                    continue
                tn += cm[ i, j ]

        for i in range(num_classes):
            if i == cls:
                continue
            fn += cm[ cls, i ] * punish_mat[ cls, i ]

        for i in range(num_classes):
            if i == cls:
                continue
            fp += cm[ i, cls ] * punish_mat[ cls, i ]
                    
        acc = (tp + tn) / (tp + tn + fp + fn + eps)
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = (2 * prec * rec) / (prec + rec + eps)

        accuracy.append( acc * class_samples )
        precision.append( prec * class_samples )
        recall.append( rec * class_samples )
        f1_score.append( f1 * class_samples )


    met_dict = dict()
    met_dict['accuracy'] = np.sum(accuracy) / ground_truth.size(0)
    met_dict['precision'] = np.sum(precision) / ground_truth.size(0)
    met_dict['recall'] = np.sum(recall) / ground_truth.size(0)
    met_dict['f1'] = np.sum(f1_score) / ground_truth.size(0)
    met_dict['conf_mat'] = calc_conf_mat_img(ground_truth, predicted, phase)
    met_dict['conf_mat_norm_true'] = calc_conf_mat_img(ground_truth, predicted, phase, norm='true')
    met_dict['conf_mat_norm_pred'] = calc_conf_mat_img(ground_truth, predicted, phase, norm='pred')
    met_dict['conf_mat_norm_all'] = calc_conf_mat_img(ground_truth, predicted, phase, norm='all')

    del ground_truth, probability, predicted

    return met_dict


def train_loop(model, loaders, num_epochs, criterion, optimizer, scheduler, run_path, augmentation, tta=False):
    model_loss_save_path = '{}/models/model_loss_checkpoint.pth'.format(run_path)
    model_f1_save_path = '{}/models/model_f1_checkpoint.pth'.format(run_path)

    model.save(model_loss_save_path)
    model.save(model_f1_save_path)

    best_valid_loss = float('inf')
    best_valid_f1 = float('-inf')
    start_epoch = 1
    end_epoch = start_epoch + num_epochs

    #   iterate over the epochs
    for epoch in range(start_epoch, end_epoch):
        print('Epoch {:3d} of {}:'.format(epoch, end_epoch-1), flush=True)

        metrics_dict = { 'train': {}, 'valid': {}, 'test': {} }

        ground_truth, probability, losses = train_valid_one_epoch(model, loaders, criterion, optimizer)
        if tta:
            test_ground_truth, test_probability, test_losses = test_time_aug_model(model, loaders['test'], criterion, get_transformer('mid_test'))
        else:
            test_ground_truth, test_probability, test_losses = test_model(model, loaders['test'], criterion, augmentation['test'])


        ground_truth.update(test_ground_truth)
        probability.update(test_probability)
        losses.update(test_losses)

        try:
            lr = scheduler.get_last_lr()[0]
        except:
            lr = [ group['lr'] for group in optimizer.param_groups ][0]

        metrics_dict['learning_rate'] = lr
        metrics_dict['epoch'] = epoch

        validation_loss = losses['valid']

        #   if there is a scheduler, perform step based on the validation loss
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_loss)  
            else:
                scheduler.step()

        print_str = ''
        phases = ['train', 'valid', 'test']
        for phase in phases:
            if loaders[phase]:
                print_str += '{}:\t'.format(phase)

                metrics_dict[phase] = calc_metrics_new(phase, ground_truth[phase], probability[phase], num_classes, punish_matrix)
                metrics_dict[phase]['loss'] = losses[phase]

                for key in metrics_dict[phase]:
                    if 'conf_mat' in key:
                        continue
                    val = metrics_dict[phase][key]
                    print_str += '{}={:.5f} '.format(key, val)
                print_str += '\n'

                metrics_dict[phase]['ground_truth'] = ground_truth[phase]
                metrics_dict[phase]['outputs'] = probability[phase]

                if not DEBUG:
                    for key in metrics_dict[phase]:
                        if 'conf_mat' in key:
                            metrics_dict[phase][key] = wandb.Image(metrics_dict[phase][key])
        
        print(print_str)
        if not DEBUG:
            wandb.log( metrics_dict )

        #   if validation loss is better, save model chechpoint
        if validation_loss < best_valid_loss:
            best_valid_loss = validation_loss
            model.save(model_loss_save_path)

        validation_f1 = metrics_dict['valid']['f1']
        if validation_f1 > best_valid_f1:
            best_valid_f1 = validation_f1
            model.save(model_f1_save_path)

        del metrics_dict, ground_truth, probability, losses, test_ground_truth, test_probability, test_losses




def train_model(model_name=model_name, epochs=num_epochs, lr=learning_rate, drop_prob=drop_prob, freeze=freeze, train_size=train_size, prev_data=False, seed=SEED):
    seed_everything(seed)

    datetime_srt = datetime.today().strftime("%m-%d-%y_%H%M")
    run_path = os.path.join(sys.path[0], 'runs', datetime_srt)
    print('Generating running environment', run_path)
    create_env(run_path)

    test_path = '/home/linuxu/Desktop/testset3'
    main_path = '/home/linuxu/Desktop/miriam'
    all_paths = [ os.path.join(main_path, folder, 'separated_tiles') for folder in os.listdir(main_path) ]
    if prev_data:
        all_paths.append('/home/linuxu/Desktop/prev_normal_dataset2',)
    loaders = get_loaders(test_path, all_paths, batch_size=batch_size, train_size=train_size, overfit=overfit)

    for phase in loaders:
        loaders[phase].dataset.export( os.path.join(run_path, 'logs', '{}_data.csv'.format(phase)) )

    model = TilesModel(name=model_name, drop_prob=drop_prob, freeze=freeze)
    
    criterion = CostSensitiveRegularizedLoss(n_classes=num_classes, base_loss='focal_loss')
    M = (punish_matrix / punish_matrix.max())
    criterion.M = torch.from_numpy(M)

    params = [ p for p in model.parameters() if p.requires_grad ]
    optimizer = optim. AdamW(params, lr=lr)

    scheduler_dict = { 
        'step_per_epoch': 1,
        'init_warmup_lr': lr,
        'warm_up_steps': 2,
        'max_lr': 1e-4,
        'min_lr': 1e-8,
        'num_step_down': 30,
        'num_step_up': 1,
        'max_lr_decay': 'Half',
        'gamma': 0.5,
        'min_lr_decay': 'Exp',
        'alpha': 0.6
    }
    scheduler = CosineDecayWithWarmUpScheduler(optimizer, **scheduler_dict)

    model = model.to(device)
    criterion = criterion.to(device)

    augmentation = {
        phase: get_transformer(phase) for phase in loaders
    }

    hyperparams = {
        'seed': SEED,
        'learning_rate': lr,
        'epochs': num_epochs,
        'batch_size': batch_size,
        'train_size': train_size,
        'number_samples': len(loaders['train'].dataset) + len(loaders['valid'].dataset),
        'drop_prob': drop_prob,
        'optimizer': type(optimizer).__name__,
        'loss': type(criterion).__name__,
        'scheduler': type(scheduler).__name__ if scheduler is not None else None,
        'scheduler_dict': scheduler_dict if scheduler is not None else None,
        'num_classes': num_classes,
        'freeze': freeze,
        'prev_data': prev_data,
    }

    if not DEBUG:
        wandb.init(project="RCM-2022-Composite", entity="guykabiri", config=hyperparams, tags=[model_name]) # group=cv_fold_#
        wandb.watch(model, log_freq=1)

    train_loop(model, loaders, num_epochs, criterion, optimizer, scheduler, run_path, augmentation, tta=False)



if __name__ == '__main__':
    train_model(
        prev_data = True
    )