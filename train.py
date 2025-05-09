import os.path
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from common.commonUtils import get_classes
from common.configUtils import getConfig
from common.dataloader import ObjectDetectionDataset, frcnn_dataset_collate
from forward.process_epoch import process_epoch
from nets.frcnn import FasterRCNN
from nets.frcnn_training import FasterRCNNTrainer
from utils.callbacks import LossHistory
from utils.trainingUtils_torch import weights_init

configFilePath = 'configCpu.yaml'

cfg = getConfig(configFilePath)
logDirName = input("weights && loss's save name:")
if logDirName == "":
    logDirName = 'unset'

random.seed(0)
torch.manual_seed(0)

class_names, num_classes = get_classes(cfg.classesPath)
print('get classes success')
"""
Load model && Pretrain dict
"""
model = FasterRCNN(num_classes, anchor_scales=cfg.anchors_size, backbone=cfg.backbone, pretrained=cfg.pretrained).to(cfg.device)
print('get model success')

if not cfg.pretrained:
    weights_init(model)
if cfg.model_path != '':
    print('Load weights {}.'.format(cfg.model_path))
    model_dict = model.state_dict()
    pretrained_dict = torch.load(cfg.model_path, map_location=cfg.device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
with open(cfg.train_annotation_path) as f:
    train_lines = f.readlines()
with open(cfg.val_annotation_path) as f:
    val_lines = f.readlines()
train_Len = len(train_lines)
val_Len = len(val_lines)
one_epoch_step = train_Len // cfg.batch_size
one_epoch_step_val = val_Len // cfg.batch_size
"""
optimizer setting
"""
optimizer = optim.Adam(model.parameters(), float(cfg.lr), weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
"""
dataset setting
"""
train_dataset = ObjectDetectionDataset(train_lines, cfg.input_shape, train=True)
val_dataset = ObjectDetectionDataset(val_lines, cfg.input_shape, train=False)
train_datas = DataLoader(train_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True,
                 drop_last=True, collate_fn=frcnn_dataset_collate)
val_datas = DataLoader(val_dataset, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True,
                     drop_last=True, collate_fn=frcnn_dataset_collate)
print('get dataloader success')

loss_history = LossHistory("logs/" + logDirName)

"""
begin training
"""
train_util = FasterRCNNTrainer(model, optimizer)
min_loss = 1000
for epoch in range(cfg.training_epochs):
    save_dir = os.path.join('logs', logDirName, str(epoch) + '.pth')
    best_model_save_dir = os.path.join('logs', logDirName, 'best.pth')
    epoch_state = f'Epoch {epoch + 1}/{cfg.training_epochs}'
    model, val_loss = process_epoch(epoch, epoch_state, model, train_datas, val_datas, one_epoch_step, one_epoch_step_val, train_util, optimizer, loss_history, cfg.device)
    torch.save(model.state_dict(), save_dir)
    if val_loss < min_loss:
        torch.save(model.state_dict(), best_model_save_dir)

print('training complete')
