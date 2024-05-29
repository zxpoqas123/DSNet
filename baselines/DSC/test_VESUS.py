import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import soundfile as sound
import datetime
import sys, subprocess
import math
import random
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import logging

from data import Mydataset
from model import BLSTM
#from augment_cpu import Spectrogram, MelScale, MelSpectrogram, AxisMasking, FrequencyMasking, TimeMasking, AddNoise, Crop, TimeStretch, PitchShift, Deltas_Deltas, TimeShift
#from augment_cpu import Spectrogram, MelScale, MelSpectrogram, AxisMasking, FrequencyMasking, TimeMasking, AddNoise, Crop, TimeStretch, PitchShift, Deltas_Deltas, TimeShift
from utils import mixup_data, mixup_criterion
import argparse
from sklearn.metrics import confusion_matrix,classification_report,recall_score
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def setup_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(2021)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)                                                                                                                                                                                     
    logger.addHandler(fh)                                                                                                                                                                                          
                                                                                                                                                                                                                   
    sh = logging.StreamHandler()                                                                                                                                                                                   
    sh.setFormatter(formatter)                                                                                                                                                                                     
    logger.addHandler(sh)                                                                                                                                                                                          
                                                                                                                                                                                                                   
    return logger

def pad_collate(batch):
    (xx, y1, y2) = zip(*batch)
    x_lens = torch.tensor([x.shape[0] for x in xx])
    x_lens, perm_idx = x_lens.sort(descending=True)
    xx_pad = pad_sequence(xx, batch_first=True)
    xx_pad = xx_pad[perm_idx]
    xx_pack = pack_padded_sequence(xx_pad,x_lens,batch_first=True)
    #y_emo = torch.tensor([y[0] for y in yy])[perm_idx]
    #y_emo = torch.tensor(list(y1))
    #y_spk = torch.tensor([y[0] for y in yy])[perm_idx]
    y_emo = torch.tensor(list(y1))[perm_idx]
    y_spk = torch.tensor(list(y2))[perm_idx]
    return xx_pack, y_emo, y_spk

def test(model, device, val_loader, criterion, logger, target_names):
    model = model.to(device)
    model.eval()                                                                                                                                                                                                   
    test_loss = 0
    correct = 0
    pred_all = np.array([],dtype=np.long)
    true_all = np.array([],dtype=np.long)                                                                                                                                                                                                                      
    #aug = Augment(False).to(device)
                                                                                                                                                                                                                   
    with torch.no_grad():
        for spec, label, _ in tqdm(val_loader):
            spec, label = spec.to(device), label.to(device)                    
            #mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, f_min=0.0, f_max=sr/2, n_mels=n_mels)(wav)
            #mel_specgram = torch.log2(mel_specgram+1e-8)
            #mel_specgram, label = mel_specgram.to(device), label.to(device)
            #output = model(mel_specgram)
            #spec = aug(wav)
            output = model(spec)                                                                                                                                                                          
            #test_loss += F.nll_loss(output, label, reduction='sum').item()
            test_loss += criterion(output, label).item()                                                                                                                                                           
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            pred = output.data.max(1)[1].cpu().numpy()  
            true = label.data.cpu().numpy()            
            pred_all = np.append(pred_all,pred)
            true_all = np.append(true_all,true)                                                                                                                                                          
                                                                                                                                                                                                                   
    test_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)
                                                                                                                                                                                                                   
#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#        test_loss, correct, len(val_loader.dataset),
#        100. * correct / len(val_loader.dataset)))
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), acc))

    con_mat = confusion_matrix(true_all,pred_all)
    cls_rpt = classification_report(true_all,pred_all,target_names=target_names,digits=3)
    UA = recall_score(true_all,pred_all,average='macro')
    WA = recall_score(true_all,pred_all,average='weighted')

    logger.info('Confusion Matrix:\n{}\n'.format(con_mat))
    logger.info('Classification Report:\n{}\n'.format(cls_rpt))         
                                                                                                                                                                                                                   
    return UA,WA

def test_epoch(model_root, fold, logger, model_name, max_len):   
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset_type = "VESUS"
    train_set = Mydataset(dataset_type=dataset_type, mode='train', fold=fold) 
    dev_set = Mydataset(dataset_type=dataset_type, mode='val', fold=fold)
    test_set = Mydataset(dataset_type=dataset_type, mode='test', fold=fold)  

    model = BLSTM(feature_size=512, emotion_cls=train_set.NumClasses, speaker_cls=train_set.NumSpeakers, h_dims=128, MAF_dims=64, dropout=0.25)
    ckpt = torch.load(model_root)
    model.load_state_dict(ckpt)                                                                                                                                                            

    target_names = test_set.ClassNames
    dev_loader = DataLoader(dev_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, collate_fn=pad_collate)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, collate_fn=pad_collate)
                                                                                                                                                                                                                   
    #criterion_test = FocalLoss(alpha=0.25, gamma=2., reduction='sum')
    criterion_test = nn.NLLLoss(reduction='sum')
                                                                                                                                                                                                                   
    #logger = get_logger('log/exp.log')
    logger.info('testing {}'.format(model_name))
    UA_dev,WA_dev = test(model, device, dev_loader, criterion_test, logger, target_names)
    UA_test,WA_test = test(model, device, test_loader, criterion_test, logger, target_names)

    return UA_dev,WA_dev,UA_test,WA_test                                                                                                                                            

def main(fold_list, fold_root, max_len):
    subprocess.check_call(["cp", "test.py", fold_root])
    #logpath = os.path.join(fold_root, "{}_test.log".format(condition))
    logpath = os.path.join(fold_root, "test.log")
    logger = get_logger(logpath)
    UA_dev_ls = []
    WA_dev_ls = []
    UA_test_ls = []
    WA_test_ls = []    
    for fold in fold_list:
        logger.info('fold: {}'.format(fold))
        root = os.path.join(fold_root, 'fold_{}'.format(fold))
        #sub_fold = list(filter(lambda n:n.startswith('condition_'+condition), os.listdir(root)))[0]
        #model_name = list(filter(lambda n:n.startswith('bestmodel-'), os.listdir(os.path.join(root,sub_fold))))[0]
        model_ls = list(i for i in os.listdir(root) if i.startswith('best_epoch'))
        epoch_ls = [int(i[:-3].split('_')[-1]) for i in model_ls]
        best_model = model_ls[np.argmax(epoch_ls)]
        model_root = os.path.join(root,best_model)
        UA_dev,WA_dev,UA_test,WA_test = test_epoch(model_root, fold, logger, best_model, max_len)
        #test_epoch(model_root, condition, fold, logger, best_model)
        UA_dev_ls.append(UA_dev)
        WA_dev_ls.append(WA_dev)
        UA_test_ls.append(UA_test)
        WA_test_ls.append(WA_test)            
    logger.info('UA_dev list: {}'.format(UA_dev_ls))
    logger.info('WA_dev list: {}'.format(WA_dev_ls))
    logger.info('UA_test list: {}'.format(UA_test_ls))
    logger.info('WA_test list: {}'.format(WA_test_ls))

    logger.info('UA_dev avg: {}'.format(np.mean(UA_dev_ls)))
    logger.info('WA_dev avg: {}'.format(np.mean(WA_dev_ls)))
    logger.info('UA_test avg: {}'.format(np.mean(UA_test_ls)))
    logger.info('WA_test avg: {}'.format(np.mean(WA_test_ls)))

if __name__ == '__main__':
    #condition = 'impro'
    fold_list = list(range(10))
    max_len = 6
    fold_root = '/home/chenchengxin/cross_corpus_new/asr_embedding_version/RNN+MLP_half/seed_2021+RNN_MLP_half/VESUS+lr1e-3+batch_size32+CE+Adam+4s+40words'
    main(fold_list, fold_root, max_len)