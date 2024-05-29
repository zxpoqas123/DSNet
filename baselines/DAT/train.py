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
import time
from data import Mydataset
from model import FocalLoss, Mymodel
from utils import mixup_data, mixup_criterion
import argparse
from sklearn.metrics import confusion_matrix,classification_report,recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser(description="SER-Baseline")
parser.add_argument('--epochs',default=100,type=int)
parser.add_argument('--batch-size',default=16,type=int)
parser.add_argument('--seed',default=2021,type=int)
parser.add_argument('--max-len',default=6, type=int)
parser.add_argument('--lr',default=1e-3,type=float)
parser.add_argument('--fold',type=int,required=True)
parser.add_argument("--root",type=str,required=True)
parser.add_argument("--feature",type=str,required=True)
parser.add_argument("--dataset-type",type=str,required=True)
parser.add_argument("--loss-type",default='Focal',type=str)
parser.add_argument("--optimizer-type",default='SGD',type=str)
parser.add_argument("--device-number",default='0',type=str)
parser.add_argument("--condition",default='all',type=str)
parser.add_argument("--backbone-type",default='CNN6',type=str)

def setup_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
    #(xx1,xx2,y) = zip(*batch)
    xx1,y,y2 = zip(*batch)
    x_lens1 = torch.tensor([x.shape[0] for x in xx1])
    x_lens1, perm_idx1 = x_lens1.sort(descending=True)
    xx_pad1 = pad_sequence(xx1, batch_first=True)
    xx_pad1 = xx_pad1[perm_idx1]
    xx_pack1 = pack_padded_sequence(xx_pad1,x_lens1,batch_first=True,enforce_sorted=False)
    '''
    x_lens2 = torch.tensor([x.shape[0] for x in xx2])
    xx_pad2 = pad_sequence(xx2, batch_first=True)
    xx_pad2 = xx_pad2[perm_idx1]
    x_lens2 = x_lens2[perm_idx1]
    xx_pack2 = pack_padded_sequence(xx_pad2,x_lens2,batch_first=True,enforce_sorted=False)
    '''
    y_emo = torch.tensor(list(y))[perm_idx1]
    y_domain = torch.tensor(list(y2))[perm_idx1]
    return xx_pack1, y_emo, y_domain

def train(model, device, train_loader, dev_loader, val_loader, criterion, optimizer, epoch, logger, epochs):
    model.train()                                                                                                                                                                                                  
    logger.info('start training')

    lr = optimizer.param_groups[0]["lr"]
    logger.info('lr: {:.5f}'.format(lr))
                                                                                                                                                                                                                   
    correct_domain = 0
    correct_emo = 0
    iter_dev=enumerate(dev_loader)
    iter_val=enumerate(val_loader)
    lambda_ = 1.0

    len_dataloader = len(train_loader)                                                                                                                                                       
    for batch, train_batch in tqdm(enumerate(train_loader)):
        try:
            dev_batch = next(iter_dev)
        except StopIteration:
            dev_batch = dev_batch
        try:
            val_batch = next(iter_val)
        except StopIteration:
            val_batch = val_batch

        spec_train, emo_label, domain_train = train_batch
        spec_train, emo_label, domain_train = spec_train.to(device), emo_label.to(device), domain_train.to(device)
        (_,(spec_dev,_,domain_dev)) = dev_batch
        (_,(spec_val,_,domain_val)) = val_batch
        spec_dev, spec_val, domain_dev, domain_val = spec_dev.to(device), spec_val.to(device), domain_dev.to(device), domain_val.to(device)
        
        if type(spec_train)==torch.Tensor:
            spec = torch.cat([spec_train,spec_dev,spec_val],0)
        else:
            spec_train,lens_train = pad_packed_sequence(spec_train,batch_first=True)
            spec_dev,lens_dev = pad_packed_sequence(spec_dev,batch_first=True)
            spec_val,lens_val = pad_packed_sequence(spec_val,batch_first=True)
            spec_all = list(spec_train) + list(spec_dev) + list(spec_val)
            spec_all_pad = pad_sequence(spec_all, batch_first=True)
            lens_all = torch.cat([lens_train,lens_dev,lens_val],0)
            spec = pack_padded_sequence(spec_all_pad,lens_all,batch_first=True,enforce_sorted=False)
            
        domain_label = torch.cat([domain_train,domain_dev,domain_val],0)

        #p = float(batch + (epoch-1) * len_dataloader) / (epochs * len_dataloader)
        #lambda_ = 2. / (1. + np.exp(-10 * p)) - 1

        optimizer.zero_grad()                                                                                                                                                                                                                                                                                                                                                                                      
        #spec = aug(wav)
        res = model(spec,lambda_)                                                                                                                                                                                       
        loss = criterion(res['emo_pred'][:len(emo_label)], emo_label) + criterion(res['domain_pred'], domain_label)
                                                                                                                                                                                                                   
        #loss_func = mixup_criterion(label_a, label_b, lam)                                                                                                                                                         
        #loss = loss_func(criterion, output)                                                                                                                                                                        
        loss.backward()
        nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], max_norm=10, norm_type=2)                                                                                                                                                                                             
        optimizer.step()

        pred_domain = res['domain_pred'].argmax(dim=1, keepdim=True)
        correct_domain += pred_domain.eq(domain_label.view_as(pred_domain)).sum().item()

        pred_emo = res['emo_pred'][:len(emo_label)].argmax(dim=1, keepdim=True)
        #correct += lam * pred.eq(label_a.view_as(pred)).sum().item() + (1 - lam) * pred.eq(label_b.view_as(pred)).sum().item()
        correct_emo += pred_emo.eq(emo_label.view_as(pred_emo)).sum().item()                                                                                                                                                                                        
        
        if batch % 20 == 0:
            logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t loss={:.5f}\t '.format(epoch , batch * len(emo_label), len(train_loader.dataset), 100. * batch / len(train_loader), loss.item()))
                                                                                                                                                                                           
    logger.info('Train set domain Accuracy: {}/{} ({:.3f}%)'.format(correct_domain, len(train_loader.dataset)+len(dev_loader.dataset)+len(val_loader.dataset), 100. * correct_domain / (len(train_loader.dataset)+len(dev_loader.dataset)+len(val_loader.dataset))))
    logger.info('Train set emotion Accuracy: {}/{} ({:.3f}%)'.format(correct_emo, len(train_loader.dataset), 100. * correct_emo / len(train_loader.dataset)))
    logger.info('finish training!')

def test(model, device, val_loader, criterion, logger, target_names):
    model.eval()                                                                                                                                                                                                   
    test_loss = 0
    correct = 0
    logger.info('testing on dev_set')
                                                                                                                                                                                                                   
    pred_all = np.array([],dtype=np.long)
    true_all = np.array([],dtype=np.long)  
    alpha = 0

    with torch.no_grad():
        for spec, emo_label, _ in tqdm(val_loader):
            #spec, label = [x.to(device) for x in spec], label.to(device)   
            spec, emo_label = spec.to(device), emo_label.to(device)            
            res = model(spec, alpha)                                                                                                                                                                                  
            test_loss += criterion(res['emo_pred'], emo_label).item()                                                                                                                                                           
            pred = res['emo_pred'].argmax(dim=1, keepdim=True)
            correct += pred.eq(emo_label.view_as(pred)).sum().item()  

            pred = res['emo_pred'].data.max(1)[1].cpu().numpy()
            true = emo_label.data.cpu().numpy()            
            pred_all = np.append(pred_all,pred)
            true_all = np.append(true_all,true)            

    test_loss /= len(val_loader.dataset)
    acc = 100. * correct / len(val_loader.dataset)

    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), acc))

    con_mat = confusion_matrix(true_all,pred_all)
    cls_rpt = classification_report(true_all,pred_all,target_names=target_names,digits=3)
    logger.info('Confusion Matrix:\n{}\n'.format(con_mat))
    logger.info('Classification Report:\n{}\n'.format(cls_rpt))
    UA = recall_score(true_all,pred_all,average='macro')     
    WA = recall_score(true_all,pred_all,average='weighted')                                                                                                                                                                                                             
    return test_loss,UA,WA

def early_stopping(network,savepath,metricsInEpochs,gap):
    best_metric_inx=np.argmax(metricsInEpochs)
    if best_metric_inx+1==len(metricsInEpochs):
        best = os.path.join(savepath, 'best_epoch_{}.pt'.format(best_metric_inx+1))
        torch.save(network,best)
        return False
    elif (len(metricsInEpochs)-best_metric_inx >= gap):
        return True
    else:
        return False

def main():
    args = parser.parse_args()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    seed = args.seed
    max_len = args.max_len
    feature = args.feature
    dataset_type = args.dataset_type
    root = args.root
    fold = args.fold
    loss_type = args.loss_type
    optimizer_type = args.optimizer_type
    device_number = args.device_number   
    condition = args.condition    
    os.environ["CUDA_VISIBLE_DEVICES"] = device_number

    setup_seed(seed)
    #lr_min = lr * 1e-4
    stamp = datetime.datetime.now().strftime('%y%m%d%H%M')
    tag = stamp + '_' + str(epochs)
    #savedir = os.path.join(root, tag) 
    savedir = os.path.join(root, 'fold_{}'.format(fold))                                                                                                                                                         
    try:
        os.makedirs(savedir)                                                                                                                                                                                       
    except OSError:
        if not os.path.isdir(savedir):
            raise

    feature_size_dic = {'spectrogram':257,'mfcc':39,'FBank':80,'egemaps':88,'compare':6373,'WavLM':768}                                                                                                                                                                                                   
    subprocess.check_call(["cp", "model.py", savedir])
    subprocess.check_call(["cp", "train.py", savedir])
    subprocess.check_call(["cp", "data.py", savedir])
    subprocess.check_call(["cp", "utils.py", savedir])
    subprocess.check_call(["cp", "run.sh", savedir])
    subprocess.check_call(["cp", "augment_cpu.py", savedir])
                                                                                                                                                                                                                   
    logpath = savedir + "/exp.log"
    modelpath = savedir + "/model.pt"
                                                                                                                                                                                                                   
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    feature_ls = feature.split(',')   

    train_set = Mydataset(dataset_type=dataset_type, mode='train', max_len=max_len, fold=fold, feature_ls=feature_ls, condition=condition)
    dev_set = Mydataset(dataset_type=dataset_type, mode='val', max_len=max_len, fold=fold, feature_ls=feature_ls, condition=condition)
    val_set = Mydataset(dataset_type=dataset_type, mode='test', max_len=max_len, fold=fold, feature_ls=feature_ls, condition=condition)

    drop_last = True if len(train_set)%batch_size<2 else False
    if len(feature_ls)==1 and feature_ls[0] in ['WavLM','mfcc']:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,collate_fn=pad_collate, drop_last=drop_last)
        dev_loader = DataLoader(dev_set, batch_size=batch_size//8, shuffle=True, num_workers=1, pin_memory=True,collate_fn=pad_collate)
        val_loader = DataLoader(val_set, batch_size=batch_size//8, shuffle=True, num_workers=1, pin_memory=True,collate_fn=pad_collate)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=drop_last)
        dev_loader = DataLoader(dev_set, batch_size=batch_size//8, shuffle=True, num_workers=1, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size//8, shuffle=True, num_workers=1, pin_memory=True)        
    
    if len(feature_ls)==1:                                                                                                                                                                                                          
        model = Mymodel(feature_size=[feature_size_dic[feature] for feature in feature_ls], h_dims=128, emotion_cls=train_set.NumClasses, domain_cls=2, backbone_type=args.backbone_type).to(device)
    
    if loss_type == 'CE':
        #criterion = nn.NLLLoss(weight=train_set.weight.to(device))
        criterion = nn.NLLLoss()
        criterion_test = nn.NLLLoss(reduction='sum')
    elif loss_type == 'Focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.)
        criterion_test = FocalLoss(alpha=0.25, gamma=2., reduction='sum')
    else:
        raise NameError
                                                                                                                                                                                                                   
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise NameError

    logger = get_logger(logpath)
    logger.info(args)
    logger.info('train_set speaker names: {}'.format(train_set.SpeakerNames))
    logger.info('val_set speaker names: {}'.format(dev_set.SpeakerNames))
    logger.info('test speaker names: {}'.format(val_set.SpeakerNames))
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=15, factor=0.1, verbose=True)

    val_UA_list = []
    test_UA_dic = {}
    test_WA_dic = {}

    for epoch in range(1, epochs+1):
        start = time.time()
        train(model, device, train_loader, dev_loader, val_loader, criterion, optimizer, epoch, logger, epochs)   
        time.sleep(0.003)                                                                                                                 
        val_loss,val_UA,_ = test(model, device, dev_loader, criterion_test, logger, train_set.ClassNames)
        time.sleep(0.003)
        test_loss,test_UA,test_WA = test(model, device, val_loader, criterion_test, logger, train_set.ClassNames)
        end = time.time()
        duration = end-start
        val_UA_list.append(val_UA)
        if early_stopping(model,savedir,val_UA_list,gap=20):
            break
        test_UA_dic[test_UA] = epoch
        test_WA_dic[test_WA] = epoch
        scheduler.step(val_UA)
        logger.info("-"*50)
        logger.info('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        logger.info("-"*50)
        time.sleep(0.003)

    best_UA=max(test_UA_dic.keys())
    best_WA=max(test_WA_dic.keys())
    logger.info('UA dic: {}'.format(test_UA_dic))
    logger.info('WA dic: {}'.format(test_WA_dic))    
    logger.info('best UA: {}  @epoch: {}'.format(best_UA,test_UA_dic[best_UA]))
    logger.info('best WA: {}  @epoch: {}'.format(best_WA,test_WA_dic[best_WA]))   
    torch.save(model, modelpath)                                                                                                                                                                      
                                                                                                                                                                                                                   
if __name__ == '__main__':
    main()
