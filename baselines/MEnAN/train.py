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
from model import FocalLoss, Mymodel, SpkCLS, EmoCLS
from utils import mixup_data, mixup_criterion
import argparse
from sklearn.metrics import confusion_matrix,classification_report,recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser(description="SER-MEnAN")
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
    y_spk = torch.tensor(list(y2))[perm_idx1]
    return xx_pack1, y_emo, y_spk

def train(ENC, SC, EC , device, train_loader, criterion_D, criterion_H, optimizer1, optimizer2, epoch, logger):
    #======================STEP 1======================
    SC.train()
    ENC.eval()
    EC.eval()                                                                                                                                                                                                  
    #logger = get_logger('log/exp.log')
    logger.info('======================start training SC:======================')
                                                                                                                                                                                                                   
    lr = optimizer1.param_groups[0]["lr"]
    logger.info('lr: {:.5f}'.format(lr))                                                                                                                                                                                                         
    correct = 0
    lambda1 = 0.0                                                                                                                                                                                                           
    for batch, data in tqdm(enumerate(train_loader)):
        spec, _, spk_label = data
        spec, spk_label = spec.to(device), spk_label.to(device)                                                                                                                                                            
        #spec, label_a, label_b, lam = mixup_data(spec, label, device, 0.2)                                                                                                                                                                                      
        with torch.no_grad():
            v,_ = ENC(spec, lambda1)

        optimizer1.zero_grad()
        spk_output = SC(v)                                                                                                                                                                                       
        loss1 = criterion_D(spk_output, spk_label)/len(spec)                                                                                                                                                                                                                   
        #loss_func = mixup_criterion(label_a, label_b, lam)                                                                                                                                                         
        #loss = loss_func(criterion, output)                                                                                                                                                                        
        loss1.backward()
        nn.utils.clip_grad_norm_([param for param in SC.parameters() if param.requires_grad], max_norm=10, norm_type=2)                                                                                                                                                                               
        optimizer1.step()

        pred = spk_output.argmax(dim=1, keepdim=True)
        #correct += lam * pred.eq(label_a.view_as(pred)).sum().item() + (1 - lam) * pred.eq(label_b.view_as(pred)).sum().item()
        correct += pred.eq(spk_label.view_as(pred)).sum().item()                                                                                                                                                                                        
        if batch % 20 == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch * len(wav), len(train_loader.dataset),
#                100. * batch / len(train_loader), loss.item()))
            logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t loss={:.5f}\t '.format(epoch , batch * len(spec), len(train_loader.dataset), 100. * batch / len(train_loader), loss1.item()))
                                                                                                                                                                                                
    logger.info('Train set Accuracy: {}/{} ({:.3f}%)'.format(correct, len(train_loader.dataset), 100. * correct / (len(train_loader.dataset))))
    #logger.info('Train set Accuracy: {}/{} ({:.3f}%)'.format(correct, len(train_loader_n.dataset), 100. * correct / len(train_loader_n.dataset)))
    logger.info('finish training SC!')

    #======================STEP 2======================
    SC.eval()
    ENC.train()
    EC.train()                                                                                                                                                                                                  
    #logger = get_logger('log/exp.log')
    logger.info('======================start training ENC and EC:======================')
                                                                                                                                                                                                                   
    lr = optimizer2.param_groups[0]["lr"]
    logger.info('lr: {:.5f}'.format(lr))                                                                                                                                                                                                         
    correct = 0
    lambda2 = 0.5                                                                                                                                                                                                           
    for batch, data in tqdm(enumerate(train_loader)):
        spec, emo_label, spk_label = data
        spec, emo_label, spk_label = spec.to(device), emo_label.to(device), spk_label.to(device)                                                                                                                                                            
        #spec, label_a, label_b, lam = mixup_data(spec, label, device, 0.2)  
        optimizer2.zero_grad()                                                                                                                                                                                    
        v, v_reversed = ENC(spec, lambda2)
        emo_output = EC(v)
        with torch.no_grad():
            spk_output = SC(v_reversed)

        emo_loss = criterion_D(emo_output, emo_label)
        bsz,dim = spk_output.shape
        spk_loss = 0.
        for i in range(dim):
            label_tmp = (torch.ones([bsz],dtype=int)*i).to(device)
            spk_loss += criterion_H(spk_output, label_tmp)

        loss2 = (emo_loss + spk_loss)/len(spec)
        #loss_func = mixup_criterion(label_a, label_b, lam)                                                                                                                                                         
        #loss = loss_func(criterion, output)                                                                                                                                                                        
        loss2.backward()    
        nn.utils.clip_grad_norm_([param for param in ENC.parameters() if param.requires_grad], max_norm=10, norm_type=2)
        nn.utils.clip_grad_norm_([param for param in EC.parameters() if param.requires_grad], max_norm=10, norm_type=2)                                                                                                                                                                                    
        optimizer2.step()

        pred = emo_output.argmax(dim=1, keepdim=True)
        #correct += lam * pred.eq(label_a.view_as(pred)).sum().item() + (1 - lam) * pred.eq(label_b.view_as(pred)).sum().item()
        correct += pred.eq(emo_label.view_as(pred)).sum().item()                                                                                                                                                                                        
        if batch % 20 == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch * len(wav), len(train_loader.dataset),
#                100. * batch / len(train_loader), loss.item()))
            logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t loss={:.5f}\t '.format(epoch , batch * len(spec), len(train_loader.dataset), 100. * batch / len(train_loader), loss2.item()))
                                                                                                                                                                                                
    logger.info('Train set Accuracy: {}/{} ({:.3f}%)'.format(correct, len(train_loader.dataset), 100. * correct / (len(train_loader.dataset))))
    #logger.info('Train set Accuracy: {}/{} ({:.3f}%)'.format(correct, len(train_loader_n.dataset), 100. * correct / len(train_loader_n.dataset)))
    logger.info('finish training ENC and EC!')

def test(ENC, EC, device, val_loader, criterion, logger, target_names):
    ENC.eval()                                                                                                                                                                                                   
    EC.eval()                                                                                                                                                                                                
    test_loss = 0
    correct = 0
    logger.info('testing on dev_set')
                                                                                                                                                                                                                   
    pred_all = np.array([],dtype=np.long)
    true_all = np.array([],dtype=np.long)  
    lambda_ = 0.0

    with torch.no_grad():
        for spec, emo_label, _ in tqdm(val_loader):
            spec, emo_label = spec.to(device), emo_label.to(device)                 
            v,_ = ENC(spec, lambda_)
            emo_output = EC(v)                                                                                                                                                                              
            test_loss += criterion(emo_output, emo_label).item()                                                                                                                                                           
            pred = emo_output.argmax(dim=1, keepdim=True)
            correct += pred.eq(emo_label.view_as(pred)).sum().item()  

            pred = emo_output.data.max(1)[1].cpu().numpy()
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

def early_stopping(ENC,SC,EC,savepath,metricsInEpochs,gap):
    best_metric_inx=np.argmax(metricsInEpochs)
    if best_metric_inx+1==len(metricsInEpochs):
        best_ENC = os.path.join(savepath, 'best_ENC_epoch_{}.pt'.format(best_metric_inx+1))
        torch.save(ENC,best_ENC)
        best_SC = os.path.join(savepath, 'best_SC_epoch_{}.pt'.format(best_metric_inx+1))
        torch.save(SC,best_SC)
        best_EC = os.path.join(savepath, 'best_EC_epoch_{}.pt'.format(best_metric_inx+1))
        torch.save(EC,best_EC)        
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
    #modelpath = savedir + "/model.pt"
                                                                                                                                                                                                                   
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    feature_ls = feature.split(',')   

    train_set = Mydataset(dataset_type=dataset_type, mode='train', max_len=max_len, fold=fold, feature_ls=feature_ls, condition=condition) 
    dev_set = Mydataset(dataset_type=dataset_type, mode='val', max_len=max_len, fold=fold, feature_ls=feature_ls, condition=condition)
    val_set = Mydataset(dataset_type=dataset_type, mode='test', max_len=max_len, fold=fold, feature_ls=feature_ls, condition=condition)

    drop_last = True if len(train_set)%batch_size<2 else False
    if len(feature_ls)==1 and feature_ls[0] in ['WavLM','mfcc']:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,collate_fn=pad_collate, drop_last=drop_last)
        dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,collate_fn=pad_collate)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,collate_fn=pad_collate)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=drop_last)
        dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)       
    
    if len(feature_ls)==1:                                                                                                                                                                                                          
        ENC = Mymodel(feature_size=[feature_size_dic[feature] for feature in feature_ls], h_dims=128, backbone_type=args.backbone_type).to(device)
        bi = 2 if feature_ls[0] in ['WavLM','mfcc','spectrogram'] else 1
        SC = SpkCLS(speaker_cls=train_set.NumSpeakers, h_dims=128*bi).to(device)  
        EC = EmoCLS(emotion_cls=train_set.NumClasses, h_dims=128*bi).to(device)   
    
    spk_p = train_set.speaker_p.float().to(device)

    if loss_type == 'CE':
        #criterion = nn.NLLLoss(weight=train_set.weight.to(device))
        criterion_D = nn.NLLLoss(reduction='sum')
        criterion_H = nn.NLLLoss(weight=spk_p, reduction='sum')
        criterion_test = nn.NLLLoss(reduction='sum')
    elif loss_type == 'Focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.)
        criterion_test = FocalLoss(alpha=0.25, gamma=2., reduction='sum')
    else:
        raise NameError
                                                                                                                                                                                                                   
    if optimizer_type == 'SGD':
        optimizer1 = optim.SGD(SC.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
        optimizer2 = optim.SGD([ENC.parameters(),EC.parameters()], lr=lr, momentum=0.9, weight_decay=1e-3)
    elif optimizer_type == 'Adam':
        optimizer1 = optim.Adam(SC.parameters(), lr=lr)
        optimizer2 = optim.Adam([{'params':ENC.parameters()},{'params':EC.parameters()}], lr=lr)
    else:
        raise NameError

    logger = get_logger(logpath)
    logger.info(args)
    logger.info('train_set speaker names: {}'.format(train_set.SpeakerNames))
    logger.info('val_set speaker names: {}'.format(dev_set.SpeakerNames))
    logger.info('test speaker names: {}'.format(val_set.SpeakerNames))
    scheduler1 = ReduceLROnPlateau(optimizer1, mode='max', patience=15, factor=0.1, verbose=True)
    scheduler2 = ReduceLROnPlateau(optimizer2, mode='max', patience=15, factor=0.1, verbose=True)

    val_UA_list = []
    test_UA_dic = {}
    test_WA_dic = {}

    for epoch in range(1, epochs+1):
        start = time.time()
        train(ENC, SC, EC, device, train_loader, criterion_D, criterion_H, optimizer1, optimizer2, epoch, logger)   
        time.sleep(0.003)                                                                                                                 
        val_loss,val_UA,_ = test(ENC, EC, device, dev_loader, criterion_test, logger, train_set.ClassNames)
        time.sleep(0.003)
        test_loss,test_UA,test_WA = test(ENC, EC, device, val_loader, criterion_test, logger, train_set.ClassNames)
        end = time.time()
        duration = end-start
        val_UA_list.append(val_UA)
        if early_stopping(ENC,SC,EC,savedir,val_UA_list,gap=20):
            break
        test_UA_dic[test_UA] = epoch
        test_WA_dic[test_WA] = epoch
        scheduler1.step(val_UA)
        scheduler2.step(val_UA)
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
    torch.save(ENC, savedir + "/ENC.pt")  
    torch.save(EC, savedir + "/EC.pt") 
    torch.save(SC, savedir + "/SC.pt")                                                                                                                                                                     
                                                                                                                                                                                                                   
if __name__ == '__main__':
    main()
