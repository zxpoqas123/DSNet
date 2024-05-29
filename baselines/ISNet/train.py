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
from model import Encoder, Translator, Classifier
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
parser.add_argument("--mode",default='isnet',type=str)
args = parser.parse_args() 

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
    xx1,y = zip(*batch)
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
    return xx_pack1, y_emo

def train(model_dic, device, train_loader, loss_class, loss_dist, opt_dic, epoch, logger):
    E,C = model_dic['E'],model_dic['C']
    opt_E,opt_C = opt_dic['opt_E'],opt_dic['opt_C']
    E.train() 
    C.train()
    if args.mode == 'isnet':
        E2,C2,T = model_dic['E2'],model_dic['C2'],model_dic['T']
        opt_E2,opt_C2,opt_T = opt_dic['opt_E2'],opt_dic['opt_C2'],opt_dic['opt_T']
        E2.train()
        C2.train()
        T.train()                                                                                                                                                                   
    logger.info('start training')
                                                                                                                                                                                                                   
    lr = opt_E.param_groups[0]["lr"]
    logger.info('lr: {:.5f}'.format(lr))
    
    loss_tr, loss2_tr = 0.0, 0.0                                                                                                                                                                                                              
    correct = 0
    num_neu = 3
                                                                                                                                                                                                                   
    for batch, (xs, xs_pairs, ys) in tqdm(enumerate(train_loader)):
        xs, ys = xs.to(device), ys.to(device)                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        ## Train E, C
        out, features = C(E(xs))
        loss = loss_class(out, ys)

        opt_E.zero_grad()
        opt_C.zero_grad()
        loss.backward()
        opt_E.step()
        opt_C.step()

        if args.mode == 'isnet':
            ## Train E
            out_E_pairs = []
            for xs_pair in xs_pairs:
                out_E_pairs.append(E2(xs_pair.to(device)).unsqueeze(1))
            out_E_pairs = torch.cat(out_E_pairs, 1)     # B x N_pair x H
            loss3 = 0
            for i in range(num_neu-1):
                for j in range(i+1, num_neu):
                    loss3 += loss_dist(out_E_pairs[:,i], out_E_pairs[:,j])
            opt_E2.zero_grad()
            loss3.backward()
            opt_E2.step()

            ## Train T
            out_E_pairs = []
            for xs_pair in xs_pairs:
                out_E_pairs.append(E2(xs_pair.to(device)).unsqueeze(1))
            out_E_pairs = torch.cat(out_E_pairs, 1)     # B x N_pair x H
            out_E_pair = torch.mean(out_E_pairs, 1)     # B x H
            out_T = T(E(xs))
            loss2 = loss_dist(out_T, out_E_pair)
            opt_T.zero_grad()
            loss2.backward()
            opt_T.step()

            ## Train C2
            out_E = E(xs)
            out_E = out_E - T(out_E)
            out, features = C2(out_E)
            loss = loss_class(out, ys)
            opt_C2.zero_grad()
            loss.backward()
            opt_C2.step()

        else:
            loss2 = loss

        pred = out.argmax(dim=1, keepdim=True)
        #correct += lam * pred.eq(label_a.view_as(pred)).sum().item() + (1 - lam) * pred.eq(label_b.view_as(pred)).sum().item()
        correct += pred.eq(ys.view_as(pred)).sum().item()                                                                                                                                                                                        
        loss_tr += loss.cpu().item()
        loss2_tr += loss2.cpu().item()
        #if batch % 20 == 0:
        #    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t loss={:.5f}\t '.format(epoch , batch * len(ys), len(train_loader.dataset), 100. * batch / len(train_loader), loss.item()))
    
    loss_tr = loss_tr / len(train_loader)
    loss2_tr = loss2_tr / len(train_loader)   
    acc_tr = 100. * correct / (len(train_loader.dataset))
    logger.info('Train set Accuracy: {}/{} ({:.3f}%) \t Train loss={:.5f} \t loss2={:.5f}\n'.format(correct, len(train_loader.dataset), acc_tr, loss_tr, loss2_tr))
    logger.info('finish training!')
    return acc_tr, loss_tr, loss2_tr

def test(model_dic, device, val_loader, loss_class, loss_dist, logger, target_names):
    E,C = model_dic['E'],model_dic['C']
    E.eval()
    C.eval()
    if args.mode == 'isnet':
        E2,C2,T = model_dic['E2'],model_dic['C2'],model_dic['T']
        E2.eval()
        C2.eval()
        T.eval()          
    loss_te, loss2_te = 0.0, 0.0
    correct = 0
    logger.info('testing on dev_set')
                                                                                                                                                                                                                   
    pred_all = np.array([],dtype=np.long)
    true_all = np.array([],dtype=np.long)  

    with torch.no_grad():
        for step, (xs, xs_pairs, ys) in tqdm(enumerate(val_loader)):
            xs, ys = xs.to(device), ys.to(device)

            out, _ = C(E(xs))
            loss = loss_class(out, ys)

            if args.mode == 'isnet':
                out_E_pairs = []
                for xs_pair in xs_pairs:
                    out_E_pairs.append(E2(xs_pair.to(device)).unsqueeze(1))
                out_E_pairs = torch.cat(out_E_pairs, 1)     # B x N_pair x H
                out_E_pair = torch.mean(out_E_pairs, 1)     # B x H
                out_T = T(E(xs))
                loss2 = loss_dist(out_T, out_E_pair)

                out_E = E(xs)
                out_E = out_E - T(out_E)
                out, features = C2(out_E)
                loss = loss_class(out, ys)
            else:
                loss2 = loss
                                                                                                                                                          
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(ys.view_as(pred)).sum().item()  

            pred = out.data.max(1)[1].cpu().numpy()
            true = ys.data.cpu().numpy()            
            pred_all = np.append(pred_all,pred)
            true_all = np.append(true_all,true)            

            loss_te += loss.cpu().item()
            loss2_te += loss2.cpu().item()

    loss_te = loss_te / len(val_loader)
    loss2_te = loss2_te / len(val_loader)
    acc = 100. * correct / len(val_loader.dataset)

    logger.info('Test set: Average loss: {:.4f} \t loss2: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
       loss_te, loss2_te, correct, len(val_loader.dataset), acc))

    con_mat = confusion_matrix(true_all,pred_all)
    cls_rpt = classification_report(true_all,pred_all,target_names=target_names,digits=3)
    logger.info('Confusion Matrix:\n{}\n'.format(con_mat))
    logger.info('Classification Report:\n{}\n'.format(cls_rpt))
    UA = recall_score(true_all,pred_all,average='macro')     
    WA = recall_score(true_all,pred_all,average='weighted')                                                                                                                                                                                                             
    return loss_te, loss2_te, UA, WA

def early_stopping(model_state_dic,savepath,metricsInEpochs,gap):
    best_metric_inx=np.argmax(metricsInEpochs)
    if best_metric_inx+1==len(metricsInEpochs):
        best = os.path.join(savepath, 'best_epoch_{}.pt'.format(best_metric_inx+1))
        torch.save(model_state_dic,best)
        return False
    elif (len(metricsInEpochs)-best_metric_inx >= gap):
        return True
    else:
        return False

def main():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
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
        dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,collate_fn=pad_collate)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,collate_fn=pad_collate)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=drop_last)
        dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)       
                                                                                                                                                                                                         
    E = Encoder().to(device)
    C = Classifier().to(device)
    if args.mode == 'isnet':
        E2 = Encoder().to(device)
        C2 = Classifier().to(device)
        T = Translator().to(device)

    if loss_type == 'CE':
        #criterion = nn.NLLLoss(weight=train_set.weight.to(device))
        loss_class = nn.CrossEntropyLoss()
        #criterion_test = nn.NLLLoss(reduction='sum')
    elif loss_type == 'Focal':
        loss_class = FocalLoss(alpha=0.25, gamma=2.)
        #criterion_test = FocalLoss(alpha=0.25, gamma=2., reduction='sum')
    else:
        raise NameError
    loss_dist = nn.L1Loss()

    if optimizer_type == 'SGD':
        opt_E = optim.SGD(E.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        opt_C = optim.SGD(C.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        if args.mode == 'isnet':
            opt_E2 = optim.SGD(E2.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
            opt_C2 = optim.SGD(C2.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5) 
            opt_T = optim.SGD(T.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5) 
    elif optimizer_type == 'Adam':
        opt_E = optim.Adam(E.parameters(), lr=lr)
        opt_C = optim.Adam(C.parameters(), lr=lr)
        if args.mode == 'isnet':
            opt_E2 = optim.Adam(E2.parameters(), lr=lr)
            opt_C2 = optim.Adam(C2.parameters(), lr=lr)
            opt_T = optim.Adam(T.parameters(), lr=lr)
    else:
        raise NameError

    logger = get_logger(logpath)
    logger.info(args)
    logger.info('train_set speaker names: {}'.format(train_set.SpeakerNames))
    logger.info('val_set speaker names: {}'.format(dev_set.SpeakerNames))
    logger.info('test speaker names: {}'.format(val_set.SpeakerNames))
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=15, factor=0.1, verbose=True)
    scheduler_E = ReduceLROnPlateau(opt_E, mode='max', patience=15, factor=0.1, verbose=True)
    scheduler_C = ReduceLROnPlateau(opt_C, mode='max', patience=15, factor=0.1, verbose=True)
    if args.mode == 'isnet':
        scheduler_E2 = ReduceLROnPlateau(opt_E2, mode='max', patience=15, factor=0.1, verbose=True)
        scheduler_C2 = ReduceLROnPlateau(opt_C2, mode='max', patience=15, factor=0.1, verbose=True)        
        scheduler_T = ReduceLROnPlateau(opt_T, mode='max', patience=15, factor=0.1, verbose=True) 

    val_UA_list = []
    test_UA_dic = {}
    test_WA_dic = {}
    loss_train, loss_test = [], []
    loss2_train, loss2_test = [], []
    acc_train, acc_test = [], []

    if args.mode == 'isnet':
        model_dic = {'E':E,'C':C,'E2':E2,'C2':C2,'T':T}
        opt_dic = {'opt_E':opt_E, 'opt_C':opt_C, 'opt_E2':opt_E2, 'opt_C2':opt_C2, 'opt_T':opt_T}
    else:
        model_dic = {'E':E,'C':C}
        opt_dic = {'opt_E':opt_E, 'opt_C':opt_C}

    for epoch in range(1, epochs+1):
        start = time.time()
        train_WA,loss_tr,loss2_tr = train(model_dic, device, train_loader, loss_class, loss_dist, opt_dic, epoch, logger)   
        time.sleep(0.003)                                                                                                                 
        loss_te,loss2_te,val_UA,val_WA = test(model_dic, device, dev_loader, loss_class, loss_dist, logger, train_set.ClassNames)
        time.sleep(0.003)
        _,_,test_UA,test_WA = test(model_dic, device, val_loader, loss_class, loss_dist, logger, train_set.ClassNames)        
        end = time.time()
        duration = end-start
        val_UA_list.append(val_UA)
        if args.mode == 'isnet':
            model_state_dic = {'E':E.state_dict(),'C':C.state_dict(),'E2':E2.state_dict(),'C2':C2.state_dict(),'T':T.state_dict()}
        else:
            model_state_dic = {'E':E.state_dict(),'C':C.state_dict()}
        if early_stopping(model_state_dic,savedir,val_UA_list,gap=20):
            break
        test_UA_dic[test_UA] = epoch
        test_WA_dic[test_WA] = epoch
        scheduler_E.step(val_UA)
        scheduler_C.step(val_UA)
        if args.mode == 'isnet':
            scheduler_E2.step(val_UA)
            scheduler_C2.step(val_UA)
            scheduler_T.step(val_UA)
        logger.info("-"*50)
        logger.info('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} Valid Loss {:5.4f}'.format(epoch, duration, loss_tr, loss_te))
        logger.info("-"*50)
        time.sleep(0.003)

        loss_train.append(loss_tr)
        loss2_train.append(loss2_tr)
        acc_train.append(train_WA)
        loss_test.append(loss_te)
        loss2_test.append(loss2_te)
        acc_test.append(val_WA)

    dataframe = pd.DataFrame({'loss_train':loss_train, 'loss2_train':loss2_train, 'loss_test':loss_test,'loss2_test':loss2_test, 'accuracy_train':acc_train,'accuracy_test':acc_test})      
    dataframe.to_csv(os.path.join(savedir,'train_test_{}.tsv'.format(args.mode)), index=False, sep='\t')
    best_UA=max(test_UA_dic.keys())
    best_WA=max(test_WA_dic.keys())
    logger.info('UA dic: {}'.format(test_UA_dic))
    logger.info('WA dic: {}'.format(test_WA_dic))    
    logger.info('best UA: {}  @epoch: {}'.format(best_UA,test_UA_dic[best_UA]))
    logger.info('best WA: {}  @epoch: {}'.format(best_WA,test_WA_dic[best_WA]))   
    torch.save(model_state_dic, modelpath)                                                                                                                                                                      
                                                                                                                                                                                                                   
if __name__ == '__main__':
    main()
