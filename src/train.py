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
#from torchvision import transforms, utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import logging
import time
from data import Mydataset
from model import Encoder, Translator, Decoder, Classifier, DiffLoss, MSE, JS
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
parser.add_argument("--alpha",default='0.0',type=float)
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

def train(model_dic, device, train_loader, cri_dic, opt_dic, epoch, logger):
    E,C,T,T2,D = model_dic['E'],model_dic['C'],model_dic['T'],model_dic['T2'],model_dic['D']
    opt_E,opt_C,opt_T,opt_T2,opt_D = opt_dic['opt_E'],opt_dic['opt_C'],opt_dic['opt_T'],opt_dic['opt_T2'],opt_dic['opt_D']
    cri_task,cri_sim,cri_diff,cri_recon = cri_dic['cri_task'],cri_dic['cri_sim'],cri_dic['cri_diff'],cri_dic['cri_recon']
                                                                                                                                                         
    logger.info('======================start training:======================')    
    E.train() 
    T.train()
    T2.train()
    D.train()
    C.train()                                                                                                                                                                                           
    lr = opt_E.param_groups[0]["lr"]
    logger.info('lr: {:.5f}'.format(lr))
    loss_task, loss_sim, loss_diff,loss_recon = 0.0, 0.0, 0.0, 0.0
    correct = 0

    for batch, (xs, xs_neu, ys) in tqdm(enumerate(train_loader)):
        xs, xs_neu, ys = xs.to(device), xs_neu.to(device), ys.to(device)
        ## Train E, T, T2, D
        embedding = E(xs)
        embedding_neu = E(xs_neu)
        features_spk = T(embedding)
        features_nonspk = T2(embedding)
        embedding_recon = D(torch.cat([features_spk,features_nonspk],-1))
        out, _ = C(features_spk*args.alpha+features_nonspk)

        loss_sim_batch = cri_sim(features_spk, embedding_neu)
        loss_diff_batch = cri_diff(features_spk,features_nonspk)
        loss_recon_batch = cri_recon(embedding,embedding_recon)
        loss_task_batch = cri_task(out, ys)
        loss = loss_task_batch + loss_sim_batch + loss_diff_batch + loss_recon_batch

        opt_E.zero_grad()
        opt_T.zero_grad()
        opt_T2.zero_grad()
        opt_D.zero_grad()
        opt_C.zero_grad()
        loss.backward()
        opt_E.step()
        opt_T.step()
        opt_T2.step()
        opt_D.step()
        opt_C.step()

        loss_sim += loss_sim_batch.cpu().item()
        loss_diff += loss_diff_batch.cpu().item()
        loss_recon += loss_recon_batch.cpu().item()
        loss_task += loss_task_batch.cpu().item()
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(ys.view_as(pred)).sum().item()   
        #if batch % 20 == 0:
        #    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t loss={:.5f}\t '.format(epoch , batch * len(ys), len(train_loader.dataset), 100. * batch / len(train_loader), loss.item()))
    
    loss_sim = loss_sim / len(train_loader)
    loss_diff = loss_diff / len(train_loader)   
    loss_recon = loss_recon / len(train_loader)
    loss_task = loss_task / len(train_loader)  
    acc = 100. * correct / (len(train_loader.dataset))
    logger.info('Train loss_sim={:.5f} \t loss_diff={:.5f} \t loss_recon={:.5f}\n'.format(loss_sim, loss_diff, loss_recon))
    logger.info('Train set Accuracy: {}/{} ({:.3f}%) \t loss_task={:.5f}\n'.format(correct, len(train_loader.dataset), acc, loss_task))
    logger.info('finish training!')
    result = {'acc':acc, 'loss_task':loss_task, 'loss_sim':loss_sim, 'loss_diff':loss_diff, 'loss_recon':loss_recon}
    return result

def test(model_dic, device, val_loader, cri_dic, logger, target_names):
    E,C,T,T2,D = model_dic['E'],model_dic['C'],model_dic['T'],model_dic['T2'],model_dic['D']
    cri_task,cri_sim,cri_diff,cri_recon = cri_dic['cri_task'],cri_dic['cri_sim'],cri_dic['cri_diff'],cri_dic['cri_recon']
    E.eval()
    T.eval()
    T2.eval()
    C.eval()
    D.eval()   
    loss_sim, loss_diff,loss_recon,loss_task = 0.0, 0.0, 0.0, 0.0
    correct = 0
    logger.info('testing on dev_set')
                                                                                                                                                                                                                   
    pred_all = np.array([],dtype=np.long)
    true_all = np.array([],dtype=np.long)  

    with torch.no_grad():
        for step, (xs, xs_neu, ys) in tqdm(enumerate(val_loader)):
            xs, xs_neu, ys = xs.to(device), xs_neu.to(device), ys.to(device)
            embedding = E(xs)
            embedding_neu = E(xs_neu)
            features_spk = T(embedding)
            features_nonspk = T2(embedding)
            embedding_recon = D(torch.cat([features_spk,features_nonspk],-1))
            out, _ = C(features_spk*args.alpha+features_nonspk)

            loss_sim_batch = cri_sim(features_spk, embedding_neu)
            loss_diff_batch = cri_diff(features_spk,features_nonspk)
            loss_recon_batch = cri_recon(embedding,embedding_recon)  
            loss_task_batch = cri_task(out, ys)
                                                                                                                                                          
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(ys.view_as(pred)).sum().item()

            pred = out.data.max(1)[1].cpu().numpy()
            true = ys.data.cpu().numpy()
            pred_all = np.append(pred_all,pred)
            true_all = np.append(true_all,true)

            loss_sim += loss_sim_batch.cpu().item()
            loss_diff += loss_diff_batch.cpu().item()
            loss_recon += loss_recon_batch.cpu().item()
            loss_task += loss_task_batch.cpu().item()

    loss_sim = loss_sim / len(val_loader)
    loss_diff = loss_diff / len(val_loader)   
    loss_recon = loss_recon / len(val_loader)
    loss_task = loss_task / len(val_loader)                

    acc = 100. * correct / len(val_loader.dataset)

    logger.info('Test set: loss_task={:.5f} \t loss_sim={:.5f} \t loss_diff={:.5f} \t loss_recon={:.5f} \t Accuracy: {}/{} ({:.4f}%)\n'.format(
       loss_task, loss_sim, loss_diff, loss_recon, correct, len(val_loader.dataset), acc))

    con_mat = confusion_matrix(true_all,pred_all)
    cls_rpt = classification_report(true_all,pred_all,target_names=target_names,digits=3)
    logger.info('Confusion Matrix:\n{}\n'.format(con_mat))
    logger.info('Classification Report:\n{}\n'.format(cls_rpt))
    UA = recall_score(true_all,pred_all,average='macro')     
    WA = recall_score(true_all,pred_all,average='weighted') 
    result = {'acc':acc, 'loss_task':loss_task, 'loss_sim':loss_sim, 'loss_diff':loss_diff, 'loss_recon':loss_recon, 'UA':UA, 'WA':WA}
    return result                                                                                                                                                                                                                

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
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,collate_fn=pad_collate, drop_last=drop_last)
        dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,collate_fn=pad_collate)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,collate_fn=pad_collate)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)
        dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)       
                                                                                                                                                                                                         
    E = Encoder().to(device)
    C = Classifier().to(device)
    T = Translator().to(device)
    T2 = Translator().to(device)
    D = Decoder().to(device)

    if loss_type == 'CE':
        #criterion = nn.NLLLoss(weight=train_set.weight.to(device))
        cri_task = nn.NLLLoss()
        #criterion_test = nn.NLLLoss(reduction='sum')
    elif loss_type == 'Focal':
        cri_task = FocalLoss(alpha=0.25, gamma=2.)
        #criterion_test = FocalLoss(alpha=0.25, gamma=2., reduction='sum')
    else:
        raise NameError
    cri_diff = DiffLoss()
    #cri_sim = nn.CosineSimilarity(dim=1)
    cri_sim = JS()
    cri_recon = MSE()
    #loss_dist = nn.L1Loss()

    if optimizer_type == 'SGD':
        opt_E = optim.SGD(E.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        opt_C = optim.SGD(C.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
        opt_T = optim.SGD(T.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5) 
        opt_T2 = optim.SGD(T2.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5) 
        opt_D = optim.SGD(D.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    elif optimizer_type == 'Adam':
        opt_E = optim.Adam(E.parameters(), lr=lr)
        opt_C = optim.Adam(C.parameters(), lr=lr)
        opt_T = optim.Adam(T.parameters(), lr=lr)
        opt_T2 = optim.Adam(T2.parameters(), lr=lr)
        opt_D = optim.Adam(D.parameters(), lr=lr)
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
    scheduler_T = ReduceLROnPlateau(opt_T, mode='max', patience=15, factor=0.1, verbose=True) 
    scheduler_T2 = ReduceLROnPlateau(opt_T2, mode='max', patience=15, factor=0.1, verbose=True)
    scheduler_D = ReduceLROnPlateau(opt_D, mode='max', patience=15, factor=0.1, verbose=True) 

    val_UA_list = []
    test_UA_dic = {}
    test_WA_dic = {}
    loss_task_train, loss_task_val, loss_task_test = [], [], []
    loss_sim_train, loss_sim_val, loss_sim_test = [], [], []
    loss_diff_train, loss_diff_val, loss_diff_test = [], [], []
    loss_recon_train, loss_recon_val, loss_recon_test = [], [], []
    acc_train, acc_val, acc_test = [], [], []

    model_dic = {'E':E, 'C':C, 'T':T, 'T2':T2, 'D':D}
    opt_dic = {'opt_E':opt_E, 'opt_C':opt_C, 'opt_T':opt_T, 'opt_T2':opt_T2, 'opt_D':opt_D}
    cri_dic = {'cri_task':cri_task, 'cri_diff':cri_diff, 'cri_sim':cri_sim, 'cri_recon':cri_recon}

    for epoch in range(1, epochs+1):
        start = time.time()
        result_train = train(model_dic, device, train_loader, cri_dic, opt_dic, epoch, logger)   
        time.sleep(0.003)                                                                                                                 
        result_val = test(model_dic, device, dev_loader, cri_dic, logger, train_set.ClassNames)
        time.sleep(0.003)
        result_test = test(model_dic, device, val_loader, cri_dic, logger, train_set.ClassNames)        
        end = time.time()
        duration = end-start
        val_UA_list.append(result_val['UA'])
        model_state_dic = {'E':E.state_dict(),'C':C.state_dict(),'T':T.state_dict(),'T2':T2.state_dict(),'D':D.state_dict()}
        if early_stopping(model_state_dic,savedir,val_UA_list,gap=20):
            break
        test_UA_dic[result_test['UA']] = epoch
        test_WA_dic[result_test['WA']] = epoch
        scheduler_E.step(result_val['UA'])
        scheduler_C.step(result_val['UA'])
        scheduler_T.step(result_val['UA'])
        scheduler_T2.step(result_val['UA'])
        scheduler_D.step(result_val['UA'])
        logger.info("-"*50)
        logger.info('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} Valid Loss {:5.4f}'.format(epoch, duration, result_train['loss_task'], result_val['loss_task']))
        logger.info("-"*50)
        time.sleep(0.003)

        acc_train.append(result_train['acc'])
        loss_task_train.append(result_train['loss_task'])
        loss_sim_train.append(result_train['loss_sim'])
        loss_diff_train.append(result_train['loss_diff'])
        loss_recon_train.append(result_train['loss_recon'])

        acc_val.append(result_val['acc'])
        loss_task_val.append(result_val['loss_task'])
        loss_sim_val.append(result_val['loss_sim'])
        loss_diff_val.append(result_val['loss_diff'])
        loss_recon_val.append(result_val['loss_recon']) 

        acc_test.append(result_test['acc'])
        loss_task_test.append(result_test['loss_task'])
        loss_sim_test.append(result_test['loss_sim'])
        loss_diff_test.append(result_test['loss_diff'])
        loss_recon_test.append(result_test['loss_recon'])        

    dataframe = pd.DataFrame({'loss_task_train':loss_task_train, 'loss_sim_train':loss_sim_train, 'loss_diff_train':loss_diff_train,'loss_recon_train':loss_recon_train, 'accuracy_train':acc_train,
                            'loss_task_val':loss_task_val, 'loss_sim_val':loss_sim_val, 'loss_diff_val':loss_diff_val,'loss_recon_val':loss_recon_val, 'accuracy_val':acc_val,
                            'loss_task_test':loss_task_test, 'loss_sim_test':loss_sim_test, 'loss_diff_test':loss_diff_test,'loss_recon_test':loss_recon_test, 'accuracy_test':acc_test})
    dataframe.to_csv(os.path.join(savedir,'metri_all_{}.tsv'.format(args.mode)), index=False, sep='\t')
    best_UA=max(test_UA_dic.keys())
    best_WA=max(test_WA_dic.keys())
    logger.info('UA dic: {}'.format(test_UA_dic))
    logger.info('WA dic: {}'.format(test_WA_dic))    
    logger.info('best UA: {}  @epoch: {}'.format(best_UA,test_UA_dic[best_UA]))
    logger.info('best WA: {}  @epoch: {}'.format(best_WA,test_WA_dic[best_WA]))   
    torch.save(model_state_dic, modelpath)                                                                                                                                                                      
                                                                                                                                                                                                                   
if __name__ == '__main__':
    main()
