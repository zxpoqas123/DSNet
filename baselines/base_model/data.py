import os
import numpy as np
import pandas as pd
from pandas import Series
import soundfile as sound
import random
import librosa
import pickle
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from augment_cpu import Pad_trunc_seq, Trunc_seq, Deltas_Deltas_FBank, Deltas_Deltas_mfcc

class Augment(object):
    def __init__(self, max_len=5, feature='egemaps'):
        super(Augment, self).__init__()
        self.feature = feature
        self.feature_coeff_dic={'spectrogram':100,'mfcc':100,'FBank':100,'egemaps':1,'compare':1,'WavLM':50}
        self.l = int(self.feature_coeff_dic[feature]*max_len)
        self.trunc_seq = Trunc_seq(max_len = self.l)
        self.pad_trunc_seq = Pad_trunc_seq(max_len = self.l)
        self.deltas_deltas_mfcc = Deltas_Deltas_mfcc()
        self.deltas_deltas_fbank = Deltas_Deltas_FBank()

    def __call__(self, x):
        if self.feature in ['egemaps','compare']:
            return x
        elif self.feature=='spectrogram':
            out = self.pad_trunc_seq(x)            
            return out            
        elif self.feature=='mfcc':
            out = self.trunc_seq(x)            
            out = self.deltas_deltas_mfcc(out)
            return out
        elif self.feature=='FBank':
            out = self.pad_trunc_seq(x)            
            out = self.deltas_deltas_fbank(out)
            return out
        elif self.feature=='WavLM':
            out = self.trunc_seq(x)       
            return out

class Mydataset(Dataset):
    def __init__(self, dataset_type='IEMOCAP_4', mode='train', max_len=6, fold=0, num=1, feature_ls=['FBank'], condition='impro'):
        data_all = pd.read_csv('/home/chenchengxin/spk_adaptiation_for_SER/meta/{}.tsv'.format(dataset_type), sep='\t')
        self.dataset_type = dataset_type.split('_')[0]
        SpkNames = np.unique(data_all['speaker'])
        self.mode = mode
        self.podcast = {'train':'Train','val':'Validation','test':'Test1'}  # for MSP-Podcast
        if condition != 'all':  #for IEMOCAP
            data_all = data_all[data_all.condition==condition].reset_index(drop=True)
        self.data_info = self.split_dataset(data_all, fold, SpkNames)
        feature_detail_dic={'spectrogram':'spectrogram_257','mfcc':'mfcc_13','FBank':'FBank_80','egemaps':'egemaps_88','compare':'ComParE_6373','WavLM':'WavLM_base_768'}

        self.modalities = len(feature_ls)
        if self.dataset_type=='MSP-Podcast':
            self.feature_root = ['/home/chenchengxin/spk_adaptiation_for_SER/dataset/{}_{}_{}.pkl'.format(self.dataset_type, self.podcast[self.mode], feature_detail_dic[feature]) for feature in feature_ls]
        else:
            self.feature_root = ['/home/chenchengxin/spk_adaptiation_for_SER/dataset/{}_{}.pkl'.format(self.dataset_type, feature_detail_dic[feature]) for feature in feature_ls]
        
        self.feature_dic_ls = []
        self.transform_ls = []
        for i in range(self.modalities):
            with open(self.feature_root[i], 'rb') as f:
                self.feature_dic_ls.append(pickle.load(f))
            self.transform_ls.append(transforms.Compose([Augment(max_len=max_len, feature=feature_ls[i])]))

        self.label = self.data_info['label'].astype('category').cat.codes.values
        self.speaker = self.data_info['speaker'].astype('category').cat.codes.values                                                                                                                                                                               
        self.ClassNames = np.unique(self.data_info['label'])
        self.SpeakerNames = np.sort(np.unique(self.data_info['speaker']))
        self.speaker_p = torch.tensor([(self.data_info.speaker==x).sum()/len(self.speaker) for x in self.SpeakerNames])
        self.NumClasses = len(self.ClassNames)
        self.NumSpeakers = len(self.SpeakerNames)
        self.weight = 1/torch.tensor([(self.label==i).sum() for i in range(self.NumClasses)]).float()

        #self.dic_neutral = self.get_neutral_dic(self.data_info, self.SpeakerNames, data_type=mode, num=num)

    def split_dataset(self, df_all, fold, speakers):
        if self.dataset_type=='MSP-Podcast':
            return df_all[df_all.Split_Set==self.podcast[self.mode]].reset_index(drop=True)
        spk_len = len(speakers)
        #test_idx = np.array(df_all['speaker']==speakers[fold*2%spk_len])+np.array(df_all['speaker']==speakers[(fold*2+1)%spk_len])
        #val_idx = np.array(df_all['speaker']==speakers[(fold*2-2)%spk_len])+np.array(df_all['speaker']==speakers[(fold*2-1)%spk_len])
        #train_idx = True^(test_idx+val_idx)
        #train_idx = True^test_idx
        test_idx = np.array(df_all['speaker']==speakers[fold%spk_len])
        if fold%2==0:
            val_idx = np.array(df_all['speaker']==speakers[(fold+1)%spk_len])
        else:
            val_idx = np.array(df_all['speaker']==speakers[(fold-1)%spk_len])
        train_idx = True^(test_idx+val_idx)
        train_data_info = df_all[train_idx].reset_index(drop=True)
        val_data_info = df_all[val_idx].reset_index(drop=True)
        test_data_info = df_all[test_idx].reset_index(drop=True)
        #val_data_info = test_data_info = df_all[test_idx].reset_index(drop=True)

        if self.mode == 'train':
            data_info = train_data_info
        elif self.mode == 'val':
            data_info = val_data_info
        elif self.mode == 'test':
            data_info = test_data_info
        else:
            data_info = df_all
        return data_info

    def get_neutral_dic(self, df, speakers, data_type='val', num=10):
        # here df_N = df[df.label=='Neu'] for dataset_type=='VESUS'
        df_N = df[df.label=='N']
        dic={}
        if data_type == 'val' or data_type == 'test':
            for s in speakers:
                df_tmp = df_N[df_N.speaker==s].reset_index(drop=True)
                dic[s] = df_tmp.filename[num]
        else:
            for s in speakers:
                df_tmp = df_N[df_N.speaker==s].reset_index(drop=True)
                dic[s] = list(df_tmp.filename)            
        return dic 

    def __len__(self):
        return len(self.data_info)
                                                                                                                                                                                                                   
    def __getitem__(self, idx):
        spec = self.transform_ls[0](torch.tensor(self.feature_dic_ls[0][self.data_info.filename[idx]])).float()
        label = self.label[idx]                                                                                                                                                                                    
        label = np.array(label)                                                                                                                                                                                    
        label = label.astype('float').reshape(1)
        label = torch.Tensor(label).long().squeeze()
        return spec, label
