import os
import numpy as np
import pandas as pd
from pandas import Series
import soundfile as sound
import random
import librosa
import torchaudio.compliance.kaldi as kaldi
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from augment_cpu import Pad_trunc_seq, Spectrogram, MelScale, MelSpectrogram, AxisMasking, FrequencyMasking, TimeMasking, AddNoise, Crop, TimeStretch, PitchShift, Deltas_Deltas, TimeShift

class Augment_wav(object):
    def __init__(self, run=True, p=True):
        super(Augment_wav, self).__init__()
        self.run = run
        self.p = p
        self.spectrogram = Spectrogram(n_fft=2048, win_length=2048,hop_length=1024,
                                   pad=0, window_fn=torch.hann_window, power=2.,
                                   normalized=False, wkwargs=None)
        self.spectrogram_c = Spectrogram(n_fft=2048, win_length=2048,hop_length=1024,
                                   pad=0, window_fn=torch.hann_window, power=None,
                                   normalized=False, wkwargs=None)
        self.mel_scale = MelScale(128, 44100, 0, 22050, 1025)
        self.mel_specgram = MelSpectrogram(sample_rate = 16000, n_fft = 400, hop_length = 160, f_min = 0.0, f_max = 8000.0, n_mels = 40)
        self.pad_trunc_seq = Pad_trunc_seq(max_len = 96000)
        #self.pad_trunc_seq = Pad_trunc_seq(max_len = 80000)
        self.freqmask = FrequencyMasking(freq_mask_param = 12, iid_masks = False)
        self.timemask = TimeMasking(time_mask_param = 40, iid_masks = False)
        self.addnoise = AddNoise()
        self.crop = Crop()
        self.timestretch = TimeStretch(n_freq=1025, n_fft=2048)
        self.pitchshift = PitchShift(44100, 12)
        self.deltas_deltas = Deltas_Deltas()
        self.timeshift = TimeShift()

    def __call__(self, x):
        out = x                                                                                                                                                                                                    
        if self.run == True:
            a = np.random.randint(5)
            #if torch.rand(1) < 1.:
            if a == 0:
                out = self.addnoise(out, [5, 10, 20])
            #if torch.rand(1) < 1.:
            if a == 1:
                rate = np.random.uniform(0.9, 1.1)
                t = out.numpy().squeeze(0)
                t = librosa.effects.time_stretch(t, rate)                                                                                                                                                          
                out = torch.from_numpy(t).unsqueeze(0)
                if out.shape[-1] < 441000:
                    out = torch.cat((out, out), -1)
                    out_t = out[:, 0:441000]
                    out = out_t                                                                                                                                                                                    
                else:
                    out_t = out[:, 0:441000]
                    out = out_t                                                                                                                                                                                    
                #out_2 = self.timestretch(out, 1, True)
                #out = torch.cat((out, out_2), 0)
            #if torch.rand(1) < 1.:
            if a == 2:
                n_steps = np.random.uniform(-4, 4)
                n = out.numpy().squeeze(0)
                n = librosa.effects.pitch_shift(n, 44100, n_steps=n_steps)
                out = torch.from_numpy(n).unsqueeze(0)
                #out_3 = self.pitchshift(out)
                #out = torch.cat((out, out_3), 0)
            #if torch.rand(1) < 1.:
            if a == 3:
                out = self.timeshift(out)                                                                                                                                                                          
                                                                                                                                                                                                                   
            out = self.mel_specgram(out)                                                                                                                                                                           
            out = torch.log(out+1e-8)
            out = self.deltas_deltas(out)                                                                                                                                                                          
                                                                                                                                                                                                                   
            if torch.rand(1) < 1.:
                out = self.crop(out, 400)
            #if torch.rand(1) < 1.:
            if a == 4:
                out = self.freqmask(out)                                                                                                                                                                           
            #if torch.rand(1) < 1.:
                out = self.timemask(out)                                                                                                                                                                           
        else:
            #out = self.pad_trunc_seq(out)
            out = self.mel_specgram(out)                                                                                                                                                                           
            out = torch.log(out+1e-8)
            #out = self.deltas_deltas(out)
            out = out.squeeze().permute(1,0)                                                                                                                                                                          
#            out = out
                                                                                                                                                                                                                   
        if self.p == True:
            out = self.crop(out, 400)
#            out = self.freqmask(out)
#            out = self.timemask(out)
        return out

class IEMOCAP(Dataset):
    def __init__(self, root_dir='/home/chenchengxin/dataset/IEMOCAP_full_release/'):
        self.data_info = pd.read_csv('/home/chenchengxin/spk_adaptiation_for_SER/meta/IEMOCAP_4.tsv', sep='\t')
        self.root_dir = root_dir                                                                                                                                                                                                                                                                                                                                                       
        self.label = self.data_info['label'].astype('category').cat.codes.values
        #self.speaker = self.data_info['person_id']
        #l = self.data_info['scene_label']
        #self.label = Series.as_matrix(l)                                                                                                                                                                                                    
        self.ClassNames = np.unique(self.data_info['label'])
        self.NumClasses = len(self.ClassNames)
        #self.audio_label = torch.zeros(len(self.data_info), NumClasses).scatter_(1, self.label, 1)
    
    def get_audio_dir_path_from_meta(self, filename):
        # Ses02F_script03_1_F006
        session = 'Session'+filename.split('_')[0][-2]
        dialog = '_'.join(filename.split('_')[:-1])
        audio_dir = os.path.join(self.root_dir, session,'sentences/wav',dialog)
        return audio_dir        

    def __len__(self):
        return len(self.data_info)
                                                                                                                                                                                                                   
    def __getitem__(self, idx):
        audio_name = os.path.join(self.get_audio_dir_path_from_meta(self.data_info.filename[idx]), self.data_info.filename[idx]+'.wav')
        wav, sample_rate = torchaudio.load(audio_name)
        wav = wav * (1 << 15)
        if sample_rate!=16000:
            wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wav)
        #spec = kaldi.fbank(wav,num_mel_bins=128,frame_length=64,frame_shift=16,sample_frequency=16000)
        spec = kaldi.fbank(wav,num_mel_bins=128,frame_length=40,frame_shift=10,sample_frequency=16000,high_freq=8000,low_freq=0,window_type='hamming')
        #len_spec = torch.tensor(spec.shape[0])
        l = self.data_info.filename[idx]

        return spec,l