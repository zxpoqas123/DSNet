from data2 import IEMOCAP
import pickle
import os

import torch

from torch.nn.utils.rnn import pad_sequence

'''
class ENC(torch.nn.Module):
    def __init__(self, input_dim, global_cmvn, config):
        super().__init__()
        self.encoder = ConformerEncoder(input_dim,global_cmvn=global_cmvn,**config)
    def forward(self, speech, speech_lengths):
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        return encoder_out, encoder_mask

symbol_table = read_symbol_table('/home/chenchengxin/wenet/examples/librispeech/s0_pretrained/20210811_conformer_bidecoder_exp/words.txt')
with open('/home/chenchengxin/wenet/examples/librispeech/s0_pretrained/20210811_conformer_bidecoder_exp/train.yaml', 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
cmvn_file = '/home/chenchengxin/wenet/examples/librispeech/s0_pretrained/20210811_conformer_bidecoder_exp/global_cmvn'
mean, istd = load_cmvn(cmvn_file,True)
global_cmvn = GlobalCMVN(torch.from_numpy(mean).float(),torch.from_numpy(istd).float())
encoder = ENC(input_dim=input_dim, global_cmvn=global_cmvn ,config=configs['encoder_conf'])


configs['input_dim'] = input_dim
configs['output_dim'] = len(symbol_table)
configs['cmvn_file'] = cmvn_file
configs['is_json_cmvn'] = True

model = init_asr_model(configs)
ckpt = torch.load('/home/chenchengxin/wenet/examples/librispeech/s0_pretrained/20210811_conformer_bidecoder_exp/final.pt')
model.load_state_dict(ckpt)

pretrained_dict = model.state_dict()
model_dict = encoder.state_dict()
print('copied dict :\n{}'.format(pretrained_dict))
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
encoder.load_state_dict(model_dict)

torch.save(encoder.state_dict(), '/home/chenchengxin/dataset/preprocessed/pretrained_asr/IEMOCAP/asr_encoder.pt') 
'''
train_set = IEMOCAP()
#target_root = '/home/chenchengxin/NCMMSC2022/dataset/IEMOCAP_FBank_80.pkl'
target_root = '/home/chenchengxin/spk_adaptiation_for_SER/dataset/IEMOCAP_FBank_128.pkl'

dic = {}
count = 0
dpc_ls = []

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

for i in range(len(train_set)):
    spec,fn = train_set[i]
    dic[fn] = spec.numpy()
    count+=1
    if count%500 == 0:
        print('preprocessed: {}'.format(count)) 

with open(target_root, 'wb') as f:
    pickle.dump(dic, f)


'''
#norm

def calculate_mean_norm(x_ls):
    x = np.concatenate(x_ls)
    mean_ = np.mean(x,0)
    std_ = np.std(x,0)
    return [(x-mean_)/std_ for x in x_ls]


target_root = '/home/chenchengxin/dataset/preprocessed/mel_spectrogram/IEMOCAP_sr16000_fft400_hop160_mels40.pkl'
target_root_2 = '/home/chenchengxin/dataset/preprocessed/mel_spectrogram/IEMOCAP_sr16000_fft400_hop160_mels40_spk_normalization.pkl'    
with open(target_root, 'rb') as f:
    dic = pickle.load(f)
df = pd.read_csv('/home/chenchengxin/cross_corpus_new/meta/IEMOCAP_4.tsv', sep='\t')
spk_ls = np.unique(df.speaker)
dic_new = {}
for spk in spk_ls:
    df_tmp = df[df.speaker==spk].reset_index(drop=True)
    spec_ls = [dic[x] for x in df_tmp.filename]
    norm_spec_ls = calculate_mean_norm(spec_ls)
    dic_tmp = {}
    for i in range(len(df_tmp)):
        dic_tmp[df_tmp.filename[i]] = norm_spec_ls[i]
    dic_new.update(dic_tmp)
with open(target_root_2, 'wb') as f:
    pickle.dump(dic_new, f)   
'''
'''
target_root = '/home/chenchengxin/dataset/preprocessed/mel_spectrogram/IEMOCAP_sr16000_fft400_hop160_mels40.pkl'
target_root_3 = '/home/chenchengxin/dataset/preprocessed/mel_spectrogram/IEMOCAP_sr16000_fft400_hop160_mels40_global_normalization.pkl'    
with open(target_root, 'rb') as f:
    dic = pickle.load(f)
df = pd.read_csv('/home/chenchengxin/cross_corpus_new/meta/IEMOCAP_4.tsv', sep='\t')

dic_new = {}
spec_ls = [dic[x] for x in df.filename]
norm_spec_ls = calculate_mean_norm(spec_ls)
for i in range(len(df)):
    dic_new[df.filename[i]] = norm_spec_ls[i]
with open(target_root_3, 'wb') as f:
    pickle.dump(dic_new, f)   
'''    