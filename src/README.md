## DSNet

### Model training:

bash run.sh

### Model evaluation:

bash test.sh (within-corpus)
bash test_cross.sh (cross-corpus)

### Configuration statement:

gpu_id=0 (indicating which gpu to use) 

dataset_type= IEMOCAP_4 (indicating the dataset for model training)

max_len=6 (indicating the max length of speech in seconds)

seed=12 (indicating the random seed)
