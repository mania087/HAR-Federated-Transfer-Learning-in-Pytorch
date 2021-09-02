# This implementation follows the Federated Transfer Learning 
# Idea from https://arxiv.org/abs/1907.09173
import torch
import numpy as np
from model import SimpleCNN
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from util import *

CSV_PATH = '../../dataset/UCI_smartphone/UCI_Smartphone_Raw.csv'
TRAIN_ON_GPU = torch.cuda.is_available()
PUBLIC_DATASET= range(1,26)
CLIENT_DATASET = [26,27,28,29,30]
BATCH_SIZE = 64
    
if __name__ == '__main__':
    # load public dataset
    public_data, public_label = load_subjects_data(CSV_PATH, PUBLIC_DATASET)
    public_data = np.concatenate([np.array(i) for i in public_data])
    public_label = np.concatenate([np.array(i) for i in public_label])
    
    print(f'Public Data shape: {public_data.shape}')
    
    public_dataset = SmartphoneDataset(public_data, public_label)
    
    # split into training and validation
    train_idx, valid_idx= train_test_split(
        np.arange(len(public_label)), test_size=0.3, random_state=42, shuffle=True, stratify=public_label)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    trainloader = torch.utils.data.DataLoader(public_dataset,
                                              batch_size=BATCH_SIZE,
                                              sampler=train_sampler, num_workers=1)

    validloader = torch.utils.data.DataLoader(public_dataset,
                                              batch_size=1,
                                              sampler=valid_sampler, num_workers=1)
    
    # load client dataset
    client_data, client_label = load_subjects_data(CSV_PATH, CLIENT_DATASET)
    
    # set model
    model = SimpleCNN(9,6)
    
    
    