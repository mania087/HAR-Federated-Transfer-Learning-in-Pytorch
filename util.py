import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SmartphoneDataset(Dataset):
    def __init__(self, list_of_data, label):
        self.data = list_of_data
        self.labels = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx])
        label = self.labels[idx]
        
        return data, label

def load_subjects_data(csv, list_of_subjects):
    # this functions load data of each subject in list_of_subjects
    # and return the list of subjects data
    data = []
    label = []
    
    df = pd.read_csv(csv)
    
    # turn label into 0 for the first label 
    df.loc[:, "label"] = df["label"].apply(lambda x: x - 1)
    # acc
    df['body_acc_x'] = df['body_acc_x'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_acc_y'] = df['body_acc_y'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_acc_z'] = df['body_acc_z'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    # gyro
    df['body_gyro_x'] = df['body_gyro_x'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_gyro_y'] = df['body_gyro_y'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['body_gyro_z'] = df['body_gyro_z'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    # total acc
    df['total_acc_x'] = df['total_acc_x'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['total_acc_y'] = df['total_acc_y'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    df['total_acc_z'] = df['total_acc_z'].apply(lambda x: np.array(x.replace('  ', ' ').strip().split(' '), dtype='f'))
    
    for x in list_of_subjects:
        client_pd = df.loc[df['subject'] == x]
        client_label = client_pd["label"].to_numpy() 
        client_data = np.transpose(np.apply_along_axis(np.stack, 1, client_pd.drop(["label","subject"], axis=1).to_numpy()),(0,1,2))
        data.append(client_data)
        label.append(client_label)
    
    return data,label
    