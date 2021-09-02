# This implementation follows the Federated Transfer Learning 
# Idea from https://arxiv.org/abs/1907.09173
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
from model import SimpleCNN
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch.utils.data import ConcatDataset
from util import *

CSV_PATH = '../../dataset/UCI_smartphone/UCI_Smartphone_Raw.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PUBLIC_DATASET= range(1,26)
CLIENT_DATASET = [26,27,28,29,30]
BATCH_SIZE = 64
LEARNING_RATE = 0.01

def train(n_epochs, trainloader, validloader, model, optimizer, criterion, save_dir = 'global_model/Global_CNN.pt'):
    
    model_name = save_dir
    scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5,verbose=True,min_lr=1e-10)
    
    valid_loss_min = np.Inf # track change in validation loss
    epoch_train_loss =[]
    epoch_val_loss =[]
    
    for epoch in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        # train
        model.train()
        for data, target in trainloader:
            # move tensors to GPU if CUDA is available
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        
        # validation
        model.eval()
        y_pred = np.array([],dtype='i')
        y_truth = np.array([],dtype='i')
        with torch.no_grad():
            for data, target in validloader:
                # move tensors to GPU if CUDA is available
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()*data.size(0)
                y_pred = np.concatenate((y_pred, np.argmax(output.clone().detach().cpu().numpy(),axis=1)))
                y_truth = np.concatenate((y_truth, target.clone().detach().cpu().numpy()))
        
        # calculate average losses
        train_loss = train_loss/len(trainloader.sampler)
        valid_loss = valid_loss/len(validloader.sampler)
        epoch_train_loss.append(train_loss)
        epoch_val_loss.append(valid_loss)
        
        acc = accuracy_score(y_truth, y_pred)
        rec = recall_score(y_truth, y_pred, average='macro')
        prec = precision_score(y_truth, y_pred, average='macro')
        f1 = f1_score(y_truth, y_pred, average='macro')
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tAccuracy: {:.2f} \tF1-Score: {:.2f}'.format(
        epoch, train_loss, valid_loss, acc, f1))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), model_name)
            valid_loss_min = valid_loss
            
        scheduler.step(valid_loss)
    
    return 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='Resume from checkpoint file')
    parser.add_argument('--csv_path', type=str, default='../../dataset/UCI_smartphone/UCI_Smartphone_Raw.csv')
    parser.add_argument('--global_model_path', type=str, default ='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    # load public dataset
    public_data, public_label = load_subjects_data(args.csv_path, PUBLIC_DATASET)
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
                                              batch_size=args.batch_size,
                                              sampler=train_sampler, num_workers=1)

    validloader = torch.utils.data.DataLoader(public_dataset,
                                              batch_size=1,
                                              sampler=valid_sampler, num_workers=1)
    
    # load client dataset
    client_data, client_label = load_subjects_data(args.csv_path, CLIENT_DATASET)
    
    # set model
    model = SimpleCNN(9,6)
    if args.global_model_path:
        print('Loaded Previous Global Model...')
        model.load_state_dict(torch.load(args.global_model_path))
    else:
        print('Training global model first...')
        # send model
        model.to(DEVICE)
        # set optimizer and criterion
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min',patience=5,verbose=True,min_lr=1e-10)
        
        # train global model if no weight is loaded
        train(80, trainloader, validloader, model, optimizer, criterion)
        
        model.cpu()
        model.load_state_dict(torch.load('global_model/Global_CNN.pt'))
    
    ## Federated Transfer Learning Setup
    