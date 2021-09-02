# This is a modified implementation of FedAvg from https://github.com/vaseline555/Federated-Learning-FedAvg-PyTorch
# Edited for Federated Transfer Learning
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from coral import CORAL
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class Client():
    def __init__(self, client_id, local_data, device, connected_public_dataset = None):
        self.id = client_id
        self.data = local_data
        self.device = device
        self.best_acc = 0.0
        self.lr_counter = 0
        self.coral_lambda = 0.01
        self.best_performance = {
            "loss": 0.0,
            "acc": 0.0,
            "rec": 0.0,
            "prec": 0.0,
            "f1": 0.0
        }
        self.connected_public_dataset = connected_public_dataset
        self.__model = None # private can't be accessed
        self.__lr = None
        
    @property
    def lr(self):
        return self.__lr
    
    @lr.setter
    def lr(self, lr):
        self.__lr = lr
        
    @property
    def model(self):
        # get model
        return self.__model

    @model.setter
    def model(self, model):
        # to set private model
        self.__model = model
            
    
    def setup(self, client_config):
        
        if self.device == 'cuda':
            pin_memory = True
        else:
            pin_memory = False
        
        labels = []
        for _, label in self.data:
            labels.append(label)

        train_idx, valid_idx= train_test_split(
            np.arange(len(labels)), test_size=client_config["val_split"], random_state=42,stratify=labels)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        self.dataloader = DataLoader(self.data, 
                                        batch_size=client_config["batch_size"],
                                        pin_memory= pin_memory,
                                        num_workers=1, 
                                        sampler=train_sampler)
        
        self.test_dataloader = DataLoader(self.data, 
                                            batch_size=1, 
                                            pin_memory= pin_memory,
                                            num_workers=1,  
                                            sampler=valid_sampler)
            
        self.local_epoch = client_config["num_local_epochs"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.lr = client_config["lr"]
        self.momentum = client_config["momentum"]
        self.weight_decay = client_config["weight_decay"]
        self.lr_patience = client_config["lr_patience"]
        self.coral_lambda = client_config["coral_lambda"]
    
    def __len__(self):
        """Total local data."""
        return len(self.dataloader.sampler)
                    
    def transfer_learning(self):
        # use coral and calculate forward with local model on public dataset and own dataset
        self.model.train()
        self.model.to(self.device)
        
        opt = self.optimizer(self.model.parameters(),
                             lr=self.lr, 
                             weight_decay= self.weight_decay, 
                             momentum = self.momentum)
        criterion = self.criterion()
        
        for epoch in range(self.local_epoch):
            client_dataset = iter(self.dataloader)
            for source_img, source_label in self.connected_public_dataset:
                try:
                    target_img, target_label = next(client_dataset)
                except StopIteration:
                    client_dataset = iter(self.dataloader)
                    target_img, target_label = next(client_dataset)

                source_img = source_img.to(self.device)
                source_label = source_label.to(self.device)
                target_img = target_img.to(self.device)
                target_label = target_label.to(self.device)
                
                opt.zero_grad()
                
                # forward
                out_source = self.model(source_img)
                out_target = self.model(target_img)
                
                # calculate loss
                cl_loss_source = criterion(out_source, source_label)
                cl_loss_target = criterion(out_target, target_label)
                
                # calculate CORAL
                coral_loss = CORAL(out_source, out_target)
                
                total_loss =  self.coral_lambda * coral_loss + cl_loss_target + cl_loss_source
                
                total_loss.backward()
                
                opt.step()
                
        self.model.to("cpu")
                
    def local_update(self):
        self.model.train()
        self.model.to(self.device)
        
        # update learning rate
        if self.lr_counter > self.lr_patience:
            self.lr = self.lr/10
            logging = f"\tClient{self.id} decreasing learning rate to {self.lr}"
            print(logging)
            self.lr_counter = 0
        
        opt = self.optimizer(self.model.parameters(),
                             lr=self.lr, 
                             weight_decay= self.weight_decay, 
                             momentum = self.momentum)
        criterion = self.criterion()
        
        for epoch in range(self.local_epoch):
            for data,target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)

                opt.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)

                loss.backward()
                opt.step() 
        
        self.model.to("cpu")
        
    def local_evaluation(self):
        self.model.eval()
        self.model.to(self.device)
        
        loss = 0.0
        criterion = self.criterion()
        
        y_pred = np.array([],dtype='i')
        y_truth = np.array([],dtype='i')
        with torch.no_grad():
            for data,target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                loss += criterion(outputs, target).item()*data.size(0)

                y_pred = np.concatenate((y_pred, np.argmax(outputs.clone().detach().cpu().numpy(),axis=1)))
                y_truth = np.concatenate((y_truth, target.clone().detach().cpu().numpy()))

                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")
        
        test_loss = loss / len(self.dataloader)
        acc = accuracy_score(y_truth, y_pred)
        rec = recall_score(y_truth, y_pred, average='macro')
        prec = precision_score(y_truth, y_pred, average='macro')
        f1 = f1_score(y_truth, y_pred, average='macro')
        
        # decrease lr if counter pass lr_patience
        if acc > self.best_acc:
            self.best_acc = acc
            # record performance
            self.best_performance['loss'] = test_loss
            self.best_performance['acc'] = acc
            self.best_performance['rec'] = rec
            self.best_performance['prec'] = prec
            self.best_performance['f1'] = f1
        # save model if in test mode
            if self.device == 'cuda':
                torch.save(self.model.state_dict(), f'client_model/UCI_{self.id}.pt')
            else:
                torch.save(self.model.state_dict(), f'client_model/UCI_{self.id}.pt')
            self.lr_counter = 0
        else:
            self.lr_counter +=1
            
        return test_loss, acc, rec, prec, f1