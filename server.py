# This is a modified implementation of FedAvg from https://github.com/vaseline555/Federated-Learning-FedAvg-PyTorch
# Edited for Federated Transfer Learning

import torch
import copy
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm
from util import SmartphoneDataset
from client import Client
from collections import OrderedDict


class Central():
    def __init__(self, client_data, public_dataset, device, fed_config, model, optim_config, public_eval=None):
        self.clients = None
        self.local_clients = None
        self.round = 0
        
        self.spreaded_data = client_data
        self.fraction = fed_config["C"]
        self.num_rounds = fed_config["R"]
        self.max_epoch = fed_config["epoch"]
        self.batch_size = fed_config["batch_size"]
        self.client_val_split = fed_config["val_split"]
        
        self.device = device
        
        self.model = model
        
        self.lr_patience = optim_config["lr_patience"]
        self.criterion = optim_config["criterion"]
        self.coral_lambda = optim_config["lambda_coral"]
        self.optimizer = optim_config["name"]
        self.lr = optim_config["lr"]
        self.momentum = optim_config["momentum"]
        self.weight_decay = optim_config["weight_decay"]
        
        self.public_dataset = public_dataset
        self.public_eval = public_eval
        self.num_clients = None
        
    def setup(self):
        # can only call this if the round still 0
        assert self.round == 0
        
        # client dataset
        client_data = [
            SmartphoneDataset(self.spreaded_data[0][index], self.spreaded_data[1][index]) for index in range(len(self.spreaded_data[0]))
        ]
        
        # assign dataset to each client
        client_config = {
            "num_local_epochs": self.max_epoch,
            "criterion": self.criterion,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
            "val_split": self.client_val_split,
            "weight_decay": self.weight_decay,
            "lr_patience": self.lr_patience,
            "coral_lambda": self.coral_lambda
        }
        
        # create federated setting clients
        logging = f"Round {self.round}: \tSetup {len(client_data)} clients for Federated Setting"
        print(logging)
        
        self.clients = self.create_clients(client_data, client_config, connected_public_dataset=self.public_dataset)
        self.num_clients = len(self.clients)

        # send the model to all clients
        self.transmit_model()
        
    def create_clients(self, local_datasets, client_config, connected_public_dataset):
        
        clients = []
        logging = f"Round {self.round}: \tSetup all {len(local_datasets)} clients"
        print(logging)
        for idx, dataset in tqdm(enumerate(local_datasets), leave=False):
            client_id = idx
                
            client = Client(client_id=client_id, 
                            local_data=dataset, 
                            device=self.device,
                            connected_public_dataset= connected_public_dataset
                           )
            client.setup(client_config)
            clients.append(client)

        return clients
    
    def transmit_model(self, sampled_client_indices=None):
    
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self.round == 0) or (self.round == self.num_rounds)
            
            logging = f"Round {self.round}: \ttransmitt models to all {len(self.clients)} clients"
            print(logging)
            
            for client in tqdm(self.clients, leave=False):
                client.model = copy.deepcopy(self.model)
            
        else:
            # send the global model to selected clients
            assert self.round != 0
            
            logging = f"Round {self.round}: \ttransmit models to all {len(sampled_client_indices)} clients"
            print(logging)

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
            
    def sample_clients(self):
        # sample clients randommly
        logging = f"Round {self.round}: \tSelect clients..."
        print(logging)

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices
    
    def transfer_learning_clients(self, sampled_clients):
        logging = f"Round {self.round}: \tPerform Transfer Learning on {len(sampled_clients)} clients"
        print(logging)
        
        total_size = 0 
        for idx in tqdm(sampled_clients, leave = False):
            # finetune
            self.clients[idx].transfer_learning()
            total_size += len(self.clients[idx])
    
    def update_selected_clients(self, sampled_clients):
        logging = f"Round {self.round}: \tTrain on {len(sampled_clients)} clients"
        print(logging)
        
        total_size = 0 
        for idx in tqdm(sampled_clients, leave = False):
            # train
            self.clients[idx].local_update()
            total_size += len(self.clients[idx])
            
        return total_size
    
    def average_model(self, sampled_clients, coefficients):
        # federated averaging
        logging = f"Round {self.round}: \tAggregate updated weights from {len(sampled_clients)} clients"
        print(logging)

        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_clients), leave=False):

            if self.device=='cuda':
                local_weights = self.clients[idx].model.state_dict()           
            else:
                local_weights = self.clients[idx].model.state_dict()
                
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
                    
        self.model.load_state_dict(averaged_weights)
        
        logging = f"Round {self.round}: \tUpdated weights from {len(sampled_clients)} clients are complete"
        print(logging)
    
                
    def client_evalutions(self, sampled_clients = None):
        
        if sampled_clients is not None:
            for idx in tqdm(sampled_clients, leave = False):
                # do evalution
                loss, accuracy, recall, precission, f1score  = self.clients[idx].local_evaluation()
                
                logging = f"Round {self.round}: \t Client {self.clients[idx].id} Accuracy: {accuracy} \tRecall: {recall} \tPrecission:{precission} \tF1-score: {f1score}"
                print(logging)
        else:
            for client in tqdm(self.clients, leave=False):
                loss, accuracy, recall, precission, f1score  = client.local_evaluation()
                
                logging = f"Round {self.round}: \t Client {client.id} Accuracy: {accuracy} \tRecall: {recall} \tPrecission:{precission} \tF1-score: {f1score}"
                print(logging)
    
    def evaluate_global_model(self):
        self.model.eval()
        self.model.to(self.device)

        test_loss = 0.0
        criterion = self.criterion()
        
        y_pred = np.array([],dtype='i')
        y_truth = np.array([],dtype='i')
        with torch.no_grad():
            for data, target in self.valid_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                test_loss += criterion(outputs, target).item()
            
                y_pred = np.concatenate((y_pred, np.argmax(outputs.clone().detach().cpu().numpy(),axis=1)))
                y_truth = np.concatenate((y_truth, target.clone().detach().cpu().numpy()))
                
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.valid_dataloader.sampler)
        acc = accuracy_score(y_truth, y_pred)
        rec = recall_score(y_truth, y_pred, average='macro')
        prec = precision_score(y_truth, y_pred, average='macro')
        f1 = f1_score(y_truth, y_pred, average='macro')
        
        return test_loss, acc, rec, prec, f1
    
    def print_best_performance(self):
        for client in tqdm(self.clients, leave=False):
            logging = f"Round {self.round}: \t Client {client.id} Accuracy: {client.best_performance}"
            print(logging)
    
    def train_federated_model(self):
        sampled_clients = self.sample_clients()
        
        self.transmit_model(sampled_clients)
        
        # train locally
        selected_total_size = self.update_selected_clients(sampled_clients)
        
        mixing_coefficients = [1.0 / len(sampled_clients) for idx in sampled_clients]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_clients, mixing_coefficients)
        
        # transmit again
        self.transmit_model(sampled_clients)
        
        # perform transfer learning
        self.transfer_learning_clients(sampled_clients)
        
        # evaluate clients
        self.client_evalutions(sampled_clients)
                                                       
    def train(self):
        
        # Global model Performance without federated transfer
        self.client_evalutions()
        
        for rond in range(self.num_rounds):
            
            self.round += 1
            
            # do training
            self.train_federated_model()
            
            # evaluate on validation dataset if available
            if self.public_eval:
                loss, accuracy, recall, precission, f1score  = self.evaluate_global_model()
                
                print(f'Round {self.round}: Global Loss:{loss} \tAccuracy:{accuracy} \tF1-Score:{f1score} \tPrecission:{precission} \tRecall:{recall}')
            
            logging = f"Round {self.round}: \tRound Finished..."
            print(logging)
            
        # print result
        self.print_best_performance()