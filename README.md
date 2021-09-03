# HAR Federated Transfer Learning in PyTorch [![arXiv](https://img.shields.io/badge/arXiv-1907.05629-f9f107.svg)](https://arxiv.org/abs/1907.09173)

This is an unofficial implementation of Federated Transfer Learning using UCI Smarthphone dataset from FedHealth [paper](https://arxiv.org/abs/1907.09173)
You can download the dataset at [here](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

## Setup

- You need to first download and extract the dataset (example put it in `./data`)
- Generate the training CSV
```sh
python create_csv.py --path UCI_HAR_DATASET_lOC
```
- Run training with
```sh
python main.py
```
- Using trained global model uses
```sh
python main.py --global_model_path ./global_model/saved_model.pt
```
## Configuration

| Parameter | Description |
| ------ | ------ |
| csv_path | CSV data path |
| round | Round for federated learning |
| internal_epoch | Internal epoch of each client |
| global_model_path | Trained global model path |
| batch_size' | Batch size used |
| lr | Learning rate used |
| C | Fraction of client for each round averaging |
| val_split | Validation split for test |
| lambda_coral | trade off parameter in CORAL loss |
| momentum | momentum for SGD |
| weight_decay' | weight decay for SGD |
| lr_patience | learning patience before reduced when loss does not go down in each client |

## Reference

This implementation use some code from:
- https://github.com/vaseline555/Federated-Learning-FedAvg-PyTorch
- https://github.com/SSARCandy/DeepCORAL


