import os
import argparse
import pandas as pd

COLUMNS = [
           'body_acc_x', 
           'body_acc_y',
           'body_acc_z',
           'body_gyro_x',
           'body_gyro_y',
           'body_gyro_z',
           'total_acc_x',
           'total_acc_y',
           'total_acc_z'
           ]

def read_txt (column_name):
    train_pd = pd.read_csv(os.path.join(data_path, r"train/Inertial Signals", f'{column_name}_train.txt'), header=None, names=[column_name])
    test_pd = pd.read_csv(os.path.join(data_path, r"test/Inertial Signals", f'{column_name}_test.txt'), header=None, names=[column_name])
    
    returned_columns = pd.concat([train_pd, test_pd], axis=0)
    
    return returned_columns

if __name__ == '__main__':
    data_path = r"data/UCI HAR Dataset/UCI HAR Dataset"
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=r"data/UCI HAR Dataset/UCI HAR Dataset", help='UCI dataset data path')
    parser.add_argument('--save', type=str, default='data/UCI_Smartphone_Raw.csv', help='save file name')
    args = parser.parse_args()
    
    data_path = args.path
    # read train subjects
    train_subjects = pd.read_csv(os.path.join(data_path, 'train/subject_train.txt'), header=None, names=['subject'])
    # read test subjects
    test_subjects = pd.read_csv(os.path.join(data_path, 'test/subject_test.txt'), header=None, names=['subject'])
    # concat
    subjects = pd.concat([train_subjects, test_subjects], axis=0)
    
    # read train labels
    train_labels = pd.read_csv(os.path.join(data_path, 'train/y_train.txt'), header=None, names=['label'])
    # read train labels
    test_labels = pd.read_csv(os.path.join(data_path, 'test/y_test.txt'), header=None, names=['label'])
    # labels
    labels = pd.concat([train_labels, test_labels], axis=0)
    
    final_dataframe = pd.concat([subjects, labels], axis=1)
    
    data = []
    for name in COLUMNS:
        final_dataframe = pd.concat([final_dataframe, read_txt(name)], axis=1)
        
    final_dataframe.to_csv(args.save,index=False)
    