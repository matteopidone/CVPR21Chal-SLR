import pandas as pd
from scipy.io import savemat

def from_csv_to_mat(train_csv_path, test_csv_path, output_mat_path):
    # Funzione per filtrare e modificare i nomi dei file
    def process_file_names(df):
        # Filtra i file che non finiscono con 'depth.mkv'
        filtered_df = df[~df[0].str.endswith('depth.mkv')]
        # Rimuovi '_rgb.mkv' dalla fine dei nomi dei file
        filtered_df[0] = filtered_df[0].str.replace('_rgb.mkv', '', regex=False)
        return filtered_df

    # Leggi il file CSV di addestramento e processa i nomi dei file
    train_df = pd.read_csv(train_csv_path, header=None)
    train_df = process_file_names(train_df)
    # Estrai i nomi dei file e le etichette
    train_file_name = train_df[0].values.reshape(-1, 1)
    train_label = train_df[1].values.reshape(-1, 1)
    # Calcola il numero di dati di addestramento
    train_count = len(train_df)

    # Leggi il file CSV di test e processa i nomi dei file
    test_df = pd.read_csv(test_csv_path, header=None)
    test_df = process_file_names(test_df)
    # Estrai i nomi dei file e le etichette
    test_file_name = test_df[0].values.reshape(-1, 1)
    test_label = test_df[1].values.reshape(-1, 1)
    # Calcola il numero di dati di test
    test_count = len(test_df)

    # Prepara il dizionario da salvare nel file .mat
    mat_dict = {
        'train_file_name': train_file_name,
        'train_label': train_label,
        'train_count': [[train_count]],
        'test_file_name': test_file_name,
        'test_label': test_label,
        'test_count': [[test_count]]
    }

    # Salva il dizionario in un file .mat
    savemat(output_mat_path, mat_dict)

train_csv_path = '/home/perceive/slr/rgbd/CVPR21Chal-SLR/train_sstcn.csv'
test_csv_path = '/home/perceive/slr/rgbd/CVPR21Chal-SLR/val_sstcn.csv'
output_mat_path = '/home/perceive/slr/rgbd/CVPR21Chal-SLR/data.mat'

from_csv_to_mat(train_csv_path, test_csv_path, output_mat_path)
