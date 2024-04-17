import os
import csv

def copia_righe_corrispondenti(dir_path, csv_input_path, csv_output_path):
    # Leggere i dati dal file CSV iniziale e memorizzarli in un dizionario con chiave il nome base
    dati = {}
    with open(csv_input_path, mode='r', encoding='utf-8') as file_in:
        reader = csv.reader(file_in)
        for riga in reader:
            nome_base_csv = riga[0].rsplit('.', 1)[0]  # Rimuovere l'estensione .mkv per confrontare i nomi base
            dati[nome_base_csv] = riga

    # Creare il file CSV di output
    with open(csv_output_path, mode='w', encoding='utf-8', newline='') as file_out:
        writer = csv.writer(file_out)

        # Ottenere un elenco ordinato dei file nella directory e esaminarli, ignorando i file "_flip"
        files_ordinati = sorted(os.listdir(dir_path))
        for filename in files_ordinati:
            if '_flip' not in filename and filename.endswith('.pt'):
                nome_base_file = os.path.splitext(filename)[0]  # Rimuovere l'estensione .pt
                # Cerca il nome base nel dizionario e scrivi la riga corrispondente nel CSV di output
                if nome_base_file in dati:
                    writer.writerow(dati[nome_base_file])

if __name__ == '__main__':
    dir_path = '/home/perceive/slr/rgbd/data/test_wholepose_feature'
    csv_input_path = '/home/perceive/slr/rgbd/CVPR21Chal-SLR/test.csv'
    csv_output_path = '/home/perceive/slr/rgbd/CVPR21Chal-SLR/test_sstcn.csv'
    copia_righe_corrispondenti(dir_path, csv_input_path, csv_output_path)
