import os
import re
import csv

try:
    from mappature import segno_to_number
except ImportError:
    segno_to_number = {}

segno_to_progressivo = {}

def update_name(segno, riferimento_cartella):
    if segno not in segno_to_number:
        segno_to_number[segno] = len(segno_to_number)
    if segno not in segno_to_progressivo:
        segno_to_progressivo[segno] = len(segno_to_progressivo)
    return f"signer{riferimento_cartella}_sample{segno_to_progressivo[segno]}"

# Directory principale contenente le cartelle numeriche
root_directory = "/home/perceive/ProgettoRadarLIS_onlyRGB-D/20220829"
csv_file_path = "/home/perceive/ProgettoRadarLIS_onlyRGB-D/train_2.csv"  # Percorso dove salvare il file CSV

# Aprire una volta il file CSV per scriverci in append
with open(csv_file_path, mode='a', newline='') as file:
    csv_writer = csv.writer(file)

    # Ciclare attraverso le sottocartelle di root_directory
    for folder_name in sorted(os.listdir(root_directory)):
        folder_path = os.path.join(root_directory, folder_name)
        if os.path.isdir(folder_path) and folder_name.isdigit():  # Controllare se è una cartella e il nome è numerico
            RIFERIMENTOCARTELLA = folder_name  # Aggiornare il riferimento alla cartella corrente
            segno_to_progressivo = {}  # Reset di segno_to_progressivo per ogni nuova cartella

            file_list = sorted(os.listdir(folder_path))
            for filename in file_list:
                if filename.endswith(".mkv"):
                    parts = re.match(r"(.+?)_realsense_(depthMapAlignedRGBFiltered|rgb)_(\d{8}_\d{6})\.mkv", filename)
                    if parts:
                        segno, tipo, sample_code = parts.groups()
                        new_name = update_name(segno, RIFERIMENTOCARTELLA)
                        if tipo == "depthMapAlignedRGBFiltered":
                            new_name += "_depth.mkv"
                        else:  # Gestione file RGB
                            new_name += "_rgb.mkv"
                        full_new_path = os.path.join(folder_path, new_name)
                        os.rename(os.path.join(folder_path, filename), full_new_path)
                        # Scrive nel file CSV
                        csv_writer.writerow([new_name, segno_to_number[segno]])
                    else:
                        print(f"Nome file non valido: {filename}")

# Salva le mappature aggiornate in un file Python
with open("mappature.py", "w") as f:
    f.write(f"segno_to_number = {segno_to_number}\n")
    # Non salviamo segno_to_progressivo poiché è specifico per questa esecuzione e viene resettato per ogni cartella

print("Rinomina completata e mappature aggiornate.")
