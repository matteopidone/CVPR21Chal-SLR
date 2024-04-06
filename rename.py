import os
import re
import csv

# Assumendo che le mappature siano caricate o inizializzate come prima
try:
    from mappature import segno_to_number
except ImportError:
    segno_to_number = {}

segno_to_progressivo = {}
RIFERIMENTOCARTELLA = "9"

def update_name(segno, riferimento_cartella):
    if segno not in segno_to_number:
        segno_to_number[segno] = len(segno_to_number)
    if segno not in segno_to_progressivo:
        segno_to_progressivo[segno] = len(segno_to_progressivo)
    return f"signer{riferimento_cartella}_sample{segno_to_progressivo[segno]}"

directory = "/home/perceive/ProgettoRadarLIS_onlyRGB-D/20220816/10"
csv_file_path = "/home/perceive/ProgettoRadarLIS_onlyRGB-D/train.csv"  # Percorso dove salvare il file CSV


with open(csv_file_path, mode='a', newline='') as file:
    csv_writer = csv.writer(file)

    file_list = sorted(os.listdir(directory))

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
                full_new_path = os.path.join(directory, new_name)
                os.rename(os.path.join(directory, filename), full_new_path)
                # Scrive nel file CSV
                csv_writer.writerow([new_name, segno_to_number[segno]])
            else:
                print(f"Nome file non valido: {filename}")

# Salva le mappature aggiornate in un file Python
with open("mappature.py", "w") as f:
    f.write(f"segno_to_number = {segno_to_number}\n")
    # Non salviamo segno_to_progressivo poiché è specifico per questa esecuzione

print("Rinomina completata e mappature aggiornate.")
