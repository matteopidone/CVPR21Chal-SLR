import torch
import os
import numpy as np
from T_Pose_model import *
import torch.nn.parallel
import pickle
import csv
from tqdm import tqdm

if __name__ == "__main__":
    dataset_path = "/home/perceive/slr/rgbd/data/test_wholepose_feature"
    checkpoint_model = "/home/perceive/slr/rgbd/CVPR21Chal-SLR/SSTCN/model_checkpoints/T_Pose_model_16_96.77083333333333.pth"

    print(f"dataset_path: {dataset_path}, checkpoint_model: {checkpoint_model}")

    test_files = []
    test_labels = []
    with open('/home/perceive/slr/rgbd/CVPR21Chal-SLR/test_sstcn.csv', mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            test_files.append(row[0])
            test_labels.append(int(row[1]))  # Aggiungi anche le etichette vere
    test_files = np.array(test_files)
    test_labels = np.array(test_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T_Pose_model(frames_number=60, joints_number=33, n_classes=156)
    model = model.to(device)
    
    if checkpoint_model:
        model.load_state_dict(torch.load(checkpoint_model, map_location=device))
    else:
        model.init_weights()
    model.eval()

    correct_predictions = 0
    preds = []
    names = []
    for name, label in tqdm(zip(test_files, test_labels), total=len(test_files)):
        names.append(name)
        fea_name = name + '_rgb.pt'
        fea_path = os.path.join(dataset_path, fea_name)
        data = torch.load(fea_path)
        data = data.contiguous().view(1, -1, 24, 24)
        data_in = data.to(device)
        with torch.no_grad():
            pred = model(data_in)
            pred_label = torch.argmax(pred, dim=1)
            correct_predictions += (pred_label.cpu() == label).sum().item()
        pred = pred.cpu().detach().numpy()
        preds.append(pred)
    
    accuracy = correct_predictions / len(test_labels)
    print(f'Accuracy: {accuracy:.4f}')
    
    with open('./T_Pose_model_test.pkl', 'wb') as f:
        score_dict = dict(zip(names, preds))
        pickle.dump(score_dict, f)
