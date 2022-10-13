from __future__ import print_function, division

import json
import os
import ssl
import sys

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
from data_loader import get_loader
from data_preparation import DataPreparation
from data_transforms import test_transforms
from metrics import _group_by_patients
from model import ClotModelSingle


ssl._create_default_https_context = ssl._create_unverified_context
cudnn.benchmark = True


def _get_model_path_by_center_id(folder_name: str, center_id: str) -> str:
    for file_name in os.listdir(folder_name):
        if not file_name.endswith('.h5'):
            continue
        if file_name.split('_')[2] == center_id:
            return os.path.join(folder_name, file_name)
    raise Exception(f'Model for center id {center_id} was not found in folder {folder_name}')


def main() -> None:
    settings_file_path = 'SETTINGS.json'
    if len(sys.argv) > 1:
        settings_file_path = sys.argv[1]
    with open(settings_file_path, 'r') as file:
        settings = json.load(file)

    data_prep = DataPreparation(settings['INPUT_DATA_PATH'])
    image_crops, _ = data_prep.process_test()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataloader = get_loader(
        [image_id for image_id, _, _ in data_prep.test],
        [label for _, label, _ in data_prep.test],
        image_crops,
        seed=41,
        is_test=True,
        transformations=test_transforms,
        shuffle=False,
        batch_size=16,
        num_workers=2,
    )

    center_ids = ['.'.join([str(x) for x in group]) for group in config.CENTER_GROUPS]
    models = [
        torch.load(
            _get_model_path_by_center_id(settings['MODELS_PATH'], center_id),
            map_location=torch.device('cpu'),
        ).to(device)
        for center_id in center_ids
    ]
    for model in models:
        model.eval()

    with torch.no_grad():
        y_hat, y, image_ids = [], [], []
        for image, label, image_id in tqdm(dataloader):
            image = image.to(device)
            label = label.cpu().detach().numpy().tolist()
            for model in models:
                y_hat.extend(model.forward(image).squeeze().cpu().detach().numpy().tolist())
                y.extend(label)
                image_ids.extend(image_id)

    bad_image_ids = {
        image_id
        for image_id, crops in zip([image_id for image_id, _, _ in data_prep.test], image_crops)
        if len(crops) == 0
    }
    y_hat_fixed = [
        0.5 if image_id in bad_image_ids else p
        for p, image_id in zip(y_hat, image_ids)
    ]

    labels, preds, patients = _group_by_patients(y, y_hat_fixed, image_ids)
    result = pd.DataFrame({
        'patient_id': patients,
        'CE': [pair[1].round(6) for pair in preds],
        'LAA': [pair[0].round(6) for pair in preds],
    })
    print(result)
    result.to_csv(os.path.join(settings['SUBMISSION_PATH'], 'submission.csv'), index=False)


if __name__ == "__main__":
    main()
