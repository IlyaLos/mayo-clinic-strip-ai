from __future__ import print_function, division

import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config
from data_loader import get_loader
from data_preparation import DataPreparation
from data_transforms import test_transforms, train_transforms
from metrics import get_target_metric
from model import ClotModelSingle


ssl._create_default_https_context = ssl._create_unverified_context
cudnn.benchmark = True

DUMPED_DATALOADER_PATH = '/kaggle/input/track-4-dataprep/data_loaders.pkl'
DUMPED_DATALOADER_OTHER_PATH = '/kaggle/input/track-4-dataprep/data_loaders_other.pkl'
CENTER_GROUPS = [(11,), (4,), (7,), (1, 5,), (10, 3), (6, 2, 8, 9,)]


def _get_sub_data(data, image_crops, image_crops_indices, sample_ids):
    return [data[i][0] for i in sample_ids], \
           [data[i][1] for i in sample_ids], \
           [data[i][2] for i in sample_ids], \
           [image_crops[i] for i in sample_ids], \
           [image_crops_indices[i] for i in sample_ids]


def remove_bad_model_files() -> None:
    model_files = [
        file_name
        for file_name in list(os.listdir('/kaggle/working/models'))
        if file_name.endswith('.h5')
    ]

    centers_to_models = defaultdict(list)
    for model_file in model_files:
        center_id = model_file.split('_')[2]
        epoch = int(model_file.split('_')[4])
        metric = float(model_file.split('_')[6][:-3])

        centers_to_models[center_id].append((metric, epoch, model_file))

    center_id_to_best_model_file_name = {
        center_id: sorted(model_files, key=lambda x: (x[0], x[1]))[0][2]
        for center_id, model_files in centers_to_models.items()
    }
    print(center_id_to_best_model_file_name)

    good_models = set(list(center_id_to_best_model_file_name.values()))
    for model_file in model_files:
        if model_file not in good_models:
            os.remove(os.path.join('/kaggle/working/models', model_file))


def main() -> None:
    data_prep = DataPreparation()

    with open(DUMPED_DATALOADER_PATH, 'rb') as file:
        image_crops, image_crops_indices = pickle.load(file)
    with open(DUMPED_DATALOADER_OTHER_PATH, 'rb') as file:
        image_crops_other, image_crops_indices_other = pickle.load(file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data = data_prep.train + data_prep.other
    image_crops += image_crops_other
    image_crops_indices += image_crops_indices_other

    all_metrics = []
    all_y, all_y_hat, all_image_ids = [], [], []
    for test_centers_group in CENTER_GROUPS:
        best_validation_metric = None
        best_y, best_y_hat, best_image_ids = [], [], []
        for iteration in range(3):
            for _ in range(3):
                print('-' * 80)
            test_centers_group_str = '.'.join(map(str, test_centers_group))
            print(f'CV with {test_centers_group_str} as test')
            train_sample_ids = [
                i
                for i, (_, _, center_id) in enumerate(train_data)
                if center_id not in test_centers_group
            ]
            test_sample_ids = [
                i
                for i, (_, _, center_id) in enumerate(train_data)
                if center_id in test_centers_group
            ]

            train_image_ids, train_labels, train_center_ids, train_crops, train_crop_indices = _get_sub_data(
                train_data,
                image_crops,
                image_crops_indices,
                train_sample_ids
            )
            test_image_ids, test_labels, test_center_ids, test_crops, test_crop_indices = _get_sub_data(
                train_data,
                image_crops,
                image_crops_indices,
                test_sample_ids
            )
            print(f'Train/Test sizes: {len(train_labels)}/{len(test_labels)}')
            print('Train/Test label distribution:')
            print({key: value / len(train_labels) for key, value in dict(Counter(train_labels)).items()})
            print({key: value / len(test_labels) for key, value in dict(Counter(test_labels)).items()})

            dataloaders = {
                'train': get_loader(
                    train_image_ids,
                    train_labels,
                    train_crops,
                    seed=42,
                    is_test=False,
                    transformations=train_transforms,
                    shuffle=True,
                    batch_size=64,
                    num_workers=2,
                ),
                'test': get_loader(
                    test_image_ids,
                    test_labels,
                    test_crops,
                    seed=42,
                    is_test=True,
                    transformations=test_transforms,
                    shuffle=False,
                    batch_size=64,
                    num_workers=2,
                ),
            }

            model = ClotModelSingle(encoder_model='effnet_b0').to(device)
            model.freeze_encoder(True)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.head.parameters(), lr=5e-3, weight_decay=5e-4)

            train_loss, val_loss = [], []
            for epoch in range(config.EPOCHS_NUM):
                np.random.seed(epoch)

                print('*' * 80)
                print("epoch {}/{}".format(epoch + 1, config.EPOCHS_NUM))

                model.train()
                running_loss, running_score = 0.0, 0.0
                y_hat, y, image_ids = [], [], []
                for image, label, image_id in tqdm(dataloaders['train']):
                    image = image.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    y_pred = model.forward(image).squeeze()
                    loss = criterion(y_pred, label)
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    y_pred = y_pred.cpu().detach().numpy().tolist()
                    label = label.cpu().detach().numpy().tolist()
                    y_hat.extend(y_pred)
                    y.extend(label)
                    image_ids.extend(image_id)

                    running_score += sum([int(int(y_hat > 0.5) == y) for y_hat, y in zip(y_pred, label)])

                print('Train:')
                print(Counter([int(p > 0.5) for p in y_hat]))
                print('ROC AUC metric:', roc_auc_score(y, y_hat))
                print('target metric:', get_target_metric(y, y_hat, image_ids))

                epoch_score = running_score / len(dataloaders['train'].dataset)
                epoch_loss = running_loss / len(dataloaders['train'])
                train_loss.append(epoch_loss)
                print("loss: {}, accuracy: {}".format(epoch_loss, epoch_score))

                with torch.no_grad():
                    model.eval()
                    running_loss, running_score = 0.0, 0.0
                    y_hat, y, image_ids = [], [], []
                    for image, label, image_id in tqdm(dataloaders['test']):
                        image = image.to(device)
                        label = label.to(device)
                        optimizer.zero_grad()
                        y_pred = model.forward(image).squeeze()
                        loss = criterion(y_pred, label)
                        running_loss += loss.item()

                        y_pred = y_pred.cpu().detach().numpy().tolist()
                        label = label.cpu().detach().numpy().tolist()
                        y_hat.extend(y_pred)
                        y.extend(label)
                        image_ids.extend(image_id)

                        running_score += sum([int(int(y_hat > 0.5) == y) for y_hat, y in zip(y_pred, label)])

                    bad_image_ids = {
                        image_id
                        for image_id, crops in zip(test_image_ids, test_crops)
                        if len(crops) == 0
                    }
                    y_hat_fixed = [
                        0.5 if image_id in bad_image_ids else p
                        for p, image_id in zip(y_hat, image_ids)
                    ]

                    print('Validation:')
                    print(Counter([int(p > 0.5) for p in y_hat_fixed]))
                    print('ROC AUC metric:', roc_auc_score(y, y_hat_fixed))

                    target_metric = get_target_metric(y, y_hat, image_ids)
                    print('target metric:', target_metric)
                    target_metric = get_target_metric(y, y_hat_fixed, image_ids)
                    print('target metric fixed:', target_metric)
                    if best_validation_metric is None or target_metric < best_validation_metric:
                        best_validation_metric = target_metric
                        best_y, best_y_hat, best_image_ids = y, y_hat_fixed, image_ids
                        torch.save(
                            model,
                            os.path.join(
                                '/kaggle/working/models',
                                f'center_id_{test_centers_group_str}_epoch_{epoch}_target_{round(target_metric, 3)}.h5',
                            ),
                        )

                    epoch_score = running_score / len(dataloaders['test'].dataset)
                    epoch_loss = running_loss / len(dataloaders['test'])
                    val_loss.append(epoch_loss)
                    print("loss: {}, accuracy: {}".format(epoch_loss, epoch_score))

        print(f'Best validation metric: {best_validation_metric}')
        all_metrics.append(best_validation_metric)
        all_y.extend(best_y)
        all_y_hat.extend(best_y_hat)
        all_image_ids.extend(best_image_ids)

    print(all_metrics)
    print(np.mean(all_metrics))
    final_metric = get_target_metric(
        all_y,
        all_y_hat,
        all_image_ids,
    )
    np.save('/kaggle/working/all_y.npy', all_y)
    np.save('/kaggle/working/all_y_hat.npy', all_y_hat)
    np.save('/kaggle/working/all_image_ids.npy', all_image_ids)
    print('Full validation metric:', final_metric)

    remove_bad_model_files()


if __name__ == "__main__":
    main()
