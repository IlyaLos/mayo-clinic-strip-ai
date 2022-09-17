from __future__ import print_function, division

import os
import pickle
import sys
from collections import Counter

import cv2
import numpy as np
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchvision import transforms

import config
from data_loader import get_loader
from data_preparation import DataPreparation
from model import ClotModelSingle, ClotModelMIL
from metrics import get_target_metric


ssl._create_default_https_context = ssl._create_unverified_context
cudnn.benchmark = True

DUMPED_DATALOADER_PATH = 'data/data_loaders.pkl'


def get_sub_data(data, image_crops, image_crops_indices, sample_ids):
    return [data[i][0] for i in sample_ids], \
        [data[i][1] for i in sample_ids], \
        [data[i][2] for i in sample_ids], \
        [image_crops[i] for i in sample_ids], \
        [image_crops_indices[i] for i in sample_ids]


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def main() -> None:
    data_prep = DataPreparation()
    if (len(sys.argv) > 1 and sys.argv[1] == 'recalc') or not os.path.exists(DUMPED_DATALOADER_PATH):
        image_crops, image_crops_indices, colors = data_prep.process_train()
        with open(DUMPED_DATALOADER_PATH, 'wb') as file:
            pickle.dump([image_crops, image_crops_indices, colors], file)
    else:
        with open(DUMPED_DATALOADER_PATH, 'rb') as file:
            image_crops, image_crops_indices, colors = pickle.load(file)

    # data_prep.train = [data_prep.train[i] for i in config.INTERESTING_IDS]
    # data_prep.all_center_ids = sorted(list({center_id for _, _, center_id in data_prep.train}))
    # image_crops = [image_crops[i] for i in config.INTERESTING_IDS]
    # image_crops_indices = [image_crops_indices[i] for i in config.INTERESTING_IDS]

    # image_crops = [[
    #     Image.fromarray(image_histogram_equalization(np.array(crop).astype(np.uint8)))
    #     for crop in crops
    # ] for crops in tqdm(image_crops)]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.ColorJitter(brightness=0.2, saturation=0.5, hue=0.5),
            # transforms.RandomAutocontrast(p=0.5),
            # transforms.RandomEqualize(p=0.5),
            #transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=1.0),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.ColorJitter(brightness=0.2, saturation=0.5, hue=0.5),
            # transforms.RandomAutocontrast(p=0.5),
            # transforms.RandomEqualize(p=0.5),
            #transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_metrics = []
    all_y, all_y_hat, all_image_ids = [], [], []
    for test_center_id in data_prep.all_center_ids:
        for _ in range(3):
            print('-' * 80)
        print(f'CV with {test_center_id} as test')
        train_sample_ids = [i for i, (_, _, center_id) in enumerate(data_prep.train) if center_id != test_center_id]
        test_sample_ids = [i for i, (_, _, center_id) in enumerate(data_prep.train) if center_id == test_center_id]

        train_image_ids, train_labels, train_center_ids, train_crops, train_crop_indices = get_sub_data(
            data_prep.train,
            image_crops,
            image_crops_indices,
            train_sample_ids
        )
        test_image_ids, test_labels, test_center_ids, test_crops, test_crop_indices = get_sub_data(
            data_prep.train,
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
                'data/compressed_2/train/',
                train_labels,
                train_center_ids,
                train_crops,
                train_crop_indices,
                seed=42,
                is_test=False,
                transformations=data_transforms['train'],
                shuffle=True,
                batch_size=16,
                num_workers=2,
            ),
            'test': get_loader(
                test_image_ids,
                'data/compressed_2/train/',
                test_labels,
                test_center_ids,
                test_crops,
                test_crop_indices,
                seed=42,
                is_test=True,
                transformations=data_transforms['test'],
                shuffle=False,
                batch_size=16,
                num_workers=2,
            ),
        }

        model = ClotModelSingle(encoder_model='effnet_b0').to(device)
        model.freeze_encoder(True)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.head.parameters(), lr=1e-3, weight_decay=5e-4)

        best_validation_metric = None
        best_y, best_y_hat, best_image_ids = [], [], []
        train_loss, val_loss = [], []
        for epoch in range(config.EPOCHS_NUM):
            np.random.seed(epoch)

            #if epoch == 2:
                # model.freeze_encoder(False)
            #    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)

            print('*' * 80)
            print("epoch {}/{}".format(epoch + 1, config.EPOCHS_NUM))

            model.train()
            running_loss, running_score = 0.0, 0.0
            y_hat, y = [], []
            for image, label in tqdm(dataloaders['train']):
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

                running_score += sum([int(int(y_hat > 0.5) == y) for y_hat, y in zip(y_pred, label)])

            print('Train:')
            print(Counter([int(p > 0.5) for p in y_hat]))
            print('ROC AUC metric:', roc_auc_score(y, y_hat))
            target_metric = get_target_metric(
                y,
                y_hat,
                dataloaders['train'].dataset.get_image_ids(),
            )
            print('target metric:', target_metric)

            epoch_score = running_score / len(dataloaders['train'].dataset)
            epoch_loss = running_loss / len(dataloaders['train'].dataset)
            train_loss.append(epoch_loss)
            print("loss: {}, accuracy: {}".format(epoch_loss, epoch_score))

            with torch.no_grad():
                model.eval()
                running_loss, running_score = 0.0, 0.0
                y_hat, y = [], []
                for image, label in tqdm(dataloaders['test']):
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

                    running_score += sum([int(int(y_hat > 0.5) == y) for y_hat, y in zip(y_pred, label)])

                print('Validation:')
                print(Counter([int(p > 0.5) for p in y_hat]))
                print('ROC AUC metric:', roc_auc_score(y, y_hat))

                # original_val_length = len(dataloaders['test'].dataset.image_ids)
                # print('target metric:', get_target_metric(
                #     y[:original_val_length],
                #     y_hat[:original_val_length],
                #     dataloaders['test'].dataset.get_image_ids()[:original_val_length],
                # ))
                target_metric = get_target_metric(
                    y,
                    y_hat,
                    dataloaders['test'].dataset.get_image_ids(),
                )
                print('target metric:', target_metric)
                if best_validation_metric is None or target_metric < best_validation_metric:
                    best_validation_metric = target_metric
                    best_y, best_y_hat, best_image_ids = y, y_hat, dataloaders['test'].dataset.get_image_ids()
                    torch.save(
                        model,
                        os.path.join(
                            'models',
                            f'center_id_{test_center_id}_epoch_{epoch}_target_{round(target_metric, 3)}.h5',
                        ),
                    )

                epoch_score = running_score / len(dataloaders['test'].dataset)
                epoch_loss = running_loss / len(dataloaders['test'].dataset)
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
    print('Full validation metric:', final_metric)


if __name__ == "__main__":
    main()
