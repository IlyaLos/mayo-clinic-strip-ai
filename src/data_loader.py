import random
from collections import defaultdict
from typing import List

import numpy as np
import torch
from PIL import Image

import config


class ClotImageDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_ids: List[str],
            labels: List[str],
            image_crops: List[List[np.ndarray]],
            seed: int,
            is_test: bool,
            transformations,
    ):
        self.image_ids = image_ids
        self.labels = [float(label == 'CE') for label in labels]
        self.image_crops = image_crops
        self.seed = seed
        self.is_test = is_test
        self.transformations = transformations

        if not self.is_test:
            np.random.seed(self.seed)

            label_to_indices = defaultdict(list)
            for i, (label, crops) in enumerate(zip(self.labels, self.image_crops)):
                if len(crops) > 0:
                    label_to_indices[label].append(i)

            max_size = config.TRAIN_SAMPLE_DUPL_RATE * max(len(indices) for indices in label_to_indices.values())

            self.sample_ids = []
            for i, indices in enumerate(label_to_indices.values()):
                np.random.shuffle(indices)
                while len(self.sample_ids) < max_size * (i + 1):
                    req_size = min(len(indices), max_size * (i + 1) - len(self.sample_ids))
                    self.sample_ids += indices[:req_size]
        else:
            self.sample_ids = []
            for _ in range(config.TEST_SAMPLE_DUPL_RATE):
                self.sample_ids.extend(list(range(len(self.image_ids))))

        self.image_index_ids = []
        sample_id_to_image_index = defaultdict(int)
        for sample_id in self.sample_ids:
            self.image_index_ids.append(sample_id_to_image_index[sample_id])
            image_crops_cnt = len(self.image_crops[sample_id])
            if image_crops_cnt:
                sample_id_to_image_index[sample_id] = (sample_id_to_image_index[sample_id] + 1) % image_crops_cnt

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        if self.is_test:
            np.random.seed(self.seed + idx)
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)
        idx, image_index = self.sample_ids[idx], self.image_index_ids[idx]
        if len(self.image_crops[idx]) == 0:
            return (
                self.transformations(Image.fromarray(np.zeros((224, 224, 3)).astype(np.uint8))),
                torch.tensor(self.labels[idx]),
                self.image_ids[idx],
            )
        return (
            self.transformations(self.image_crops[idx][image_index]),
            torch.tensor(self.labels[idx]),
            self.image_ids[idx],
        )


def get_loader(
        image_ids: List[str],
        labels: List[str],
        image_crops: List[List[np.ndarray]],
        seed: int,
        is_test: bool,
        transformations,
        shuffle: bool,
        batch_size: int,
        num_workers: int
):
    dataset = ClotImageDataset(
        image_ids, labels, image_crops, seed, is_test, transformations,
    )
    return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
