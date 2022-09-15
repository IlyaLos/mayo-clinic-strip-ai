import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

import config


class ClotImageDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_ids: List[str],
            base_image_path: str,
            labels: List[str],
            center_ids: List[int],
            image_crops: List[List[np.ndarray]],
            image_crops_indices: List[List[Tuple[int]]],
            seed: int,
            is_test: bool,
            transformations,
    ):
        self.image_ids = image_ids
        self.base_image_path = base_image_path
        self.labels = [float(label == 'CE') for label in labels]
        self.center_ids = center_ids
        self.image_crops = image_crops
        self.image_crops_indices = image_crops_indices
        self.seed = seed
        self.is_test = is_test
        self.transformations = transformations

        if not self.is_test:
            np.random.seed(self.seed)

            label_to_indices = defaultdict(list)
            for i, (label, crops) in enumerate(zip(self.labels, self.image_crops)):
                if len(crops) > 0:
                    label_to_indices[label].append(i)

            max_size = 4 * max(len(indices) for indices in label_to_indices.values())

            self.sample_ids = []
            for i, indices in enumerate(label_to_indices.values()):
                np.random.shuffle(indices)
                while len(self.sample_ids) < max_size * (i + 1):
                    req_size = min(len(indices), max_size * (i + 1) - len(self.sample_ids))
                    self.sample_ids += indices[:req_size]
        else:
            self.sample_ids = []
            for _ in range(8):
                self.sample_ids.extend(list(range(len(self.image_ids))))

    @staticmethod
    def _generate_random_bag(crop_indices: List[Tuple[int]], bag_size: int) -> List[int]:
        def get_dist_between_crops(crop_1: Tuple[int], crop_2: Tuple[int]) -> float:
            return abs(crop_1[0] - crop_2[0]) + abs(crop_1[1] - crop_2[1])

        def get_dist_to_all_crops(crop: Tuple[int], bag_crop_ids: List[int]) -> float:
            return min([get_dist_between_crops(crop, crop_indices[bag_crop_id]) for bag_crop_id in bag_crop_ids])

        bag_crop_ids = [np.random.randint(0, len(crop_indices))]
        for it in range(bag_size - 1):
            other_crops_weighted = [
                (i, get_dist_to_all_crops(crop, bag_crop_ids))
                for i, crop in enumerate(crop_indices)
                if i not in bag_crop_ids
            ]
            total_weights_sum = sum([pair[1] for pair in other_crops_weighted])
            random_value = np.random.random() * total_weights_sum
            new_bag_crop_id = None
            for pair in other_crops_weighted:
                if pair[1] > random_value:
                    new_bag_crop_id = pair[0]
                    break
                random_value -= pair[1]
            assert new_bag_crop_id is not None
            bag_crop_ids.append(new_bag_crop_id)

        return bag_crop_ids

    def __len__(self):
        return len(self.sample_ids)

    def get_image_ids(self):
        return [self.image_ids[i] for i in self.sample_ids]

    def getitem_mil(self, idx):
        if self.is_test:
            np.random.seed(self.seed + idx)
        idx = self.sample_ids[idx]
        image_indices = self._generate_random_bag(self.image_crops_indices[idx], config.IMAGES_PER_SAMPLE)
        images = [self.image_crops[idx][image_index] for image_index in image_indices]
        return torch.stack([self.transformations(image) for image in images]), torch.tensor(self.labels[idx])

    def __getitem__(self, idx):
        if self.is_test:
            np.random.seed(self.seed + idx)
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)
        idx = self.sample_ids[idx]
        if len(self.image_crops[idx]) == 0:
            return self.transformations(Image.fromarray(np.zeros((224, 224, 1)))), torch.tensor(0.0)
        image_index = np.random.randint(0, len(self.image_crops[idx]))
        return self.transformations(self.image_crops[idx][image_index]), torch.tensor(self.labels[idx])


def get_loader(
        image_ids: List[str],
        base_image_path: str,
        labels: List[str],
        center_ids: List[int],
        image_crops: List[List[np.ndarray]],
        image_crops_indices: List[List[Tuple[int]]],
        seed: int,
        is_test: bool,
        transformations,
        shuffle: bool,
        batch_size: int,
        num_workers: int
):
    dataset = ClotImageDataset(
        image_ids, base_image_path, labels, center_ids, image_crops, image_crops_indices, seed, is_test, transformations,
    )
    return torch.utils.data.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
