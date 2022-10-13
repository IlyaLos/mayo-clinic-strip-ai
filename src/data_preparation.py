import gc
import os
from time import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyvips
from PIL import Image
from tqdm import tqdm

import config


class DataPreparation:
    def __init__(self, input_data_path: str, visualize: bool = False, seed: int = 42):
        self.input_data_path = input_data_path
        self.visualize = visualize
        self.seed = seed

        train_metadata = pd.read_csv(os.path.join(self.input_data_path, 'train.csv'))
        train_metadata = list(zip(
            train_metadata['image_id'].tolist(),
            train_metadata['label'].tolist(),
            train_metadata['center_id'].tolist(),
        ))
        self.train = self._filter_bad_images(train_metadata)
        self.all_center_ids = sorted(list({center_id for _, _, center_id in self.train}))

        other_metadata = pd.read_csv(os.path.join(self.input_data_path, 'other.csv')).query('label == \'Other\'')
        other_metadata = list(zip(
            other_metadata['image_id'].tolist(),
            ['LAA' for _ in range(other_metadata.shape[0])],
            [-1 for _ in range(other_metadata.shape[0])],
        ))
        self.other = self._filter_bad_images(other_metadata)

        test_metadata = pd.read_csv(os.path.join(self.input_data_path, 'test.csv'))
        self.test = list(zip(
            test_metadata['image_id'].tolist(),
            ['Unknown' for _ in range(test_metadata.shape[0])],
            test_metadata['center_id'].tolist(),
        ))

    @staticmethod
    def _filter_bad_images(data: List[Tuple]) -> List[Tuple]:
        return [
            (image_id, label, center_id)
            for image_id, label, center_id in data
            if image_id not in config.BAD_IMAGE_IDS
        ]

    @staticmethod
    def _add_rect_to_numpy(image: np.ndarray, x: int, y: int, size: int, thickness: int) -> None:
        image[x:x + size, y:y + thickness] = (0, 0, 0)
        image[x:x + thickness, y:y + size] = (0, 0, 0)
        image[x:x + size, y + size:y + size + thickness] = (0, 0, 0)
        image[x + size:x + size + thickness, y:y + size] = (0, 0, 0)

    @staticmethod
    def _get_blocks_map(image: np.ndarray) -> np.ndarray:
        pixels_diff = np.sum((image[:-1, :, :] - image[1:, :, :]) ** 2, axis=2)
        pixels_diff = np.cumsum(np.cumsum(pixels_diff, axis=0), axis=1)
        blocks_map = np.zeros((
            (image.shape[0] + config.BLOCK_SIZE - 1) // config.BLOCK_SIZE,
            (image.shape[1] + config.BLOCK_SIZE - 1) // config.BLOCK_SIZE,
        ))
        for x in range(0, pixels_diff.shape[0], config.BLOCK_SIZE):
            for y in range(0, pixels_diff.shape[1], config.BLOCK_SIZE):
                nx = min(x + config.BLOCK_SIZE, pixels_diff.shape[0])
                ny = min(y + config.BLOCK_SIZE, pixels_diff.shape[1])
                block_sum = int(pixels_diff[nx - 1, ny - 1])
                if x:
                    block_sum -= int(pixels_diff[x - 1, ny - 1])
                if y:
                    block_sum -= int(pixels_diff[nx - 1, y - 1])
                if x and y:
                    block_sum += int(pixels_diff[x - 1, y - 1])
                blocks_map[x // config.BLOCK_SIZE][y // config.BLOCK_SIZE] = \
                    (block_sum / config.BLOCK_SIZE / config.BLOCK_SIZE) > config.BLOCK_THR
        return blocks_map

    def _generate_crops_positions(
            self,
            image: np.ndarray,
            crop_thr: float,
    ) -> List[Tuple[int, int]]:
        blocks_map = self._get_blocks_map(image)

        if self.visualize:
            for i in range(blocks_map.shape[0]):
                for j in range(blocks_map.shape[1]):
                    if blocks_map[i][j]:
                        self._add_rect_to_numpy(
                            image,
                            i * config.BLOCK_SIZE,
                            j * config.BLOCK_SIZE,
                            config.BLOCK_SIZE,
                            1,
                        )

        good_crops_starts = []
        for x in range(0, image.shape[0] - config.CROP_SIZE + 1, config.BLOCK_SIZE):
            for y in range(0, image.shape[1] - config.CROP_SIZE + 1, config.BLOCK_SIZE):
                _x, _y = x // config.BLOCK_SIZE, y // config.BLOCK_SIZE
                crop_sum = blocks_map[_x:_x + config.BLOCKS_PER_CROP, _y:_y + config.BLOCKS_PER_CROP].sum()
                if crop_sum > config.BLOCKS_PER_CROP * config.BLOCKS_PER_CROP * crop_thr:
                    good_crops_starts.append((x, y))

        if self.visualize:
            for x, y in good_crops_starts:
                self._add_rect_to_numpy(image, x, y, config.CROP_SIZE, 1)

        return good_crops_starts

    @staticmethod
    def _process_crop(crop: np.ndarray) -> np.ndarray:
        return crop

    def _create_crops(
            self,
            image: np.ndarray,
            crops_starts: List[Tuple[int]],
    ) -> List[np.ndarray]:
        return [
            Image.fromarray(
                self._process_crop(
                    image[x:x + config.CROP_SIZE, y:y + config.CROP_SIZE],
                )
            )
            for x, y in crops_starts
        ]

    @staticmethod
    def _get_unique_crops(crop_starts: List[Tuple[int, int]], order) -> List[Tuple[int, int]]:
        def inter_size_1d(a: int, b: int, c: int, d: int) -> int:
            return max(0, min(b, d) - max(a, c))

        def inter_size_2d(crop_start_1: Tuple[int, int], crop_start_2: Tuple[int, int]) -> int:
            return inter_size_1d(
                crop_start_1[0], crop_start_1[0] + config.CROP_SIZE,
                crop_start_2[0], crop_start_2[0] + config.CROP_SIZE,
            ) * inter_size_1d(
                crop_start_1[1], crop_start_1[1] + config.CROP_SIZE,
                crop_start_2[1], crop_start_2[1] + config.CROP_SIZE,
            )

        crop_starts_sorted = sorted(crop_starts, key=order)
        final_crop_starts = []
        for crop_start in crop_starts_sorted:
            if any(
                    inter_size_2d(crop_start, crop_start_prev) > config.CROP_SIZE * config.CROP_SIZE // 2
                    for crop_start_prev in final_crop_starts
            ):
                continue
            final_crop_starts.append(crop_start)
        return final_crop_starts

    @staticmethod
    def _read_and_resize_image(image_id: str, base_image_path: str) -> np.ndarray:
        image_path = os.path.join(base_image_path, f'{image_id}.tif')
        image = pyvips.Image.new_from_file(image_path, access='sequential')
        return image.resize(1.0 / config.SCALE_FACTOR).numpy()

    def prepare_crops(
            self,
            image_ids: List[int],
            base_image_path: str,
    ) -> Tuple[List[List[np.ndarray]], List[List[Tuple[int]]]]:
        np.random.seed(self.seed)
        image_crops = []
        image_crops_indices = []
        for image_id in tqdm(image_ids):
            start_time = time()
            image = self._read_and_resize_image(image_id, base_image_path)
            gc.collect()
            print(f'Rescaling done in {time() - start_time} seconds. Image shape is {image.shape}')
            found_flag = False
            for crop_thr in np.arange(config.CROP_THR, -0.1, -0.1):
                good_crops_starts = self._generate_crops_positions(image, crop_thr)
                if len(good_crops_starts) < config.IMAGES_PER_SAMPLE:
                    print('Bad image', image_id, 'crop_thr', crop_thr, 'only', len(good_crops_starts))
                    continue

                good_crops_starts_unique = []
                for order in [
                    lambda x: (x[0], x[1]),
                    lambda x: (-x[0], -x[1]),
                ]:
                    good_crops_starts_unique.extend(self._get_unique_crops(good_crops_starts, order))
                good_crops_starts_unique = list(set(good_crops_starts_unique))

                if len(good_crops_starts_unique) < config.IMAGES_PER_SAMPLE:
                    print('Bad image', image_id, 'crop_thr', crop_thr, 'only', len(good_crops_starts_unique))
                    continue

                good_crops_starts_sample_ids = np.random.choice(
                    list(range(len(good_crops_starts_unique))),
                    min(len(good_crops_starts_unique), config.MAX_CROPS_PER_IMAGE),
                    replace=False,
                )
                good_crops_starts_sample = np.array(good_crops_starts_unique)[good_crops_starts_sample_ids]
                image_crops_indices.append(good_crops_starts_sample)
                image_crops.append(self._create_crops(image, good_crops_starts_sample))
                found_flag = True
                break
            if not found_flag:
                image_crops_indices.append([])
                image_crops.append([])
                print('No crops was found')
            print(f'Done {image_id} in {time() - start_time} seconds')
        gc.collect()
        return image_crops, image_crops_indices

    def process_train(
            self
    ) -> Tuple[List[List[np.ndarray]], List[List[Tuple[int]]]]:
        return self.prepare_crops(
            [image_id for image_id, _, _ in self.train],
            os.path.join(self.input_data_path, 'train/'),
        )

    def process_other(
            self
    ) -> Tuple[List[List[np.ndarray]], List[List[Tuple[int]]]]:
        return self.prepare_crops(
            [image_id for image_id, _, _ in self.other],
            os.path.join(self.input_data_path, 'other/'),
        )

    def process_test(
            self
    ) -> Tuple[List[List[np.ndarray]], List[List[Tuple[int]]]]:
        return self.prepare_crops(
            [image_id for image_id, _, _ in self.test],
            os.path.join(self.input_data_path, 'test/'),
        )
