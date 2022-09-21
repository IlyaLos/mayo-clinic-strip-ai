import gc
import os
from time import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import skimage.io as io
import cv2
import config


class DataPreparation:
    def __init__(self, visualize: bool = False, seed: int = 42):
        self.visualize = visualize
        self.seed = seed

        train_metadata = pd.read_csv('data/train.csv')
        train_metadata = list(zip(
            train_metadata['image_id'].tolist(),
            train_metadata['label'].tolist(),
            train_metadata['center_id'].tolist(),
        ))
        self.train = self._filter_bad_images(train_metadata)
        self.all_center_ids = sorted(list({center_id for _, _, center_id in self.train}))

        other_metadata = pd.read_csv('data/other.csv').query('label == \'Other\'')
        other_metadata = list(zip(
            other_metadata['image_id'].tolist(),
            ['LAA' for _ in range(other_metadata.shape[0])],
            [-1 for _ in range(other_metadata.shape[0])],
        ))
        self.other = self._filter_bad_images(other_metadata)

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

    @staticmethod
    def _get_mean_color_by_block(image: np.ndarray) -> np.ndarray:
        colors_cum_sum = np.cumsum(np.cumsum(image, axis=0), axis=1)
        mean_color_by_block = np.zeros((
            (image.shape[0] + config.BLOCK_SIZE - 1) // config.BLOCK_SIZE,
            (image.shape[1] + config.BLOCK_SIZE - 1) // config.BLOCK_SIZE,
            3,
        ))
        for x in range(0, image.shape[0], config.BLOCK_SIZE):
            for y in range(0, image.shape[1], config.BLOCK_SIZE):
                nx = min(x + config.BLOCK_SIZE, image.shape[0])
                ny = min(y + config.BLOCK_SIZE, image.shape[1])
                block_sum = colors_cum_sum[nx - 1, ny - 1, :].flatten()
                if x:
                    block_sum -= colors_cum_sum[x - 1, ny - 1, :].flatten()
                if y:
                    block_sum -= colors_cum_sum[nx - 1, y - 1, :].flatten()
                if x and y:
                    block_sum += colors_cum_sum[x - 1, y - 1, :].flatten()
                mean_color_by_block[x // config.BLOCK_SIZE, y // config.BLOCK_SIZE, :] = \
                    block_sum / config.BLOCK_SIZE / config.BLOCK_SIZE
        return mean_color_by_block

    def _get_color_std(
            self,
            image: np.ndarray,
            color_mean: np.ndarray,
            blocks_map: np.ndarray,
            value: int,
    ) -> np.ndarray:
        color_std_sum_by_block = self._get_mean_color_by_block((image - color_mean) ** 2)
        color_std_sum = np.sum(color_std_sum_by_block[blocks_map == value], axis=0)
        blocks_cnt = len(np.where(blocks_map == value)[0])
        return np.sqrt(color_std_sum / blocks_cnt)

    def _generate_crops_positions(
            self,
            image: np.ndarray,
            crop_thr: float,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        blocks_map = self._get_blocks_map(image)
        mean_color_by_block = self._get_mean_color_by_block(image)
        blood_color_mean = np.mean(mean_color_by_block[blocks_map == 1], axis=0)
        background_color_mean = np.mean(mean_color_by_block[blocks_map == 0], axis=0)

        # blood_color_std = self._get_color_std(image, blood_color_mean, blocks_map, 1)
        # background_color_std = self._get_color_std(image, background_color_mean, blocks_map, 0)

        min_color, max_color = np.zeros(3), np.zeros(3)
        # min_color = np.maximum(blood_color_mean - 2 * blood_color_std, 0).astype(int)
        # max_color = np.minimum(blood_color_mean + 2 * blood_color_std, background_color_mean).astype(int)

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

        return good_crops_starts, min_color, max_color, background_color_mean, blood_color_mean

    @staticmethod
    def _process_crop(crop: np.ndarray, min_color: np.ndarray, max_color: np.ndarray) -> np.ndarray:
        return crop
        #normalized_rgb_crop = np.clip((crop - min_color) / (max_color - min_color) * 255, 0, 255)
        #rgb_to_grayscale_coefficients = [0.2989, 0.5870, 0.1140]
        #return np.clip(np.sum(normalized_rgb_crop * rgb_to_grayscale_coefficients, axis=2), 0, 255)

    def _create_crops(
        self,
        image: np.ndarray,
        crops_starts: List[Tuple[int]],
        min_color: np.ndarray,
        max_color: np.ndarray,
    ) -> List[np.ndarray]:
        return [
            Image.fromarray(
                self._process_crop(
                    image[x:x + config.CROP_SIZE, y:y + config.CROP_SIZE], min_color, max_color,
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

    def prepare_crops(
            self,
            image_ids: List[int],
            base_image_path: str,
    ) -> Tuple[List[List[np.ndarray]], List[List[Tuple[int]]], List[Tuple[np.ndarray, np.ndarray]]]:
        np.random.seed(self.seed)
        image_crops = []
        image_crops_indices = []
        colors = []
        for image_id in tqdm(image_ids):
            start_time = time()
            image_path = os.path.join(base_image_path, f'{image_id}.tif')
            image = io.imread(image_path)
            image = cv2.resize(image, dsize=None, fx=1/config.SCALE_FACTOR, fy=1/config.SCALE_FACTOR,
                               interpolation=cv2.INTER_AREA)
            print(f'Rescaling done in {time() - start_time} seconds. Image shape is {image.shape}')
            found_flag = False
            for crop_thr in np.arange(config.CROP_THR, 0, -0.1):
                good_crops_starts, min_color, max_color, background_color_mean, blood_color_mean = \
                    self._generate_crops_positions(image, crop_thr)
                if len(good_crops_starts) < config.IMAGES_PER_SAMPLE:
                    print('Bad image', image_id, 'crop_thr', crop_thr, 'only', len(good_crops_starts))
                    continue

                good_crops_starts_unique = []
                for order in [
                    lambda x: (x[0], x[1]),
                    lambda x: (-x[0], x[1]),
                    lambda x: (x[0], -x[1]),
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
                image_crops.append(self._create_crops(image, good_crops_starts_sample, min_color, max_color))
                colors.append((background_color_mean, blood_color_mean))
                found_flag = True
                break
            if not found_flag:
                image_crops_indices.append([])
                image_crops.append([])
                colors.append([])
                print('No crops was found')
            print(f'Done {image_id} in {time() - start_time} seconds')
        gc.collect()
        return image_crops, image_crops_indices, colors

    def process_train(
            self
    ) -> Tuple[List[List[np.ndarray]], List[List[Tuple[int]]], List[Tuple[np.ndarray, np.ndarray]]]:
        return self.prepare_crops([image_id for image_id, _, _ in self.train], 'data/train/')

    def process_other(
            self
    ) -> Tuple[List[List[np.ndarray]], List[List[Tuple[int]]], List[Tuple[np.ndarray, np.ndarray]]]:
        return self.prepare_crops(
            [image_id for image_id, _, _ in self.other],
            '/Volumes/TOSHIBA EXT/mayo-clinic-strip-ai/data/other/',
        )
