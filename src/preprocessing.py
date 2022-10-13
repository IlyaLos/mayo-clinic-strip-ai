import json
import pickle
import sys

from data_preparation import DataPreparation


def main() -> None:
    settings_file_path = 'SETTINGS.json'
    if len(sys.argv) > 1:
        settings_file_path = sys.argv[1]
    with open(settings_file_path, 'r') as file:
        settings = json.load(file)

    data_prep = DataPreparation(settings['INPUT_DATA_PATH'])

    image_crops, image_crops_indices = data_prep.process_train()
    with open(settings['DUMPED_DATALOADER_PATH'], 'wb') as file:
        pickle.dump([image_crops, image_crops_indices], file)

    image_crops_other, image_crops_indices_other = data_prep.process_other()
    with open(settings['DUMPED_DATALOADER_OTHER_PATH'], 'wb') as file:
        pickle.dump([image_crops_other, image_crops_indices_other], file)


if __name__ == "__main__":
    main()
