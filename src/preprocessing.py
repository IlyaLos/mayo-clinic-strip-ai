import pickle

from data_preparation import DataPreparation


DUMPED_DATALOADER_PATH = '/kaggle/working/data_loaders.pkl'
DUMPED_DATALOADER_OTHER_PATH = '/kaggle/working/data_loaders_other.pkl'


def main() -> None:
    data_prep = DataPreparation()

    image_crops, image_crops_indices = data_prep.process_train()
    with open(DUMPED_DATALOADER_PATH, 'wb') as file:
        pickle.dump([image_crops, image_crops_indices], file)

    image_crops_other, image_crops_indices_other = data_prep.process_other()
    with open(DUMPED_DATALOADER_OTHER_PATH, 'wb') as file:
        pickle.dump([image_crops_other, image_crops_indices_other], file)


if __name__ == "__main__":
    main()
