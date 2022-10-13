# 2nd place solution from [Mayo Clinic - STRIP AI](https://www.kaggle.com/competitions/mayo-clinic-strip-ai/) Kaggle competition

More tech info in [discussion](https://www.kaggle.com/competitions/mayo-clinic-strip-ai/discussion/358089) on Kaggle

This code were running on kaggle code notebook system on python 3.7.
GPU was used just to speed-up training, but this code works fine and fast even on CPU.
Standard kaggle python libraries were used plus pyvips to work with large images.

**How to run it?**
1. Install python libraries:

``pip install -r requirements.txt``
2. Run tiles generation.

``python src/preprocessing.py SETTINGS.json``

Gets input files from _settings.INPUT_DATA_PATH_ directory.
Saves train pickled tiles to _settings.DUMPED_DATALOADER_PATH_ 
and other pickled tiles to _settings.DUMPED_DATALOADER_OTHER_PATH_.
Please note that these files might be overwritten.
3. Run model train.

``python src/train.py SETTINGS.json``

Gets pickled files from the previous step using the same paths.
Stores final models to _settings.MODELS_PATH_ folder.
Please note that folder _settings.MODELS_PATH_ must be clear before the process starts.

4. Run final evaluation.

``python src/evaluate.py SETTINGS.json``

Gets model files from  _settings.MODELS_PATH_ and input data from _settings.INPUT_DATA_PATH_.
Saves final prediction to _settings.SUBMISSION_PATH_.

For all commands you can pass a path to your settings file as a parameter. 

**How to evaluate it with your data?**

Just reproduce the test format: 
put your WSI files into test/ folder and store metadata into test.csv.
After that do step 4.
