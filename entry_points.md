1. ``python src/preprocessing.py``

Gets input files from _settings.INPUT_DATA_PATH_ directory.

Runs tiles generation.

Saves train pickled tiles to _settings.DUMPED_DATALOADER_PATH_ 
and other pickled tiles to _settings.DUMPED_DATALOADER_OTHER_PATH_.
2. ``python src/train.py``

Gets pickled files from the previous step using the same paths.

Runs model train.

Stores final models to _settings.MODELS_PATH_ folder.

3. ``python src/evaluate.py``

Gets model files from  _settings.MODELS_PATH_ and input data from _settings.INPUT_DATA_PATH_.

Runs final evaluation.

Saves final prediction to _settings.SUBMISSION_PATH_.
