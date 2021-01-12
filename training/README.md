# How to use this folder

## Step 1: clean up the project folder
Delete any files and folders that are not needed at runtime.
edit start_fresh.py to include any additional folders
Then run `start-fresh` in vscode

## Step 2: generate cache for dataset
Copy all neccessary data to `training/data`. \
Edit `training/data/gen_cache.py` to setup dataset converters \
Finally, if you are to train on current machine, run `gen-cache` to generate cach & DPD training \
After this step, there much be two files in `training/cache`:
- `training/cache/traing_data.json`
- `training/cache/validation_data.json`

## Step 3: run training
run `train-from-scratch` in vscode if you are training for the first time \
run `train-from-serialization` in vscode if you want to start from previous state

## Notes
- There can only be two files that contain tunable parameters for an experiment.
    1. `training/gen_cache.py` -> `cache_params`
    2. `training/temli_train.py` -> `train_params`



