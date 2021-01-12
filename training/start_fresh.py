import glob
import shutil

volatile_folders = []
# volatile_folders += glob.glob(".history*")
# volatile_folders += glob.glob(".idea*")
# volatile_folders += glob.glob(".ipynb_checkpoints*")
# volatile_folders += glob.glob(".git*")
volatile_folders += glob.glob("training/serialization*")
volatile_folders += glob.glob("training/cache")

for serialization_folder in volatile_folders:
    try:
        shutil.rmtree(serialization_folder)
    except:
        pass