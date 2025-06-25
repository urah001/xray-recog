import os
import shutil
import random

def split_data(source, train, val, split_size=0.8):
    files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
    random.shuffle(files)
    train_count = int(len(files) * split_size)
    
    train_files = files[:train_count]
    val_files = files[train_count:]

    for file in train_files:
        shutil.copy(os.path.join(source, file), os.path.join(train, file))
    for file in val_files:
        shutil.copy(os.path.join(source, file), os.path.join(val, file))

base_dir = 'dataset/all'  # original folder where all images are
train_dir = 'dataset/train/Normal'
val_dir = 'dataset/val/Normal'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

split_data(base_dir, train_dir, val_dir)
