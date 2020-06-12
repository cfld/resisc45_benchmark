
import os
from glob import glob
from tqdm import tqdm
from shutil import copy
from sklearn.model_selection import train_test_split

outdir = 'data/NWPU-RESISC45_split'

assert not os.path.exists(f'{outdir}/train')
assert not os.path.exists(f'{outdir}/valid')

os.makedirs(f'{outdir}/train')
os.makedirs(f'{outdir}/valid')

# train/ test split
fnames = glob('data/NWPU-RESISC45/*/*')
fnames_train, fnames_valid = train_test_split(fnames, train_size=0.2, random_state=123)

# copy to train
for src in tqdm(fnames_train):
    dst = src.replace('NWPU-RESISC45', 'NWPU-RESISC45_split/train')
    _ = os.makedirs(os.path.dirname(dst), exist_ok=True)
    _ = copy(src, dst)

# copy to valid
for src in tqdm(fnames_valid):
    dst = src.replace('NWPU-RESISC45', 'NWPU-RESISC45_split/valid')
    _ = os.makedirs(os.path.dirname(dst), exist_ok=True)
    _ = copy(src, dst)
