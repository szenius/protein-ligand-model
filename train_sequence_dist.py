from train_utils_dist import load_batch
from keras.utils import Sequence
from tqdm import tqdm
import numpy as np

# Here, `x_list` is list of path to the protein ligand pairs
# and `y_list` are the associated classes.

class TrainSequence(Sequence):

    def __init__(self, x_list, y_list, batch_size):
        self.x, self.y = x_list, y_list
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_x = self.x[start:end] # batch of protein, ligand filepath pairs
        batch_y = self.y[start:end]
        x = load_batch(batch_x)
        print("Check {}".format(x.shape))
        return x, batch_y