
import torch
from typing import List, Tuple
from torch.utils.data import Dataset

# My TextDataset from prev homework
class TextDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42
    VAL_RATIO = 0.05

    def __init__(self, data):
        self.x = data['de'].values
        self.target = data['en'].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Take corresponding index array from self.indices,
        add special tokens (self.bos_id and self.eos_id) and 
        pad to self.max_length using self.pad_id.
        Return padded indices of size (max_length, ) and its actual length
        """
        return self.x[item], self.target[item]

