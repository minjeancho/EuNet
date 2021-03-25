import numpy as np

from torch.utils.data import Dataset


def get_dictionaries(seq):
    idx2note = { }
    note2idx = { }
    note_min = np.min(seq)
    note_max = np.max(seq)
    for idx in range(0, note_max - note_min + 1):
        note = note_min + idx 
        idx2note[idx] = note
        note2idx[note] = idx
    return note2idx, idx2note

def get_data(seq, note2idx, window_size=4):
    seq_length = seq.shape[0]
    x_length   = seq_length - window_size
    idx_max    = np.max(list(note2idx.values()))
    # === x_train (notes data) 
    x_train = np.zeros((x_length, window_size))
    y_note  = np.zeros((x_length))
    for t in range(x_length):
        window = seq[t:(t+window_size)]
        x_train[t] = [note2idx[w] for w in window]
        y_note[t] = note2idx[seq[t+window_size]]
    return x_train, y_note

class SongDataset(Dataset):
    def __init__(self, song):
        # === get note2idx and idx2note dictionaries
        self.note2idx, self.idx2note = get_dictionaries(song)
        self.x_train, self.y_train = get_data(song, self.note2idx)

    def __len__(self):
        return len(self.x_train)
    
    def __getitem__(self, idx):
        item = {
            'train_input': self.x_train[idx],
            'train_label': self.y_train[idx]
        }
        return item 
    
    


