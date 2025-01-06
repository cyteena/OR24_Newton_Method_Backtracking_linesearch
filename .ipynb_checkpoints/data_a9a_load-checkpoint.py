import numpy as np
from scipy.sparse import lil_matrix, vstack
from tqdm import tqdm

file_name = 'a9a.txt'
is_sparse = 1
feat = 123
batch_size = 32

filepath = './' + file_name

def parse_line(line, change_label = False):
    parts = line.strip().split()
    label = int(parts[0])
    if change_label:
        # 将标签从 -1 转换为 0
        label = 0 if label == -1 else 1
    indices = []
    values = []
    for item in parts[1:]:
        index, value = item.split(':')
        indices.append(int(index))
        values.append(float(value))
    return label, indices, values

def load_data(filepath, feat, is_sparse=True, batch_size=1000, change_label = False):
    if is_sparse:
        Xtrain = lil_matrix((0, feat))
    else:
        Xtrain = np.zeros((0, feat))
    
    Ylabel = []
    batch_X = []
    batch_Y = []

    with open(filepath, 'r') as fid:
        for line in tqdm(fid, desc="Reading data"):
            label, indices, values = parse_line(line, change_label)
            batch_Y.append(label)
            
            if is_sparse:
                new_row = lil_matrix((1, feat))
                for idx, val in zip(indices, values):
                    if 0 <= idx < feat:
                        new_row[0, idx] = val
                batch_X.append(new_row)
            else:
                new_row = np.zeros((1, feat))
                for idx, val in zip(indices, values):
                    if 0 <= idx < feat:
                        new_row[0, idx] = val
                batch_X.append(new_row)
            
            if len(batch_X) >= batch_size:
                if is_sparse:
                    Xtrain = vstack([Xtrain] + batch_X)
                else:
                    Xtrain = np.vstack([Xtrain] + batch_X)
                Ylabel.extend(batch_Y)
                batch_X = []
                batch_Y = []

    if batch_X:
        if is_sparse:
            Xtrain = vstack([Xtrain] + batch_X)
        else:
            Xtrain = np.vstack([Xtrain] + batch_X)
        Ylabel.extend(batch_Y)

    return Xtrain, Ylabel
