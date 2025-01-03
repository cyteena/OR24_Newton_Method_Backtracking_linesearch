import numpy as np
from scipy.sparse import lil_matrix, hstack
import scipy.io

file_name = 'a9a.txt'
is_sparse = 1
feat = 123

filepath = './' + file_name

if is_sparse:
    Xtrain = lil_matrix((feat, 0))
else:
    Xtrain = np.zeros((feat, 0))

Ylabel = []
data = [Xtrain, Ylabel]
Data = [data]

with open(filepath, 'r') as fid:
    for k in range(int(1e10)):
        str_line = fid.readline()
        if k % 100 == 0:
            print(k)
        
        if not str_line:
            break
        
        n = len(str_line)
        if n <= 3:
            if is_sparse:
                Xtrain = hstack([Xtrain, lil_matrix((feat, 1))])
            else:
                Xtrain = np.hstack([Xtrain, np.zeros((feat, 1))])
            
            if n < 2:
                Ylabel.append(int(str_line))
            else:
                Ylabel.append(int(str_line[:2]))
            continue
        
        J = []
        Value = []
        Ylabel.append(int(str_line[:2]))
        
        b1 = b2 = 0
        for j in range(n):
            id = 0
            if j == n - 1:
                Value.append(float(str_line[b2 + 1:n]))
                break
            if str_line[j] == ' ':
                b1 = j
                id = 1
            if str_line[j] == ':':
                b2 = j
                id = 2
            if id == 1 and b1 >= b2:
                Value.append(float(str_line[b2 + 1:b1]))
            if id == 2:
                J.append(int(str_line[b1 + 1:b2]))
        
        if is_sparse:
            Xtrain = lil_matrix(hstack([Xtrain, lil_matrix((feat, 1))]))
            for idx, val in zip(J, Value):
                Xtrain[idx, -1] = val
        else:
            new_col = np.zeros((feat, 1))
            for idx, val in zip(J, Value):
                new_col[idx, 0] = val
            Xtrain = np.hstack([Xtrain, new_col])
        
        if k % 1001 == 0:
            if is_sparse:
                data = [Xtrain.tocsr(), Ylabel]
            else:
                data = [Xtrain, Ylabel]
            Data.append(data)
            if is_sparse:
                Xtrain = lil_matrix((feat, 0))
            else:
                Xtrain = np.zeros((feat, 0))
            Ylabel = []

data = [Xtrain, Ylabel]
Data.append(data)

Xtrain = lil_matrix((feat, 0)) if is_sparse else np.zeros((feat, 0))
Ylabel = []
Data = Data[1:]

for d in Data:
    Xtrain = hstack([Xtrain, d[0]]) if is_sparse else np.hstack([Xtrain, d[0]])
    Ylabel.extend(d[1])

data = [Xtrain.tocsr() if is_sparse else Xtrain, Ylabel]
scipy.io.savemat(file_name, {'data': data})