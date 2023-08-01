import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
from sympy import symbols, lambdify, sympify

class PrecomputedDataset(Dataset):
    def __init__(self, data_file):
        self.data = torch.load(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class CustomEquationDataset(Dataset):
    def __init__(self, equation_file, transform=None, target_transform=None):
        self.eq_labels = pd.read_csv(equation_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.eq_labels)

    def __getitem__(self, idx):
        try:
            equation = self.eq_labels.iloc[idx, 1]
            support = self.eq_labels.iloc[idx, 2]
            support = ast.literal_eval(support)
            length = len(support)

            x_1, x_2, x_3 = symbols('x_1 x_2 x_3')
            sympified_eq = sympify(str(equation))
            eq_lambda = lambdify((x_1, x_2, x_3), sympified_eq, modules=['numpy'])

            params = np.zeros((4, 100))#

            a = np.random.uniform(-10, 10, (length, 100)).astype(float)
            params[:length] = a
            last_row = eq_lambda(params[0], params[1], params[2])
            params[3] = np.array(last_row)#

            while not np.all(np.isreal(params[3])):
                a = np.random.uniform(-25, 25, (length, 100))#
                params[:length] = a
                last_row = eq_lambda(params[0], params[1], params[2])
                params[:, ~np.isreal(params[3])] = np.array(last_row)[:, ~np.isreal(params[3])]

            last_row = eq_lambda(params[0], params[1], params[2])
            params[3] = np.array(last_row)#
            
            bases = self.eq_labels.iloc[idx, 4]
            # convert bases into a list of integers
            bases = [int(i) for i in bases[1:-1].split(',')]

            if self.transform:
                params = self.transform(params)

            if self.target_transform:
                bases = self.target_transform(bases)
            
            return torch.tensor(params, dtype=torch.float32), torch.tensor(bases, dtype=torch.float32).long()
        
        except:
            return False, False

if __name__ == '__main__':
    a = CustomEquationDataset('data/nesymres/val_nc.csv')
    c = 0

    data_list = []

    for i, (x, y) in enumerate(a):
        # check if the type of the number is real 
        # not nan, no inf, no -inf
        if (x is False) or torch.isnan(x).any() or torch.isinf(x).any() or torch.isneginf(x).any() or torch.isposinf(x).any():
            continue
        else:
            c += 1
            data_list.append((x, y))
        
        print(i, c)

        if i % 1000 == 0:
            torch.save(data_list, 'data/nesymres/val_nc.pt')
            print('{c} valid data'.format(c=c))
            print('saved data')
        if i > 30000:
            break


    