import os

import pandas as pd
import torch

from src.dataset import Titanic
from src.model import Model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    table_data = pd.read_csv(os.path.expanduser('~/data/kaggle/titanic/test.csv'))
    tests_data = Titanic(table_data, with_labels=False)
    model = Model(8)
    model = torch.load('./tmp/weight/model_best888.pth')
    model.eval()
    res = {'PassengerId': [], 'Survived':[]}
    #with open('./tmp/res/888.csv', 'w') as f:
    for ex, pass_id in tests_data:
        predict = int(model(torch.tensor(ex).unsqueeze(0)) > 0.5)
        res['PassengerId'].append(int(pass_id))
        res['Survived'].append(predict)

    res = pd.DataFrame.from_dict(res)
    res.to_csv('./tmp/res/out.csv', index=False)



if __name__ == '__main__':
    main()