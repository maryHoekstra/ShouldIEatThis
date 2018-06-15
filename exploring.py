import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
import pdb
import torch
import torch.utils.data
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn

class NeuralNet(nn.Module):
  # 12 input features
    def __init__(self, all_ing):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(all_ing, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256,1)

        # matrices are filled with random numbers

    def forward(self, x):
        # see the sketch above.
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)

        return out

def dict_top_ing(s):
# create dictionary of most popular ingredients

    slist = [y.strip() for x in s for y in x]

    top_items = sorted(slist, key=Counter(slist).get, reverse=True)

    top_items_unique = list(OrderedDict.fromkeys(top_items))

    return top_items_unique


def main():
    # read all the data and keep specific columns
    cols = ['product_name','ingredients_text','nutrition_grade_fr',
            'nutrition-score-fr_100g','sugars_100g', 'countries']

    all_data = pd.read_csv("en.openfoodfacts.org.products.tsv", sep = '\t',
                            low_memory=False, usecols = cols)


    # keep only english ingredients and ingredients with score
    eng_data = all_data[(all_data['countries'] == 'US') &
                            (all_data['ingredients_text'].notnull()) &
                            (all_data['nutrition-score-fr_100g'].notnull())].reset_index()


    # preprocessing on ingredients: removing parentheses, asterix

    eng_data.ingredients_text = eng_data.ingredients_text.str.lower()
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('ingredients', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace(' \(', ', ')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('\)', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace(' \[', ', ')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('\]', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace(': ', ', ')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('; ', ', ')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('\. ', ', ')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('\.', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('including ', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('contains ', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('organic ', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('org ', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('*', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace(r'[0-9]','')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('% ','')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('%','')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('&','')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('or less of','')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('less than of','')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('each of the following','')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('the following','')

    eng_data.ingredients_text = eng_data.ingredients_text.str.split(', ')
    top_ing = dict_top_ing(eng_data.ingredients_text)


    np.random.seed(1234)
    train, validate, test = np.split(eng_data.sample(frac=1, random_state=134),
                            [int(.6*len(eng_data)), int(.8*len(eng_data))])

    mlb = MultiLabelBinarizer()
    X_train = mlb.fit_transform(train['ingredients_text']).astype(np.float32)
    y_train = train['nutrition-score-fr_100g'].values

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                                   torch.from_numpy(y_train))

    all_ing = len(X_train[0])

    neural_net = NeuralNet(all_ing)

    pdb.set_trace()

if __name__ == '__main__':
    main()
