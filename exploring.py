import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
import pdb

'''
import torch
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
import torch.nn.functional as F
'''
import pickle

'''
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

'''

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

    all_data = pd.read_csv("en.openfoodfacts.org.products.tsv", sep='\t',
                           low_memory=False, usecols=cols, skipinitialspace=True)


    # keep only english ingredients and ingredients with score
    eng_data_all = all_data[(all_data['countries'] == 'US') &
                            (all_data['ingredients_text'].notnull()) &
                            (all_data['nutrition-score-fr_100g'].notnull())].reset_index()

    eng_data = eng_data_all.head(10000)

    # preprocessing on ingredients: removing parentheses, asterix
    eng_data.ingredients_text = eng_data.ingredients_text.str.lower()

    remove_words = '|'.join(['ingredients', 'ingredient', 'including ',
                             'contains ', 'organic ', 'org ', 'or less of',
                             'less than of', 'each of the following',
                             'as a ', 'contain ', 'one or more of',
                             'to preserve freshness'])

    eng_data.ingredients_text = eng_data.ingredients_text.str.replace(remove_words, '')

    eng_data.ingredients_text = eng_data.ingredients_text.str.replace(' \(', ', ')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('\(', ', ')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('\)', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace(' \[', ', ')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('\]', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace(': ', ', ')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('; ', ', ')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('\. ', ', ')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('\.', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('*', '')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace(r'[0-9]','')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('% ','')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('%','')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('&','')
    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('#','')

    eng_data.ingredients_text = eng_data.ingredients_text.str.replace('  ', ' ')

    eng_data.ingredients_text = eng_data.ingredients_text.str.split(', ')

    top_ing = dict_top_ing(eng_data.ingredients_text)[0:10000]

    pdb.set_trace()

    for ind, list_ing in enumerate(eng_data.ingredients_text):
        list_ing = [x for x in list_ing if x not in top_ing]
        eng_data.ingredients_text[ind] = list_ing

        if ind%1000 == 0:
            print(ind)

    print(eng_data.ingredients_text)
    f = open('store.pckl', 'wb')
    pickle.dump(eng_data, f)
    f.close()

    '''
    np.random.seed(1234)
    train, validate, test = np.split(eng_data.sample(frac=1, random_state=134),
                            [int(.6*len(eng_data)), int(.8*len(eng_data))])

    mlb = MultiLabelBinarizer()
    X_train = mlb.fit_transform(train['ingredients_text']).astype(np.float32)
    y_train = train['nutrition-score-fr_100g'].values

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                                   torch.from_numpy(y_train))

    all_ing = len(X_train[0])

    # neural network
    model = NeuralNet(all_ing)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=4)

    # start training
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)


    pdb.set_trace()

    for epoch in range(1, 2):

        for batch_idx, (data, target) in enumerate(train_loader):

        #    data, target = Variable(data), Variable(target)

            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))



    pdb.set_trace()
    '''

if __name__ == '__main__':
    main()
