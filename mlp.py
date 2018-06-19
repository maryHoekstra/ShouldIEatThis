import numpy as np
import pdb
import torch
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.autograd import Variable
import sklearn.linear_model as lm


class NeuralNet(nn.Module):
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


def main():

    f = open('store.pckl', 'rb')
    eng_data = pickle.load(f)
    f.close()


    eng_data = eng_data[eng_data.ingredients_text.apply(len) > 0].reset_index()

    np.random.seed(1234)
    train, validate, test = np.split(eng_data.sample(frac=1, random_state=134),
                            [int(.6*len(eng_data)), int(.8*len(eng_data))])

    mlb = MultiLabelBinarizer()
    X_train = mlb.fit_transform(train['ingredients_text']).astype(np.float32)
    y_train = train['nutrition-score-fr_100g'].values

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                   torch.from_numpy(y_train).float())



    all_ing = len(X_train[0])

    # neural network
    model = NeuralNet(all_ing)

    print(model)
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True,
                              num_workers=4)

    # start training
    learning_rate = 0.001
#    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, 20):

        total_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output[:,0], target)

            loss.backward()
            optimizer.step()

            '''
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
            '''
            # accumulate the loss of each minibatch
            total_loss += loss.data[0]*len(data)

            # compute the accuracy per minibatch
            pred_classes = output.data.max(1, keepdim=True)[1]
            correct += pred_classes.eq(target.data.view_as(pred_classes).long()).sum().double()

        # compute the mean loss for each epoch
        mean_loss = total_loss/len(train_loader.dataset)

        # compute the accuracy for each epoch
        acc = correct / len(train_loader.dataset)

        print('Train Epoch: {}   Avg_Loss: {:.5f}   Acc: {}/{} ({:.3f}%)'.format(
            epoch, mean_loss, correct, len(train_loader.dataset),
            100. * acc))

    pdb.set_trace()
    validate.ingredients_text = validate.ingredients_text.str.replace('strawberry candy', '')

    X_val = mlb.transform(validate['ingredients_text']).astype(np.float32)
    y_test = test['nutrition-score-fr_100g'].values
    regr = lm.LinearRegression()  # 1
    regr.fit(X_train, y_train)
    pdb.set_trace()



if __name__ == '__main__':
    main()
