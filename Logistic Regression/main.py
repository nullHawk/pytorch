import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#0) Prepare data
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

n_samples, n_features = x.shape
print(n_samples, n_features)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234) # 80% training data, 20% testing data

# scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train) # fit and transform
x_test = sc.transform(x_test) # only transform

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

#1) model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

#2) loss and optimizer
learning_rate = 0.01
critoerion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#3) training loop
epochs = 100
for epoch in range(epochs):
    # forward pass and loss
    y_predicted = model(x_train)
    loss = critoerion(y_predicted, y_train)
    
    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round() # round off to nearest integer, 0 or 1
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
