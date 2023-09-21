import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("GOOG.csv")
closed_prices = df["close"]
seq_len = 15

mm = MinMaxScaler()
Scaled_price = mm.fit_transform(np.array(closed_prices)[..., None]).squeeze()

x = []  # data
y = []  # target (empty list)

for i in range(len(Scaled_price) - seq_len):
    x.append(Scaled_price[i:i + seq_len])
    y.append(Scaled_price[i + seq_len])

x = np.array(x)[..., None]
y = np.array(y)[..., None]

train_x = torch.from_numpy(x[:int(0.8 * x.shape[0])]).float()
train_y = torch.from_numpy(y[:int(0.8 * x.shape[0])]).float()
test_x = torch.from_numpy(x[int(0.8 * x.shape[0]):]).float()
test_y = torch.from_numpy(y[int(0.8 * x.shape[0]):]).float()

class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return self.fc(output[:, -1, :])

model = Model(1, 96)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

num_epochs = 100
for epoch in range(num_epochs):
    output = model(train_x)
    loss = loss_fn(output, train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 and epoch != 0:
        print(epoch, "epoch loss", loss.detach().numpy())

model.eval()
with torch.no_grad():
    output = model(test_x)

pred = mm.inverse_transform(output.numpy())
real = mm.inverse_transform(test_y.numpy())

plt.plot(pred.squeeze(), color="red", label="predicted")
plt.plot(real.squeeze())
plt.legend()
plt.show()
