import torch
from torch.utils.data import DataLoader
from model import GalaxyDataset
from model import SimpleModel
import matplotlib.pyplot as plt


dataset = GalaxyDataset("train.npz")
loader = DataLoader(dataset, batch_size = 64, shuffle = True)

model = SimpleModel(input_dim = 70 * 3)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = torch.nn.MSELoss()

losses = []
for epoch in range(200):

    total_loss = 0

    for x, y in loader:

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_loss += loss.item()

    losses.append(total_loss)
    print(epoch,total_loss)

plt.plot(range(1, 21), losses, marker='o')
plt.xlabel("epoch")
plt.ylabel("Totall loss")
plt.grid(True)
plt.show()