import torch
from torch.utils.data import DataLoader
from model import GalaxyDatasetDeepSets
from model import DeepSetModel
import matplotlib.pyplot as plt
import torch

print(torch.cuda.is_available())


dataset = GalaxyDatasetDeepSets("train.npz")
loader = DataLoader(dataset, batch_size = 64, shuffle = True)

model = DeepSetModel()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = torch.nn.MSELoss()

losses = []
for epoch in range(20):

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