import torch
from torch.utils.data import DataLoader
from model import GalaxyDataset
from model import SimpleModel



dataset = GalaxyDataset("train.npz")
loader = DataLoader(dataset, batch_size = 32, shuffle = True)

model = SimpleModel(input_dim = 20 * 3)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(20):

    total_loss = 0

    for x, y in loader:

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(epoch,total_loss)