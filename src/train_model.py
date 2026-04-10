from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from model import GalaxyDatasetDeepSets
from model import DeepSetModel
import matplotlib.pyplot as plt
import torch

print(torch.cuda.is_available())


dataset = GalaxyDatasetDeepSets("train.npz")
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset,[train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size = 64, shuffle = False)

model = DeepSetModel()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = torch.nn.MSELoss()

train_losses = []
test_losses = []
for epoch in range(200):


    model.train()

    train_loss = 0
    for x, y in train_loader:
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)


    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            test_loss += loss.item()

    test_loss /= len(test_loader)



    train_losses.append(train_loss)
    test_losses.append(test_loss)


    print(epoch, train_loss, test_loss)




plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.show()
