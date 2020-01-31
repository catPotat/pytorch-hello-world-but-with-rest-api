from model import Hewwo
import torch
from torchvision import transforms, datasets
import os

PATH = os.path.dirname(os.path.realpath(__file__))
net = Hewwo()
net.load_state_dict(torch.load(PATH+"\\hewwo.pth"))


test = datasets.MNIST('', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)


correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        X, y = data
        output = net(X.view(-1,784))
        #print(output)
        for idx, i in enumerate(output):
            #print(torch.argmax(i), y[idx])
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
        break

print("Accuracy: ", round(correct/total, 3))



import matplotlib.pyplot as plt

index = 9
print(torch.argmax(net(X[index].view(-1,784))[0]))
plt.imshow(X[index].view(28,28))
plt.show()