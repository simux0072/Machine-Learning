from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Resources.plotcm import plot_confusion_matrix

import RunHelper

torch.set_printoptions(linewidth = 120)
torch.set_grad_enabled(True)

# Downloads the Fashion MNIST datasets (for training and for testing)

train_set = torchvision.datasets.FashionMNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()])
)

test_set = torchvision.datasets.FashionMNIST(
    root = './data',
    train = False,
    download = True,
    transform = transforms.Compose([transforms.ToTensor()])
)
#Returns the amount of correct model predictions
def get_num_correct(preds, labels):
    return preds.argmax(dim = 1).eq(labels).sum().item()

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([]).cuda()
    for batch in loader:
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim = 0)
    return all_preds
# CNN
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)

        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)


    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size = 2, stride = 2)

        t = F.relu(self.fc1(t.reshape(-1, 12 * 4 * 4)))
        t = F.relu(self.fc2(t))

        t = self.out(t)
        return t
# Network training function
def train():
    # Hyper rarameters for the network to test
    parameters = OrderedDict(
        lr = [0.0001], # 0.00005, 0.00001
        batch_size = [200, 250, 400, 750, 1000, 1500, 1875, 2000],
        epoch_num = [375],
        shuffled = [True],
        num_workers = [4]
    )

    m = RunHelper.RunManager()
    runs = RunHelper.RunBuilder.get_runs(parameters)

    nr_run = 1
    element_product = len(runs)
    keys = list(parameters)
    # Training with specified parameters combination
    for run in runs:
        network = Network().to('cuda')
        loader = torch.utils.data.DataLoader(train_set, batch_size = run.batch_size, shuffle = run.shuffled, num_workers = run.num_workers)
        optimizer = optim.Adam(network.parameters(), lr = run.lr)

        m.begin_run(run, network)

        max = [0, 0, 100000]
        info = ''

        for epoch in range(run.epoch_num):

            train_total_loss = 0
            train_total_correct = 0

            m.begin_epoch()

            for batch in loader:
                images = batch[0].to('cuda')
                labels = batch[1].to('cuda')

                preds = network(images)
                loss = F.cross_entropy(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_total_loss += loss.item()
                train_total_correct += get_num_correct(preds, labels)

            acc = train_total_correct/len(train_set)
            m.train_track_loss(train_total_loss)
            m.train_track_acc(acc)

            if max[0] < acc:
                torch.save(network.state_dict(), 'model.pth')

                max[0] = acc
                max[1] = epoch
                max[2] = train_total_loss 
            m.end_epoch()
        test_loss, test_acc = test(run, m, test_set)
        # Printing results to console
        for i in range(0, len(run)):
            info += ' ' + keys[i] + ' = ' + str(run[i])
        print('Run', nr_run, 'completed with settings:' + info)
        print('Training: Max Accuracy: ' + str(max[0] * 100) +'% Lowest Loss:', max[2], 'epoch:', max[1] + 1)
        print('Test: Accuracy: ' + str(test_acc * 100) + '% Loss:', test_loss)
        print(str(nr_run/element_product * 100) + '% completed')
        print('')
        nr_run += 1

        m.end_run()
    m.save('results')
        # For displaying the confusion matrix!!
        # prediction_loader = torch.utils.data.DataLoader(train_set, batch_size = 10000)
        # train_preds = get_all_preds(network, prediction_loader)

        # stacked = torch.stack((train_set.targets.cuda(), train_preds.argmax(dim = 1)), dim = 1)
        # cmt = torch.zeros(10, 10, dtype = torch.int32)

        # for p in stacked:
        #     tl, pl = p.tolist()
        #     cmt[tl, pl] = cmt[tl, pl] + 1

        # names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
        # plt.figure(figsize = (10, 10))
        # plot_confusion_matrix(cmt, names)
        # plt.show()
# Testing model
def test(run, m, test_set):

    test_total_loss = 0
    test_total_correct = 0

    network = Network().to('cuda')
    network.load_state_dict(torch.load('model.pth'))
    network.eval()
    loader = torch.utils.data.DataLoader(test_set, batch_size = 1000, shuffle = run.shuffled, num_workers = run.num_workers)
    for batch in loader:
        images = batch[0].to('cuda')
        labels = batch[1].to('cuda')

        preds = network(images)
        loss = F.cross_entropy(preds, labels)

        test_total_loss += loss.item()
        test_total_correct += get_num_correct(preds, labels)

    test_acc = test_total_correct/len(test_set)
    m.test_track_loss(test_total_loss)
    m.test_track_acc(test_acc)
    m.write_test()

    return test_total_loss, test_total_correct / len(test_set)

if __name__ == '__main__':
    train()