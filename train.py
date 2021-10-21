import torch
import torch.utils.data as td
import torch.optim as optim
import torch.nn as nn


from lib.dataset.dataset import Dataset
from lib.model.u_net import UNet

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, optimizer, loss_fn):
    model.train()
    train_loss = AverageMeter()
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.cuda()
        labels = labels.cuda()
        prediction = model(images)
        loss = loss_fn(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), images.size(0))
    print("train loss",train_loss.avg)

def test(test_loader, model, loss_fn):
    model.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            
            images = images.cuda()
            labels = labels.cuda()
            prediction = model(images)
            loss = loss_fn(prediction, labels)
            val_loss.update(loss.item(),images.size(0))
        print("Val loss",val_loss.avg)

if __name__ == "__main__":

    params = {
        'batch_size': 2,
        'shuffle': True,
        'num_workers':4,
        'pin_memory': True
    }
    train_dataset = Dataset("data/membrane",set = 'train')
    train_loader = td.DataLoader(train_dataset, **params)
    test_dataset = Dataset("data/membrane",set = 'test')
    test_loader = td.DataLoader(train_dataset, **params)

    model = UNet(in_channels = 1, out_channels = 1)

    optimizer = optim.Adam(lr = 0.0001, params = model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()

    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(20):
        train(train_loader=train_loader, model= model, optimizer=optimizer, loss_fn = loss_fn)
        test(test_loader=test_loader, model= model, loss_fn = loss_fn)
    print("done")
