'''Train CIFAR10 with PyTorch.'''
"""adapted from https://github.com/kuangliu/pytorch-cifar"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
# import models as _models

import os
import math
import json
import argparse

import logger
from utils import progress_bar
from warmup_scheduler import GradualWarmupScheduler

# reproducibility
torch.manual_seed(233)
torch.cuda.manual_seed_all(233)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--bs', default=128, type=int, help='training batch size')
parser.add_argument('--epoch', default=200, type=int, help='epoch')
parser.add_argument('--lr_decay', default=0.1, type=float, help='lr decay gamma for StepLR')
parser.add_argument('--warmup_epochs', default=1, type=int, help='warmup epochs')
parser.add_argument('--opt', default="sgd", type=str, help='optimizer')
# parser.add_argument('--psuedo', action="store_true", help='whether mimic distributed setting regarding batchnorm')
parser.add_argument('--psuedo', default=0, help='psuedo batches', type=int)

parser.add_argument('--log_dir', default="./logs", type=str, help='log dir')
parser.add_argument('--extra', default="", type=str, help='extra log name')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if not os.path.isdir(args.log_dir):
    os.makedirs(args.log_dir)

log_name = f"{args.opt}_bs-{args.bs}_lr-{args.lr}_lrDecay-{args.lr_decay}_warmupEpochs-{args.warmup_epochs}_epoch-{args.epoch}"
if args.psuedo:
    log_name += f"_psuedo-{args.psuedo}"
if args.extra:
    log_name += "_" + args.extra
logger.set_logger_dir(os.path.join(args.log_dir, log_name))
logger.info(f"log name: {log_name}")

# Data
logger.info('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), std = [0.2470322549343109, 0.24348513782024384, 0.26158788800239563]),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), std = [0.2470322549343109, 0.24348513782024384, 0.26158788800239563]),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.bs, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
logger.info('==> Building resnet18 model..')
net = models.resnet18(num_classes=len(classes))

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# optimizer
criterion = nn.CrossEntropyLoss()
if args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=5e-4)
elif args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, 
                    weight_decay=5e-4)
elif args.opt == "rmsprop":
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, 
                    weight_decay=5e-4)
elif args.opt == "adagrad":
    optimizer = optim.Adagrad(net.parameters(), lr=args.lr, 
                    weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=5e-4)
    logger.warn(f"illegal optmizer: {args.opt}, fall back to SGD")

# scheduler          
after_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=args.lr_decay)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=after_scheduler)

# summary
info_dict = {
    "train_loss": [],
    "test_loss": [],
    "train_acc": [],
    "test_acc": [],
    "lr": [],
}
writer = SummaryWriter(logger.get_logger_dir())

# Training
def raw_train(epoch):
    logger.info('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # summary
    per_sample_loss = train_loss / (len(trainloader))
    acc = correct / total
    info_dict["train_loss"].append(per_sample_loss)
    info_dict["train_acc"].append(acc)
    writer.add_scalar('train_loss', per_sample_loss, epoch)
    writer.add_scalar('train_acc', acc, epoch)

    logger.info(f"[Epoch {epoch}] train_loss(per sample): {per_sample_loss:0.4f}, train_acc: {acc:0.4f}, lr: {scheduler.get_last_lr()}")

# mimic distributed setting, where each worker's mini batch is args.psuedo
# key difference is the update of batch norm running mean and var
def psuedo_train(epoch):
    logger.info('\nPsuedo Epoch: %d, bs： %d' % (epoch, args.psuedo))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        loss = 0
        optimizer.zero_grad()
        outputs = []
        psuedo_phases = int(math.ceil(len(inputs) / args.psuedo))
        for psuedo_batch_idx in range(psuedo_phases):

            psuedo_inputs = inputs[psuedo_batch_idx*args.psuedo:psuedo_batch_idx*args.psuedo+args.psuedo]
            psuedo_targets = targets[psuedo_batch_idx*args.psuedo:psuedo_batch_idx*args.psuedo+args.psuedo]
            # if psuedo_batch_idx == psuedo_phases - 1: print(psuedo_inputs.shape)

            psuedo_outputs = net(psuedo_inputs)
            loss += criterion(psuedo_outputs, psuedo_targets)
            outputs.append(psuedo_outputs)
            # print(psuedo_inputs.size(), psuedo_targets.size(), psuedo_outputs.size())

        loss /= psuedo_phases
        loss.backward()
        optimizer.step()
        outputs = torch.cat(outputs)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # summary
    per_sample_loss = train_loss / (len(trainloader))
    acc = correct / total
    info_dict["train_loss"].append(per_sample_loss)
    info_dict["train_acc"].append(acc)
    writer.add_scalar('train_loss', per_sample_loss, epoch)
    writer.add_scalar('train_acc', acc, epoch)

    logger.info(f"[Epoch {epoch}] train_loss(per sample): {per_sample_loss:0.4f}, train_acc: {acc:0.4f}, lr: {scheduler.get_last_lr()}")

train = psuedo_train if args.psuedo else raw_train

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    per_sample_loss = test_loss / (len(testloader))
    acc = correct / total
    info_dict["test_loss"].append(per_sample_loss)
    info_dict["test_acc"].append(acc)
    writer.add_scalar('test_loss', per_sample_loss, epoch)
    writer.add_scalar('test_acc', acc, epoch)

    logger.info(f"[Epoch {epoch}] test_loss(per sample): {per_sample_loss:0.4f}, test_acc: {acc:0.4f}")


    # Save checkpoint.
    acc = correct/total
    if acc > best_acc:
        logger.info('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(logger.get_logger_dir(), "best_ckpt.pth"))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epoch):
    # required by gradual warm scheduler
    scheduler.step()
    train(epoch)
    test(epoch)
    info_dict["lr"].append(scheduler.get_last_lr()[0])
    writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
    # overwrite at each epoch
    with open(os.path.join(logger.get_logger_dir(), "info_dict"), "w") as f:
        json.dump(info_dict, f)
