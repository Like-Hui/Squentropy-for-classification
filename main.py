import argparse
import os
import sys
import math
import numpy as np
import tools
import torch.nn as nn
import torch
from fnn import fully_nn
import random
import shutil
from torch.nn import functional as F
from sklearn.utils import shuffle
from torchsummary import summary
# import pdb
# pdb.set_trace()

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default="data", type=str, help="data directory")
parser.add_argument('--file', default="result.log", type=str, help="Output File")
parser.add_argument('--max_tot', default=5000, type=int, help="Maximum number of data samples")
parser.add_argument('--max_dep', default=5, type=int, help="Maximum number of depth")
parser.add_argument('--epochs', default=2000, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', default=0.01, type=float,
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--name', default='ce', type=str,
                    help='name of experiment')
parser.add_argument('--loss_type', default='CE', type=str,
                    help='name of experiment')
parser.add_argument('--weighted', default=0, type=int, help='reweight the loss at true label by ?')
parser.add_argument('--rescale_factor', default=1, type=int, help='rescale the one hot vector by how much?')
parser.add_argument('--rescale', default=1, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--seed', default=1, type=int,
                    help='mini-batch size (default: 128)')

args = parser.parse_args()
print(args)


class _ECELoss(nn.Module):

    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                print('bin_lower=%f, bin_upper=%f, accuracy=%.4f, confidence=%.4f: ' % (bin_lower, bin_upper, accuracy_in_bin.item(),
                      avg_confidence_in_bin.item()))
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        print('ece = ', ece)
        # print('accuracy = ', accuracies)
        return ece


def validate(X, y, model, criterion, epoch, c):
    """Perform validation on the validation set"""
    test_losses = AverageMeter()
    test_top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    target = y.cuda(non_blocking=True)
    input = X.type(dtype=torch.float).cuda(non_blocking=True)
    # compute output
    with torch.no_grad():
        output = model(input)
    loss_test = criterion(output, target)

    # measure accuracy and record loss
    prec1 = accuracy(output.data, target, topk=(1,))[0]
    test_losses.update(loss_test.data.item(), input.size(0))
    test_top1.update(prec1.item(), input.size(0))
    return test_top1.avg, output


def train(X, y, model, optimizer, epoch, num_classes):
    """Train for one epoch on the training set"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    target = y.cuda(non_blocking=True)
    input = X.type(dtype=torch.float).cuda(non_blocking=True)
    # compute output
    output = model(input)

    if args.loss_type == 'MSE':
        device = target.get_device()
        target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(target.size()[0], 1), 1)
        loss = torch.mean((output - args.rescale_factor * target_final.type(torch.float)) ** 2)

    elif args.loss_type == 'mixed':
            # combine cross entropy and square loss
            ce_func = nn.CrossEntropyLoss().cuda()
            device = target.get_device()
            target_final = torch.zeros([target.size()[0], num_classes], device=device).scatter_(1, target.reshape(target.size()[0], 1), 1)
            loss = (torch.sum(output** 2)-torch.sum((output[target_final == 1]) ** 2))/(num_classes-1)/target_final.size()[0] + ce_func(output, target)
    else:
        target_final = target
        criterion = nn.CrossEntropyLoss().cuda()
        loss = criterion(output, target_final)

    # measure accuracy and record loss
    prec1 = accuracy(output.data, target, topk=(1,))[0]
    losses.update(loss.data.item(), input.size(0))
    top1.update(prec1.item(), input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #scheduler.step()

    print('Epoch: %d., loss = %.4f, acc = %.4f' %
          (epoch, losses.avg, top1.avg))

    # wandb.log({'train_loss': losses.avg})
    # wandb.log({'train_acc': top1.avg})


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

MAX_N_TOT = args.max_tot
MAX_DEP = args.max_dep
DEP_LIST = list(range(MAX_DEP))
C_LIST = [10.0 ** i for i in range(-2, 5)]
datadir = args.dir

alg = tools.svm
seed = args.seed
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

avg_acc_list = []
avg_ece_list = []
outf = open(args.file, "w")
print ("Dataset\tTest Acc\tece\tlr\t\thid dim\t\tepoch ", file = outf)
for idx, dataset in enumerate(sorted(os.listdir(datadir))):
    if not os.path.isdir(datadir + "/" + dataset):
        continue
    if not os.path.isfile(datadir + "/" + dataset + "/" + dataset + ".txt"):
        continue
    dic = dict()
    for k, v in map(lambda x: x.split(), open(datadir + "/" + dataset + "/" + dataset + ".txt", "r").readlines()):
        dic[k] = v
    c = int(dic["n_clases="])
    d = int(dic["n_entradas="])
    n_train = int(dic["n_patrons_entrena="])
    n_val = int(dic["n_patrons_valida="])
    n_train_val = int(dic["n_patrons1="])
    n_test = 0
    if "n_patrons2=" in dic:
        n_test = int(dic["n_patrons2="])
    n_tot = n_train_val + n_test

    print (idx, dataset, "\tN:", n_tot, "\td:", d, "\tc:", c)

    # load data
    f = open("data/" + dataset + "/" + dic["fich1="], "r").readlines()[1:]
    X = np.asarray(list(map(lambda x: list(map(float, x.split()[1:-1])), f)))
    y = np.asarray(list(map(lambda x: int(x.split()[-1]), f)))
    # breakpoint()

    criterion = nn.CrossEntropyLoss().cuda()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)

    print("Training")

    fold = list(map(lambda x: list(map(int, x.split())),
                    open(datadir + "/" + dataset + "/" + "conxuntos.dat", "r").readlines()))
    train_fold, val_fold = fold[0], fold[1]

    X_train, y_train = X[train_fold], y[train_fold],
    X_val, y_val = X[val_fold], y[val_fold]

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train)
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val)

    best_prec1 = 0
    # lrs = torch.logspace(start=-2.5, end=-0.8, steps=10)
    lrs = [0.005]
    hids = [256]
    actn = nn.ReLU()
    bn = True
    drop = 0
    batch_size = 64

    best_lr = 0
    best_hid = 0
    best_epoch = 0
    for lr in lrs:
        for hid in hids:
            hidden_dims = [hid, hid, hid]

            # enumerate kenerls and cost values to find the best hyperparameters
            model = fully_nn(d, hidden_dims, c, bn, drop, actn)
            model = model.cuda()

            optimizer = torch.optim.SGD(model.parameters(), lr,
                                        momentum=args.momentum, nesterov=args.nesterov,
                                        weight_decay=args.weight_decay)
            for epoch in range(args.start_epoch, args.epochs):
                train(X_train, y_train, model, optimizer, epoch, c)
                # evaluate on validation set
                prec1, output = validate(X_val, y_val, model, criterion, epoch, c)

                # remember best prec@1 and save checkpoint
                # is_best = prec1 >= best_prec1
                if prec1 > best_prec1:
                    best_lr = lr
                    best_hid = hid
                    best_epoch = epoch
                best_prec1 = max(prec1, best_prec1)
                # save_checkpoint({
                #     'epoch': epoch + 1,
                #     'lr': lr,
                #     'hid': hid,
                #     'best_pred': best_prec1,
                #     'state_dict': model.state_dict(),
                # }, is_best)

    print('dataset: %s, final valid accuracy = %.4f' % (dataset, best_prec1))

    hidden_dims_test = [best_hid, best_hid, best_hid]


    optimizer_test = torch.optim.SGD(model_test.parameters(), best_lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)
    fold = list(map(lambda x: list(map(int, x.split())),
                    open(datadir + "/" + dataset + "/" + "conxuntos_kfold.dat", "r").readlines()))

    acc_4 = 0
    ece_4 = 0
    ece_criterion = _ECELoss().cuda()
    print('selected epoch, lr = ', best_epoch, best_lr)
    for repeat in range(4):
        train_fold, test_fold = fold[repeat * 2], fold[repeat * 2 + 1]

        X_train_1 = X[train_fold]
        y_train_1 = y[train_fold]
        X_test = X[test_fold]
        y_test = y[test_fold]

        model_test = fully_nn(d, hidden_dims_test, c, bn, drop, actn)
        model_test = model_test.cuda()

        X_train_1 = torch.from_numpy(X_train_1).float()
        y_train_1 = torch.from_numpy(y_train_1)
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test)
        for epoch in range(0, best_epoch):
            # X_tr = X_train.unsqueeze(1)
            # X_tr = X_tr.unsqueeze(0)
            # breakpoint()
            permutation = torch.randperm(X_train_1.size()[0])
            # for i in range(0, X_train_1.size()[0], batch_size):
            #     indices = permutation[i:i + batch_size]
            #     batch_x, batch_y = X_train_1[indices], y_train_1[indices]
            #
            #     train(batch_x, batch_y, model_test, optimizer_test, epoch, c)
            train(X_train_1, y_train_1, model_test, optimizer_test, epoch, c)

        prec1_b, output_b = validate(X_test, y_test, model_test, criterion, epoch, c)
        ece = ece_criterion(output_b, y_test.cuda()).item()
        acc_4 += prec1_b * 0.25
        ece_4 += ece * 0.25
    print('dataset: %s, Test accuracy = %.4f, ece = %.4f' % (dataset, acc_4, ece_4))
    print (str(dataset) + '\t' + str(acc_4) + '\t' + str(ece_4) + '\t' + str(best_lr) + '\t' + str(best_hid) + '\t' + str(best_epoch), file = outf)
    avg_acc_list.append(acc_4)
    avg_ece_list.append(ece_4)

print ("avg_acc:", np.mean(avg_acc_list))
print ("avg_ece:", np.mean(avg_ece_list))

outf.close()
