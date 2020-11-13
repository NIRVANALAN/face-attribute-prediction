from . import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, accuracy_bce
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as tf
import torch.backends.cudnn as cudnn
from tqdm import tqdm


def stat(loader):

    # switch to evaluate mode
    mean = torch.zeros(3)
    std = torch.zeros(3)

    samples = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(loader)):
            images = input.view(input.size(0), input.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            samples += input.size(0)
            # measure data loading time
    std /= samples
    mean /= samples

    print('mean: {}'.format(mean))
    print('std: {}'.format(std))

    return mean, std
