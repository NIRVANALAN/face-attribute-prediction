from __future__ import print_function, absolute_import
import torch

__all__ = ["accuracy", "accuracy_bce"]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
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


def accuracy_bce(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size, label_number = target.size()

        pred = (output >= 0.5).view(1, -1)
        correct = pred.eq(target.view(1, -1))

        correct_k = correct.view(-1).float().sum(0)
        top1 = correct_k.mul_(100.0 / batch_size / label_number)

        return top1
