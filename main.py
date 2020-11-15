"""
Training script for ImageNet
Copyright (c) Wei YANG, 2017
"""
from __future__ import print_function

import argparse
import math
import os
import shutil
import time
import random

from torchvision.transforms.transforms import (
    ColorJitter,
    RandomAffine,
    RandomGrayscale,
    RandomPerspective,
)
from utils.focal import FocalLoss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as tf
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from math import cos, pi
from models.resnet import fc_block

from celeba import CelebA, LFW
from utils import (
    Bar,
    Logger,
    AverageMeter,
    accuracy,
    mkdir_p,
    savefig,
    accuracy_bce,
    stat,
)
from tensorboardX import SummaryWriter


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


# Parse arguments
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("-d", "--data", default="path to dataset", type=str)
parser.add_argument("-dl", "--data_lfw", default="path to lfw dataset", type=str)
parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)
# Optimization options
parser.add_argument(
    "--epochs", default=30, type=int, metavar="N", help="number of total epochs to run"
)

parser.add_argument(
    "-rs", action="store_true", help="ReweightedRandomSampler",
)

parser.add_argument(
    "-lw", action="store_true", help="Reverse Sample Count CE Weight",
)

parser.add_argument(
    "-fc", "--focal", action="store_true", help="Reverse Sample Count CE Weight",
)

parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--train-batch",
    default=256,
    type=int,
    metavar="N",
    help="train batchsize (default: 256)",
)
parser.add_argument(
    "--test-batch",
    default=320,
    type=int,
    metavar="N",
    help="test batchsize (default: 320)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--lr-decay", type=str, default="cos", help="mode for learning rate decay"
)

parser.add_argument("--sampler", type=str, default="uniform", help="data sampler")

parser.add_argument(
    "--step", type=int, default=20, help="interval for learning rate decay in step mode"
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=[150, 225],
    help="decrease learning rate at these epochs.",
)
parser.add_argument(
    "--turning-point",
    type=int,
    default=100,
    help="epoch number from linear to exponential decay mode",
)
parser.add_argument(
    "--gamma", type=float, default=0.1, help="LR is multiplied by gamma on schedule."
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
# Checkpoints
parser.add_argument(
    "-c",
    "--checkpoint",
    default="checkpoints",
    type=str,
    metavar="PATH",
    help="path to save checkpoint (default: checkpoints)",
)

parser.add_argument(
    "--ft", action="store_true", help="fine tune on Balance Class Sampler",
)

parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
# Architecture
parser.add_argument(
    "--cardinality", type=int, default=32, help="ResNeXt model cardinality (group)."
)
parser.add_argument(
    "--base-width",
    type=int,
    default=4,
    help="ResNeXt model base width (number of channels in each group).",
)
parser.add_argument("--groups", type=int, default=3, help="ShuffleNet model groups")
# Miscs
parser.add_argument("--manual-seed", type=int, help="manual seed")
parser.add_argument(
    "-e", "--evaluate", action="store_true", help="evaluate model on test set",
)
parser.add_argument(
    "-v", "--validate", action="store_true", help="evaluate model on validation set",
)
parser.add_argument(
    "-el", "--evaluate_lfw", action="store_true", help="evaluate model on lfw set",
)
parser.add_argument(
    "-pt",
    "--pretrained",
    dest="pretrained",
    action="store_true",
    help="use pre-trained model",
)
parser.add_argument(
    "--world-size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="gloo", type=str, help="distributed backend"
)
# Device options
parser.add_argument(
    "--gpu-id", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)


best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # Use CUDA
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    # Random seed
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manual_seed)

    # create model
    if args.resume == "" and args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch.startswith("resnext"):
        model = models.__dict__[args.arch](
            baseWidth=args.base_width, cardinality=args.cardinality,
        )
    elif args.arch.startswith("shufflenet"):
        model = models.__dict__[args.arch](groups=args.groups)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=False)

    if args.ft:
        for param in model.parameters():
            param.requires_grad = False
        classifier_numbers = model.num_attributes
        # newly constructed classifiers have requires_grad=True
        for i in range(classifier_numbers):
            setattr(
                model,
                "classifier" + str(i).zfill(2),
                nn.Sequential(fc_block(512, 256), nn.Linear(256, 1)),
            )

        args.sampler = "balance"

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
        )

    if not args.distributed:
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # optionally resume from a checkpoint
    title = "CelebA-" + args.arch
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(  # statistics from CelebA TrainSet
        mean=[0.5084, 0.4287, 0.3879], std=[0.2656, 0.2451, 0.2419]
    )
    print("=> using {} sampler to load data.".format(args.sampler))

    train_dataset = CelebA(
        args.data,
        "train_attr_list.txt",
        transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=(256, 256), scale=(0.5, 1.0)),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(),
            ]
        ),
        sampler=args.sampler,
    )

    train_sample_prob = train_dataset._class_sample_prob()

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        if args.rs:
            train_sampler = WeightedRandomSampler(
                1 / train_sample_prob, len(train_dataset)
            )
        else:
            train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        CelebA(
            args.data,
            "val_attr_list.txt",
            transforms.Compose(
                [transforms.Resize(size=(256, 256)), transforms.ToTensor(), normalize,]
            ),
        ),
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        CelebA(
            args.data,
            "test_attr_list.txt",
            transforms.Compose(
                [transforms.Resize(size=(256, 256)), transforms.ToTensor(), normalize,]
            ),
        ),
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    lfw_test_loader = torch.utils.data.DataLoader(
        LFW(
            args.data_lfw, transforms.Compose([transforms.ToTensor(), normalize,]),
        ),  # celebA mean variance
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # define loss function (criterion) and optimizer
    if args.lw:  # loss weight
        print("=> loading CE loss_weight")
        criterion = nn.BCEWithLogitsLoss(
            reduction="mean", weight=1 / torch.sqrt(train_sample_prob)
        ).cuda()
    else:
        # criterion = nn.CrossEntropyLoss().cuda()
        criterion = nn.BCEWithLogitsLoss(reduction="mean").cuda()

    if args.focal:
        print("=> using focal loss")
        criterion = FocalLoss(criterion, balance_param=5)

    print("=> using wd {}".format(args.weight_decay))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(
                os.path.join(args.checkpoint, "log.txt"), title=title, resume=True
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, "log.txt"), title=title)
        logger.set_names(
            [
                "Learning Rate",
                "Train Loss",
                "Valid Loss",
                "Train Acc.",
                "Valid Acc.",
                "LFW Loss.",
                "LFW Acc.",
            ]
        )

    if args.evaluate:  # TODO
        validate(test_loader, model, criterion)
        # stat(train_loader)
        return
    if args.validate:  # TODO
        validate(val_loader, model, criterion)
        # stat(train_loader)
        return

    if args.evaluate_lfw:
        validate(val_loader, model, criterion)
        validate(lfw_test_loader, model, criterion)
        return

    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, "logs"))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch)

        print("\nEpoch: [%d | %d] LR: %f" % (epoch + 1, args.epochs, lr))

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, prec1 = validate(val_loader, model, criterion)

        # evaluate on lfw
        lfw_loss, lfw_prec1 = validate(lfw_test_loader, model, criterion)

        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, prec1, lfw_loss, lfw_prec1])

        # tensorboardX
        writer.add_scalar("learning rate", lr, epoch + 1)
        writer.add_scalars(
            "loss",
            {
                "train loss": train_loss,
                "validation loss": val_loss,
                "lfw loss": lfw_loss,
            },
            epoch + 1,
        )
        writer.add_scalars(
            "accuracy",
            {
                "train accuracy": train_acc,
                "validation accuracy": prec1,
                "lfw accuracy": lfw_prec1,
            },
            epoch + 1,
        )
        # for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            checkpoint=args.checkpoint,
        )

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, "log.eps"))
    writer.close()

    print("Best accuracy:")
    print(best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    bar = Bar("Processing", max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    loss_avg = 0
    prec1_avg = 0

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        # measure accuracy and record loss

        #  ==== bceloss === #

        loss = criterion(output, target)  # calcualte sum loss over all attributes.
        losses.update(loss.item(), input.size(0))
        loss_avg = losses.avg

        top1.update(accuracy_bce(output, target).item(), input.size(0))
        prec1_avg = top1.avg

        # =============== #

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss_sum = sum(loss)
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}".format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss_avg,
            top1=prec1_avg,
        )
        bar.next()
    bar.finish()
    return (loss_avg, prec1_avg)


def validate(val_loader, model, criterion):
    bar = Bar("Processing", max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    loss_avg = 0
    prec1_avg = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            # measure accuracy and record loss

            #  ==== bceloss === #

            loss = criterion(output, target)  # calcualte sum loss over all attributes.
            losses.update(loss.item(), input.size(0))
            loss_avg = losses.avg

            top1.update(accuracy_bce(output, target).item(), input.size(0))
            prec1_avg = top1.avg

            # ================= #

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}".format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=loss_avg,
                top1=prec1_avg,
            )
            bar.next()
    bar.finish()
    return (loss_avg, prec1_avg)


def save_checkpoint(
    state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]["lr"]
    """Sets the learning rate to the initial LR decayed by 10 following schedule"""
    if args.lr_decay == "step":
        lr = args.lr * (args.gamma ** (epoch // args.step))
    elif args.lr_decay == "cos":
        lr = args.lr * (1 + cos(pi * epoch / args.epochs)) / 2
    elif args.lr_decay == "linear":
        lr = args.lr * (1 - epoch / args.epochs)
    elif args.lr_decay == "linear2exp":
        if epoch < args.turning_point + 1:
            # learning rate decay as 95% at the turning point (1 / 95% = 1.0526)
            lr = args.lr * (1 - epoch / int(args.turning_point * 1.0526))
        else:
            lr *= args.gamma
    elif args.lr_decay == "schedule":
        if epoch in args.schedule:
            lr *= args.gamma
    else:
        raise ValueError("Unknown lr mode {}".format(args.lr_decay))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


if __name__ == "__main__":
    main()
