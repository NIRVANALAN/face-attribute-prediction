import torch
import numpy as np
import torch.utils.data as data

from PIL import Image
import os
import os.path


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CelebA(data.Dataset):
    def __init__(
        self,
        root,
        ann_file,
        transform=None,
        target_transform=None,
        loader=default_loader,
    ):
        images = []
        targets = []

        for line in open(os.path.join(root, ann_file), "r"):
            sample = line.split()
            if len(sample) != 41:
                raise (
                    RuntimeError(
                        "# Annotated face attributes of CelebA dataset should not be different from 40"
                    )
                )
            images.append(sample[0])
            targets.append([int(i) for i in sample[1:]])
        self.images = [os.path.join(root, "img_align_celeba", img) for img in images]
        # self.images = [os.path.join(root, 'img_align_celeba_png', img) for img in images]
        self.targets = np.array(targets)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)  # 178*218
        target = self.targets[index]
        target = torch.FloatTensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)

    def __class_sample_prob__(self):
        sample_prob = np.sum(self.targets, axis=0) / self.targets.shape[0]
        return torch.Tensor(sample_prob).cuda()


class LFW(data.Dataset):
    def __init__(
        self, root, transform=None, target_transform=None, loader=default_loader,
    ):

        with open(os.path.join(root, "label.txt")) as targets:
            targets = targets.readlines()
            targets = [list(map(int, l.strip().split(","))) for l in targets]

        with open(os.path.join(root, "name.txt")) as images:
            images = images.readlines()
            images = [i.strip().replace("\\", "/") for i in images]

        self.images = [os.path.join(root, "testset", img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        # img_names = self.images_name[index]
        target = self.targets[index]
        target = torch.FloatTensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
        # return img_names, sample

    def __len__(self):
        return len(self.images)


class Partition(object):
    def __init__(self, data_path):
        self.train_file = os.path.join(data_path, "train_40_att_list.txt")
        self.val_file = os.path.join(data_path, "val_40_att_list.txt")
        self.test_file = os.path.join(data_path, "test_40_att_list.txt")
        self.raw_label_path = os.path.join(data_path, "list_attr_celeba.txt")

        partition_file = open((os.path.join(data_path, "list_eval_partition.txt")))
        self.part_dict = dict()
        for line in partition_file:
            line_sp = line.split()
            name = line_sp[0]
            part = line_sp[1]
            self.part_dict[name] = part
        partition_file.close()

    def create_partition(self, save_path, index):
        label_file = open(self.raw_label_path)
        save_path = getattr(self, save_path)
        with open(save_path, "w") as result_file:
            for line in label_file:
                line_sp = line.split()
                if line_sp[0][-3:] == "jpg" and self.part_dict[line_sp[0]] == index:
                    result_line = " ".join(line_sp)
                    result_file.write(result_line + "\n")
        result_file.close()
        label_file.close()
