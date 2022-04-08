import torch
import torchvision
import torchvision.transforms.functional as F
import random
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)
        return input


class BaseTransform:
    def __call__(self, input):
        result = []
        self.get_params(input)
        for date_type, data in input:
            func = getattr(self, "apply_" + date_type, "None")
            if func is not None:
                result.append((date_type, func(data)))
            else:
                print(f"Unexpected type: {date_type}")
                result.append((date_type, data))
        return result

    def get_params(self, input):
        self.input_size = self.get_input_size(input)

    def get_input_size(self, input):
        img_w = None
        img_h = None
        for date_type, data in input:
            if date_type == "img" or date_type == "mask":
                h, w = data.shape[-2:]
                if img_w is None:
                    img_w, img_h = w, h
                else:
                    assert img_w == w
                    assert img_h == h
        return img_h, img_w

    def apply_img(self, img: np.ndarray):
        return img

    def apply_mask(self, mask: np.ndarray):
        return mask

    def apply_box(self, box: np.ndarray):
        return box


class Resize(BaseTransform):
    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size

    def apply_img(self, img):
        return F.resize(img, self.size)

    def apply_mask(self, mask):
        return F.resize(mask, self.size, interpolation=F.InterpolationMode("nearest"))


class RandomResize(Resize):
    def __init__(self, size, scale=(0.25, 2), jitter=0.3):
        super(Resize, self).__init__()
        self.base_size = size
        self.scale = scale
        self.jitter = jitter

    def get_params(self, input):
        self.input_size = self.get_input_size(input)
        img_h, img_w = self.input_size
        height, width = self.base_size
        scale = random.uniform(self.scale[0], self.scale[1])
        img_ratio = img_w / img_h
        new_ratio = (
            img_ratio
            * random.uniform(1 - self.jitter, 1 + self.jitter)
            / random.uniform(1 - self.jitter, 1 + self.jitter)
        )
        if new_ratio < 1:
            nh = int(scale * height)
            nw = int(nh * new_ratio)
        else:
            nw = int(scale * width)
            nh = int(nw / new_ratio)

        self.size = (nh, nw)


class RandomCrop(BaseTransform):
    def __init__(self, output_size, fill=0):
        super(RandomCrop, self).__init__()
        self.output_size = output_size
        self.fill = fill

    def get_params(self, input):
        self.input_size = self.get_input_size(input)
        self.img_h, self.img_w = self.input_size
        self.height, self.width = self.output_size

        dx = int(random.uniform(0, self.img_w - self.width))
        dy = int(random.uniform(0, self.img_h - self.height))
        self.top = max(0, dy)
        self.left = max(0, dx)

        self.pad_if_needed = False
        self.padding = [0, 0, 0, 0]
        if dx < 0 or dy < 0:
            self.pad_if_needed = True
            if dx < 0:
                self.padding[0] = abs(dx)
                self.padding[2] = self.width - self.img_w - abs(dx)
            if dy < 0:
                self.padding[1] = abs(dy)
                self.padding[3] = self.height - self.img_h - abs(dy)

    def apply_img(self, img):
        if self.pad_if_needed:
            img = F.pad(img, self.padding, self.fill)
        return F.crop(img, self.top, self.left, self.height, self.width)

    def apply_mask(self, mask):
        if self.pad_if_needed:
            img = F.pad(mask, self.padding, self.fill)
        return F.crop(img, self.top, self.left, self.height, self.width)

    def apply_box(self, box):
        x1 = (box[:, 0] - box[:, 2] / 2) * self.img_w - self.left + self.padding[0]
        y1 = (box[:, 1] - box[:, 3] / 2) * self.img_h - self.top + self.padding[1]
        x2 = (box[:, 0] + box[:, 2] / 2) * self.img_w - self.left + self.padding[0]
        y2 = (box[:, 1] + box[:, 3] / 2) * self.img_h - self.top + self.padding[1]
        x1 = np.clip(x1 / self.width, 0.0, 1.0)
        y1 = np.clip(y1 / self.height, 0.0, 1.0)
        x2 = np.clip(x2 / self.width, 0.0, 1.0)
        y2 = np.clip(y2 / self.height, 0.0, 1.0)

        box[:, 0] = np.clip((x2 + x1) / 2, 0.0, 1.0)
        box[:, 1] = np.clip((y2 + y1) / 2, 0.0, 1.0)
        box[:, 2] = np.clip(x2 - x1, 0.0, 1.0)
        box[:, 3] = np.clip(y2 - y1, 0.0, 1.0)
        box = box[(box[:, 2] > 0.001) & (box[:, 3] > 0.001)]

        return box


class Saturation(BaseTransform):
    def __init__(self, saturation):
        super(Saturation, self).__init__()
        self.saturation = saturation

    def get_params(self, input):
        self.ds = random.uniform(1, self.saturation)
        if random.random() < 0.5:
            self.ds = 1.0 / self.ds

    def apply_img(self, img):
        return F.adjust_saturation(img, self.ds)


class Hue(BaseTransform):
    def __init__(self, hue):
        super(Hue, self).__init__()
        self.hue = min(hue, 0.5)

    def get_params(self, input):
        self.dh = random.uniform(-self.hue, self.hue)

    def apply_img(self, img):
        return F.adjust_hue(img, self.dh)

class Filp(BaseTransform):
    def __init__(self):
        super(Filp, self).__init__()

    def apply_img(self, img):
        return F.hflip(img)
    
    def apply_mask(self, mask):
        return F.hflip(mask)
    
    def apply_box(self, box):
        box[:, 0] = 1 - box[:, 0]
        return box

class RandomFilp(BaseTransform):
    def __init__(self, p=0.5):
        super(RandomFilp, self).__init__()
        self.p = p

    def get_params(self, input):
        if random.random() < self.p:
            self.filp = True
        else:
            self.filp = False

    def apply_img(self, img):
        if self.filp:
            return F.hflip(img)
        else:
            return img
    
    def apply_mask(self, mask):
        if self.filp:
            return F.hflip(mask)
        return mask
    
    def apply_box(self, box):
        if self.filp:
            box[:, 0] = 1 - box[:, 0]
            return box
        return box

class Normalize(BaseTransform):
    def __init__(self, mean=[0, 0, 0], std=[255, 255, 255], max_objects=50):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.max_objects = max_objects

    def apply_img(self, img):
        return F.normalize(img.float(), self.mean, self.std)

    def apply_box(self, box):
        box = box[(box[:, 2] > 0.001) & (box[:, 3] > 0.001)]
        filled_box = np.zeros((self.max_objects, 5))
        if len(box) >= self.max_objects:
            print(f"Unexpected Box num: {len(box)} > {self.max_objects}")
            filled_box = box[: self.max_objects]
        else:
            filled_box[: len(box)] = box
        return filled_box
