import torch
import torchvision

import os
import cv2
import random
import numpy as np
import easyaug.transforms as T


class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, list_path, img_root, label_root, transform=None):
        super(DemoDataset, self).__init__()
        self.list_path = list_path
        self.img_root = img_root
        self.label_root = label_root
        self.transform = transform

        self.img_list = []
        self.label_list = []
        for line in open(list_path):
            img_name = line.strip() + ".jpg"
            label_name = line.strip() + ".txt"
            self.img_list.append(os.path.join(img_root, img_name))
            self.label_list.append(os.path.join(label_root, label_name))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label_path = self.label_list[index]

        img = torchvision.io.read_file(img_path)
        img = torchvision.io.decode_image(img).cuda()
        label = np.loadtxt(label_path).reshape(-1, 5)
        label = label[:, [1, 2, 3, 4, 0]]

        if self.transform is not None:
            result = self.transform([("img", img), ("box", label)])
            return result[0][1], result[1][1]
        else:
            return img, label


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.RandomResize((480, 480), scale=(0.25, 2), jitter=0.3),
            T.RandomFilp(),
            T.RandomCrop((480, 480), 127),
            T.Saturation(1.5),
            T.Hue(0.1),
            T.Normalize([0, 0, 0], [255, 255, 255]),
        ]
    )
    train_dataset = DemoDataset(
        "./data/list.txt", "./data/JPEGImages", "./data/labels", transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, drop_last=True, num_workers=0
    )

    for step, batch_data in enumerate(train_loader):
        imgs, labels = batch_data
        print(imgs.shape, labels.shape)

        imgs = imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
        for index in range(2):
            img = imgs[index][:, :, ::-1].copy()
            out_size = img.shape
            label = labels[index]
            for box in label:
                cx, cy, cw, ch, cls = box
                if cw != 0 and ch != 0:
                    x1 = (cx - cw / 2) * out_size[1]
                    x2 = (cx + cw / 2) * out_size[1]
                    y1 = (cy - ch / 2) * out_size[0]
                    y2 = (cy + ch / 2) * out_size[0]
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(f"results/{step}_{index}.jpg", img)
