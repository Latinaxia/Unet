import cv2
import os
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ✅ 数据增强管道
transform = A.Compose([
    # ✅ 等比缩放 + 填充黑边
    A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_AREA),
    A.PadIfNeeded(
        min_height=512,
        min_width=512,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,  # ✅ 修正参数
    ),

    # ✅ 数据增强
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.6),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.GridDropout(ratio=0.4, p=0.3),  # ✅ 替换 CoarseDropout

    # ✅ 标准化和张量转换
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
], additional_targets={"degraded": "image"})


class RestorationDataset:
    def __init__(self, root_dir, transform=None, max_samples=10000):
        """
        Args:
            root_dir (str): 包含图像的根目录
            transform (albumentations.Compose): 可选的增强管道
            max_samples (int): 最多加载图像数量
        """
        self.root_dir = root_dir
        self.image_files = []
        self.transform = transform  # ✅ 支持外部传入 transform

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
                    if len(self.image_files) >= max_samples:
                        break
            if len(self.image_files) >= max_samples:
                break
        self.image_files = self.image_files[:max_samples]
        print(f"✅ Loaded {len(self.image_files)} images from {root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        # 添加人工瑕疵
        degraded = self.add_artifacts(image.copy())

        # ✅ 可选的增强操作
        if self.transform:
            transformed = self.transform(image=image, degraded=degraded)
            image = transformed["image"]  # CHW tensor
            degraded = transformed["degraded"]  # CHW tensor

        return degraded, image

    def add_artifacts(self, img):
        """添加多种老化效果"""
        # 添加划痕
        if random.random() > 0.5:
            for _ in range(random.randint(1, 5)):
                x1, y1 = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
                x2, y2 = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                thickness = random.randint(1, 3)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        # 添加噪声
        if random.random() > 0.5:
            noise = np.random.randn(*img.shape) * random.uniform(0, 25)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        # 褪色效果
        fading = np.zeros_like(img)
        fading[:, :, random.choice([0, 1])] = random.randint(20, 80)
        img = cv2.addWeighted(img, 1, fading, 0.5, 0)

        return img