import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import logging
import numpy as np
from models.unet import UNet
from utils.data_augmentation import RestorationDataset  # 确保路径正确
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm
# 设置环境变量以减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 配置参数
config = {
    "batch_size": 14,
    "epochs": 50,
    "learning_rate": 2e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_interval": 5,
    "log_interval": 100,
    "accum_steps": 4,
}

# 设置日志记录
logging.basicConfig(
    filename="logs/training.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)

def train():
    # ✅ 修改数据集路径为新路径
    dataset = RestorationDataset("/root/autodl-tmp/data/train/CASIA-WebFace", max_samples=10000)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 初始化模型
    model = UNet().to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # 判别器
    discriminator = Discriminator(in_channels=3).to(config["device"])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate"])

    # 损失函数
    criterion_pixel = nn.L1Loss() # 像素级 L1 损失
    criterion_perceptual = PerceptualLoss().to(config["device"]) # 感知损失 使用 VGG 提取高层语义特征，保证结构一致（提升清晰度）
    criterion_adversarial = nn.MSELoss()# 对抗损失通过判别器引导生成器输出更真实的图像（提升质感）

    # 混合精度训练
    scaler = GradScaler("cuda")

    # 损失记录
    loss_history = []

    # 训练循环
    for epoch in tqdm(range(config["epochs"]), desc="Training Epochs", total=config["epochs"]):
        model.train()
        discriminator.train()
        total_loss_G = 0.0
        total_loss_D = 0.0
        optimizer.zero_grad()
        optimizer_D.zero_grad()

        progress_bar = tqdm(
            enumerate(dataloader),
            desc=f"Epoch {epoch + 1}/{config['epochs']} Progress",
            total=len(dataloader),
            leave=False
        )

        for i, (degraded, target) in progress_bar:
            degraded = degraded.to(config["device"]).float()
            target = target.to(config["device"]).float()

            valid = torch.ones(degraded.size(0), 1, 30, 30).to(config["device"])
            fake = torch.zeros_like(valid)

            """ 
            loss_G生成图像质 逐渐下降 → 稳定
            loss_D判别器强度在 0.3 ~ 0.7 波动
            """
            # Step 1: 训练判别器
            with autocast(device_type="cuda"):
                output = model(degraded)
                pred_real = discriminator(degraded, target)
                loss_real = criterion_adversarial(pred_real, valid)
                pred_fake = discriminator(degraded, output.detach())
                loss_fake = criterion_adversarial(pred_fake, fake)
                loss_D = (loss_real + loss_fake) / 2

            scaler.scale(loss_D).backward()
            if (i + 1) % config["accum_steps"] == 0:
                scaler.step(optimizer_D)
                scaler.update()
                optimizer_D.zero_grad()

            # Step 2: 训练生成器
            with autocast(device_type="cuda"):
                output = model(degraded)
                loss_pixel = criterion_pixel(output, target)
                loss_perceptual = criterion_perceptual(output, target)
                
                pred_fake = discriminator(degraded, output)
                loss_adv = criterion_adversarial(pred_fake, valid)

                # 综合损失 
                loss_G = loss_pixel + 0.1 * loss_perceptual + 0.01 * loss_adv
                loss_G = loss_G / config["accum_steps"]

            scaler.scale(loss_G).backward()
            if (i + 1) % config["accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss_G += loss_G.item() * config["accum_steps"]
            total_loss_D += loss_D.item()

            # ✅ 更新进度条信息
            progress_bar.set_postfix({
                "Loss_G": f"{loss_G.item() * config['accum_steps']:.4f}",
                "Loss_D": f"{loss_D.item():.4f}"
            })

            del degraded, target, output, loss_G, loss_D
            torch.cuda.empty_cache()

        scheduler.step()

        avg_loss_G = total_loss_G / len(dataloader)
        avg_loss_D = total_loss_D / len(dataloader)
        loss_history.append(avg_loss_G)
        logging.info(
            f"Epoch {epoch + 1} Average Loss_G: {avg_loss_G:.4f}, Loss_D: {avg_loss_D:.4f}"
        )

        # save_interval次epoch 保存一次模型和 loss 曲线图
        if (epoch + 1) % config["save_interval"] == 0:
            torch.save(model.state_dict(), f"models/unet_epoch_{epoch + 1}.pth")
            torch.save(
                discriminator.state_dict(),
                f"models/discriminator_epoch_{epoch + 1}.pth",
            )
            print(f"Model saved at epoch {epoch + 1}")

            plt.figure(figsize=(10, 5))
            plt.plot(loss_history, label="Generator Loss")
            plt.title("Training Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig("logs/loss_curve.png")
            plt.close()

    print("Training complete")
    torch.save(model.state_dict(), "models/unet_final.pth")
    torch.save(discriminator.state_dict(), "models/discriminator_final.pth")


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),  # 输入 6 通道
            *discriminator_block(64, 128),                                    # 输出 128 通道
            *discriminator_block(128, 256),                                   # 输出 256 通道
            *discriminator_block(256, 512),                                   # 输出 512 通道
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),             # 输出 32x32
            nn.AdaptiveAvgPool2d((30, 30)),                                   # 调整为 30x30
        )

    def forward(self, img_A, img_B):
        input = torch.cat((img_A, img_B), dim=1)  # 拼接两个 3 通道图像 → 6 通道
        return self.model(input)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg_pretrained_features = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        ).features
        self.slice = torch.nn.Sequential()
        for x in range(16):  # 提取前16层（conv4_3）
            self.slice.add_module(str(x), vgg_pretrained_features[x])
        self.slice.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x = (x + 1) / 2
        y = (y + 1) / 2

        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)

        features_x = self.slice(x)
        features_y = self.slice(y)
        return torch.nn.functional.mse_loss(features_x, features_y)


if __name__ == "__main__":
    train()