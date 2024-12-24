import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import  time
from utils.utils import save_frames_as_gif
import matplotlib.pyplot as plt
from PIL import Image

class DiffusionDataset(Dataset):
    def __init__(self, data_path, scale_to_minus1_1=True):
        """
        假设 data 包含:
        data['inputs']: (N, h, w, c) 单张输入帧
        data['labels']: (N, seq, h, w, c) 对应的未来序列
        
        最终:
        input_tensor: (b, 3, 64, 64)
        label_tensor: (b, 7, 3, 64, 64)
        """
        data = np.load(data_path, allow_pickle=True).item()
        self.inputs = data['labels']   # (N, h, w, c)
        self.labels = data['inputs']   # (N, seq, h, w, c)
        self.scale_to_minus1_1 = scale_to_minus1_1

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_frame = self.inputs[idx]     # (h, w, c)
        label_frames = self.labels[idx]    # (seq, h, w, c)

        input_tensor = torch.tensor(input_frame).float()
        label_tensor = torch.tensor(label_frames).float()

        # 如果数据是 [0,255]，则转换到 [-1,1]
        if self.scale_to_minus1_1:
            input_tensor = (input_tensor / 255.0) * 2.0 - 1.0
            label_tensor = (label_tensor / 255.0) * 2.0 - 1.0

        # 调整维度顺序
        # input: (h, w, c) -> (c, h, w)
        input_tensor = input_tensor.permute(2, 0, 1)
        # label: (seq, h, w, c) -> (seq, c, h, w)
        label_tensor = label_tensor.permute(0, 3, 1, 2)

        # rand = torch.rand

        return input_tensor, label_tensor

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, latent_dim):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.gamma_fc = nn.Linear(latent_dim, num_features)
        self.beta_fc = nn.Linear(latent_dim, num_features)
        # 初始化为1和0，以便初始时不改变批归一化的输出
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x, z):
        gamma = self.gamma_fc(z).unsqueeze(2).unsqueeze(3)
        beta = self.beta_fc(z).unsqueeze(2).unsqueeze(3)
        out = F.batch_norm(x, running_mean=None, running_var=None, training=True)
        out = gamma * out + beta
        return out

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, latent_dim):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.gamma_fc = nn.Linear(latent_dim, num_features)
        self.beta_fc = nn.Linear(latent_dim, num_features)
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x, z):
        gamma = self.gamma_fc(z).unsqueeze(2).unsqueeze(3)
        beta = self.beta_fc(z).unsqueeze(2).unsqueeze(3)
        out = F.batch_norm(x, running_mean=None, running_var=None, training=True)
        out = gamma * out + beta
        return out

class EncoderDecoderModelCBN(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, seq_length=7, latent_dim=32, num_classes=2):
        super(EncoderDecoderModelCBN, self).__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # 将潜在向量z映射到 (latent_dim, 64, 64)
        self.fc = nn.Linear(latent_dim, latent_dim * 64 * 64)

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels + latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), # 2x2
            nn.ReLU(inplace=True),
        )

        # 无监督分类头（这里只是一个线性层输出类logits）
        # 输入尺寸：bottleneck后是 (b, 512, 2, 2)，flatten成 (b, 512*2*2)
        self.classifier = nn.Linear(512*2*2, self.num_classes)

        # 类嵌入，用来为每个类生成一个特定的条件向量
        self.class_emb = nn.Embedding(self.num_classes, latent_dim)
        nn.init.normal_(self.class_emb.weight, 0, 0.02) # 可根据需要初始化

        # Decoder with CBN layers
        # 解码器中使用ConditionalBatchNorm2d，需要latent_dim作为条件输入
        self.decoder_layers = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 4x4
            nn.ReLU(inplace=True),
            ConditionalBatchNorm2d(256, latent_dim),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(inplace=True),
            ConditionalBatchNorm2d(128, latent_dim),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            ConditionalBatchNorm2d(64, latent_dim),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.ReLU(inplace=True),
            ConditionalBatchNorm2d(32, latent_dim),

            nn.ConvTranspose2d(32, output_channels * self.seq_length, kernel_size=4, stride=2, padding=1),  # 64x64
        ])

    def forward(self, x, z, fixed_class_id=None):
        # x: (b, 3, 64, 64)
        # z: (b, latent_dim)
        b, c, h, w = x.size()

        # 将z映射成图像特征，与输入图像拼接
        z_feat = self.fc(z).view(b, self.latent_dim, 64, 64)  # (b, latent_dim, 64, 64)
        x_cond = torch.cat([x, z_feat], dim=1)  # (b, c + latent_dim, 64, 64)

        encoded = self.encoder(x_cond)
        bottleneck = self.bottleneck(encoded) # (b, 512, 2, 2)

        # 分类部分
        bottleneck_flat = bottleneck.view(b, -1)
        class_logits = self.classifier(bottleneck_flat)  # (b, num_classes)
        
        if fixed_class_id is None:
            # 如果未指定类ID，则使用argmax作为类的选择示例
            class_id = torch.argmax(class_logits, dim=1)
        else:
            # 使用固定给定的类ID（如需要特定条件控制）
            class_id = torch.full((b,), fixed_class_id, dtype=torch.long, device=x.device)

        # 获取类嵌入向量
        class_vector = self.class_emb(class_id) # (b, latent_dim)

        # 将类嵌入与z结合，这里简单相加作为条件向量
        z_combined = z + class_vector

        # 根据新条件z_combined再生成特征输入decoder
        # 与原实现不同，这里用z_combined替换z去生成z_feat是可行的选择
        # 如果希望保持与编码器输入一致，可在最初就使用z_combined代替z。
        # 为了简单，这里继续使用z_combined来对decoder中的CBN做条件。
        # decoder不需要重新映射z了，因为decoder的CBN直接接收z，本来就是从forward中传入的。
        # 不过我们最初对x做拼接时使用的是原z_feat，理论上如果需要严格意义上用分类后的z_combined作为条件，
        # 需要重新生成z_feat用于encoder的输入。但这样会导致训练不稳定（因为分类依赖encoder输出，又改变encoder输入）。
        # 简化起见，不改encoder输入，只改decoder时的条件输入。
        # 如果要求严格条件作用在最终生成，可将fc映射的代码由z替换成z_combined。
        
        # 若希望decoder也使用z_combined生成特征图，可重复:
        # z_feat_dec = self.fc(z_combined).view(b, self.latent_dim, 64, 64)
        # 但这样需对网络结构进行调整。
        # 这里示例就直接在CBN时传入z_combined。
        
        out = bottleneck
        idx = 0
        while idx < len(self.decoder_layers):
            layer = self.decoder_layers[idx]
            if isinstance(layer, ConditionalBatchNorm2d):
                # CBN层使用z_combined作为条件
                out = layer(out, z_combined)
                idx += 1
            else:
                out = layer(out)
                idx += 1

        # reshape to (b, seq, c, h, w)
        out = out.view(b, self.seq_length, c, h, w)
        return out, class_logits

def main():
    output_data_dir = "data/diffusion_model_data"
    data_path = os.path.join(output_data_dir, "diffusion_dataset.npy")
    checkpoint_path = os.path.join(output_data_dir, "model_checkpoint.pth")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集文件未找到: {data_path}")

    dataset = DiffusionDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    seq_length = 7
    latent_dim = 2
    model = EncoderDecoderModelCBN(input_channels=3, output_channels=3, seq_length=seq_length, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    

    is_train = False
    if is_train:
        num_epochs = 10000
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (input_tensor  , target_video) in enumerate(dataloader):
                # input_tensor: (b, 3, 64, 64)
                # target_video: (b, 7, 3, 64, 64)
                input_tensor = input_tensor.to(device)
                target_video = target_video.to(device)

                # if batch_idx == 1:
                #     z = torch.zeros(input_tensor.size(0), latent_dim, device=device)
                #     # print("0")
                # else:
                #     z = torch.ones(input_tensor.size(0), latent_dim, device=device)
                #     # print("1")

                z = torch.randn(input_tensor.size(0), latent_dim, device=device)

                optimizer.zero_grad()
                output, class_logits  = model(input_tensor, z) # (b, 7, 3, 64, 64)
                loss = loss_fn(output, target_video)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
            if (epoch + 1) % 1000 == 0:
                current_checkpoint_path = checkpoint_path.replace(".pth", f"_epoch{epoch+1}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, current_checkpoint_path)
                print(f"已保存 checkpoint 到: {current_checkpoint_path}")
        print("训练完成！")
    else:
        # ========== 以下是推理阶段示例 ==========

        dataset = DiffusionDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seq_length = 7
        latent_dim = 2

        model = EncoderDecoderModelCBN(
            input_channels=3, output_channels=3,
            seq_length=seq_length, latent_dim=latent_dim
        ).to(device)

        # 加载已经训练完的权重
        checkpoint_path = os.path.join(output_data_dir, "model_checkpoint_epoch10000.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        # 随机取一个batch
        input_sample, target_sample = next(iter(dataloader))
        input_sample = input_sample.to(device)

        # 目录准备
        output_dir = "output_images"
        os.makedirs(output_dir, exist_ok=True)

        # ========== 1. 保存 输入帧（仅示例第 1 张）==========
        # input_sample.shape: (1, 3, 64, 64)
        input_frame = input_sample[0]  # (3, 64, 64)
        # [-1, 1] -> [0,1]
        input_frame = (input_frame + 1.0) / 2.0
        input_frame = input_frame.clamp(0, 1)
        # 转到 [0,255] -> PIL
        input_frame_pil = Image.fromarray(
            (input_frame.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8),
            mode='RGB'
        )
        input_png_path = os.path.join(output_dir, "input_frame.png")
        input_frame_pil.save(input_png_path)
        print(f"已保存输入帧到: {input_png_path}")

        # ========== 2. 生成两种条件 (z=0, z=1) 输出序列并保存为 gif ==========
        all_conditions = {
            "z0": torch.zeros(input_sample.size(0), latent_dim, device=device),
            "z1": torch.ones(input_sample.size(0), latent_dim, device=device)
        }

        for cond_name, z_vec in all_conditions.items():
            with torch.no_grad():
                out_video = model(input_sample, z_vec)  # (1, 7, 3, 64, 64)

            # 去掉 batch 维度: (7, 3, 64, 64)
            video_frames = out_video.squeeze(0)
            # 映射到 [0,1] 再到 [0,255]
            video_frames = ((video_frames + 1.0) / 2.0).clamp(0, 1) * 255
            video_frames = video_frames.byte()

            # 收集所有帧的 PIL Image
            pil_frames = []
            for frame_idx in range(video_frames.size(0)):
                frame = video_frames[frame_idx]  # (3, 64, 64)
                frame = frame.permute(1, 2, 0).cpu().numpy()  # (64, 64, 3)
                pil_img = Image.fromarray(frame, mode='RGB')
                pil_frames.append(pil_img)

            # 保存为 gif
            gif_path = os.path.join(output_dir, f"{cond_name}_output.gif")
            save_frames_as_gif(pil_frames, gif_path, duration=200)

        # ========== 3. 画图展示输入 + 两种输出（示例）==========
        # 这里只展示各自的第 1 帧来对比，更多可自行发挥
        out_z0_first_frame = Image.open(os.path.join(output_dir, "z0_output.gif"))
        out_z1_first_frame = Image.open(os.path.join(output_dir, "z1_output.gif"))

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(input_frame_pil)
        axs[0].set_title("Input Frame")
        axs[0].axis('off')

        axs[1].imshow(out_z0_first_frame)
        axs[1].set_title("z=0 First Frame")
        axs[1].axis('off')

        axs[2].imshow(out_z1_first_frame)
        axs[2].set_title("z=1 First Frame")
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
