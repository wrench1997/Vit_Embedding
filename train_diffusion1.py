import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from dataset.game_dataset import DiffusionDataset
from torch.utils.data import DataLoader, Dataset
import os

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# 设定数据集目录
output_data_dir = "data/diffusion_model_data"
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)




    
# 加载自定义数据集
# 加载数据集
data = np.load(os.path.join(output_data_dir, "diffusion_dataset.npy"), allow_pickle=True).item()
dataset = DiffusionDataset(data)
train_dataloader = DataLoader(dataset, batch_size=1)


# 定义条件UNet
class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=256, class_emb_size=4):
        super().__init__()

        # 嵌入层将类别标签映射到一个class_emb_size大小的向量
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model是一个无条件的UNet, 添加了额外的输入通道来接受条件信息（类别嵌入）
        self.model = UNet2DModel(
            sample_size=64,  # 目标图像分辨率
            in_channels = 33,  # 额外的输入通道用于类别条件
            out_channels=7,  # 输出通道为3（RGB图像）
            layers_per_block=2,  # 每个UNet块中的ResNet层数
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",  # 正常的ResNet下采样块
                "AttnDownBlock2D",  # 使用空间自注意力的ResNet下采样块
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",  # 使用空间自注意力的ResNet上采样块
                "UpBlock2D",  # 正常的ResNet上采样块
            ),
        )

    def forward(self, x, t, class_labels):
        # 获取输入x的尺寸
        bs, seq, w, h, ch= x.shape
        embedding = class_labels.view(-1).long()  # 结果形状: [12288]




        # 类别条件的形状处理，将类别嵌入转化为合适的形状
        class_cond = self.class_emb(embedding)  # 映射到嵌入维度
        class_cond = class_cond.view(bs,4, w, h, ch)


        assert class_cond.device == x.device, "两个张量必须在相同的设备上"
        assert class_cond.dtype == x.dtype, "两个张量的数据类型必须相同"

        # x的形状为(bs, 7, 64, 64)（7帧），class_cond的形状为(bs, 4, 64, 64)

        # 将x和class_cond拼接在一起作为输入
        net_input = torch.cat((x, class_cond), dim=1)  # (1, 11, 64, 64,3)

 
        net_input = net_input.squeeze()


        net_input =  net_input.permute(0,3,1,2)  # (11, 64, 64,3)

        net_input = net_input.reshape(bs, 11 * 3, 64, 64)  # [bs, 33, 64, 64]

        ouput = self.model(net_input, t).sample  # (bs, 33,64, 64)

        ouput = ouput.unsqueeze(-1)
        # 将数据输入UNet，并返回预测
        return ouput
    

# 创建一个噪声调度器
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")


# 训练循环
n_epochs = 10

# 初始化网络
net = ClassConditionedUnet().to(device)

# 损失函数
loss_fn = nn.MSELoss()

# 优化器
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# 记录损失
losses = []

net.train()

# 训练循环
for epoch in range(n_epochs):
    for x, y in tqdm(train_dataloader):

        # 获取数据并准备腐化版本
        x = x.to(device)
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # 获取模型预测
        pred = net(noisy_x, timesteps, y)  # 这里我们传入标签y

        # 计算损失
        loss = loss_fn(pred, noise)  # 预测与噪声的差距

        # 反向传播并更新参数
        opt.zero_grad()
        loss.backward()
        opt.step()

        # 存储损失值
        losses.append(loss.item())

    # 每个epoch后打印出最近100个损失的平均值
    avg_loss = sum(losses[-100:]) / 100
    print(f"Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}")

# 绘制损失曲线
plt.plot(losses)


# # 随机生成一些样本
# x = torch.randn(80, 7, 64, 64).to(device)  # 7帧输入
# y = torch.tensor([[i] * 8 for i in range(10)]).flatten().to(device)  # 随机生成的标签

# # 采样循环
# for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

#     # 获取模型的预测
#     with torch.no_grad():
#         residual = net(x, t, y)  # 传入标签y

#     # 更新样本
#     x = noise_scheduler.step(residual, t, x).prev_sample

# # 显示结果
# fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap="Greys")

# input('exit....')
