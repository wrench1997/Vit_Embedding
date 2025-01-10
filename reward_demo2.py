import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# ------------------------------
# 1. 自定义环境定义（KeyFrameEnv）
# ------------------------------
class KeyFrameEnv(gym.Env):
    """
    自定义环境，代理需要执行特定的动作组合才能获得奖励。
    动作是离散的，需要按正确顺序执行才能结束游戏。
    """
    def __init__(self, seq_length=4, frame_dim=128):
        super().__init__()
        self.seq_length = seq_length
        self.frame_dim = frame_dim
        
        # 动作空间:4个离散动作
        self.action_space = spaces.Discrete(4)  # 0,1,2,3四种动作
        
        # 观察空间:帧特征向量
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.frame_dim,), dtype=np.float32
        )
        
        # 正确的动作序列
        self.target_sequence = [1, 2, 0, 3]  # 需要按此顺序执行的动作
        
        # 初始化回合变量
        self.current_step = 0
        self.sequence = None
        self.action_history = []
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.action_history = []
        # 生成随机序列作为观察
        self.sequence = np.random.randn(self.seq_length, self.frame_dim).astype(np.float32)
        return self.sequence[self.current_step], {}
    
    def step(self, action):
        done = False
        reward = 0
        info = {}
        
        # 记录动作
        self.action_history.append(action)
        self.current_step += 1
        
        # 检查是否完成目标序列
        if len(self.action_history) >= len(self.target_sequence):
            # 获取最近的n个动作，其中n是目标序列长度
            recent_actions = self.action_history[-len(self.target_sequence):]
            
            # 检查是否匹配目标序列
            if recent_actions == self.target_sequence:
                done = True
                reward = 10.0  # 成功完成动作序列的奖励
                info['success'] = True
            elif self.current_step >= self.seq_length:
                done = True
                reward = -5.0  # 未能在规定步数内完成的惩罚
                info['success'] = False
        
        if not done:
            if self.current_step >= self.seq_length:
                done = True
                reward = -5.0  # 超出最大步数的惩罚
                info['success'] = False
            observation = self.sequence[min(self.current_step, self.seq_length-1)]
        else:
            observation = np.zeros(self.frame_dim, dtype=np.float32)
        
        info['action_history'] = self.action_history
        
        return observation, reward, done, info

# ------------------------------
# 2. 奖励模型设计（RewardModel）
# ------------------------------
class RewardModel(nn.Module):
    def __init__(self, frame_dim=128, hidden_dim=256, num_layers=2):
        super(RewardModel, self).__init__()
        self.lstm = nn.LSTM(input_size=frame_dim + 1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # 输出一个标量，表示回合成功的概率

    def forward(self, obs, actions, lengths):
        """
        obs: Tensor of shape (batch_size, seq_length, frame_dim)
        actions: Tensor of shape (batch_size, seq_length, 1)
        lengths: Tensor of实际序列长度 (batch_size,)
        """
        # 将观察和动作拼接在一起
        x = torch.cat([obs, actions], dim=-1)  # (batch_size, seq_length, frame_dim + 1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # 取最后一个有效时间步的输出
        idx = (lengths - 1).view(-1, 1).expand(len(lengths), out.size(2)).unsqueeze(1)
        last_output = out.gather(1, idx).squeeze(1)
        
        logits = self.fc(last_output)
        prob = torch.sigmoid(logits).squeeze(-1)  # 返回形状为 (batch_size,)
        return prob

# ------------------------------
# 3. PPO 策略模型（ActorCritic）
# ------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

# ------------------------------
# 4. 自定义奖励环境（CustomRewardEnv）
# ------------------------------
class CustomRewardEnv(gym.Env):
    def __init__(self, env, reward_model, frame_dim=128, seq_length=4, device='cpu'):
        super(CustomRewardEnv, self).__init__()
        self.env = env
        self.reward_model = reward_model
        self.frame_dim = frame_dim
        self.seq_length = seq_length
        self.device = device
        
        # 动作和观察空间继承自原环境
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
        # 存储历史观察和动作
        self.observation_history = []
        self.action_history = []
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.observation_history = [obs]
        self.action_history = []
        return obs, info
    
    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)
        self.action_history.append(action)
        self.observation_history.append(obs)
        
        # 准备输入给奖励模型
        seq_obs = self.observation_history[-self.seq_length:]
        seq_actions = self.action_history[-self.seq_length:]
        seq_length = min(len(seq_obs), self.seq_length)
        
        # 填充序列
        padded_obs = np.zeros((self.seq_length, self.frame_dim), dtype=np.float32)
        padded_obs[:seq_length] = np.array(seq_obs[-self.seq_length:])
        
        padded_actions = np.zeros((self.seq_length, 1), dtype=np.float32)
        padded_actions[:seq_length] = np.array(seq_actions[-self.seq_length:]).reshape(-1, 1)
        
        # 转换为张量
        obs_tensor = torch.FloatTensor(padded_obs).unsqueeze(0).to(self.device)  # (1, seq_length, frame_dim)
        actions_tensor = torch.FloatTensor(padded_actions).unsqueeze(0).to(self.device)  # (1, seq_length, 1)
        lengths_tensor = torch.LongTensor([seq_length]).to(self.device)  # (1,)
        
        with torch.no_grad():
            success_prob = self.reward_model(obs_tensor, actions_tensor, lengths_tensor).item()
        
        # 自定义奖励，可以根据概率调整，例如：
        custom_reward = 10.0 * success_prob - 5.0 * (1 - success_prob)
        
        # 如果回合结束，基于成功与否调整奖励
        if done:
            custom_reward = 10.0 if info.get('success', False) else -5.0
        
        return obs, custom_reward, done, info

# ------------------------------
# 5. 训练流程
# ------------------------------

# 超参数
NUM_EPISODES = 1000
GAMMA = 0.99
LR_POLICY = 3e-4
LR_REWARD = 1e-3
BATCH_SIZE = 32
SEQ_LENGTH = 4
FRAME_DIM = 128
TRAIN_REWARD_EVERY = 10  # 每10个回合训练一次奖励模型

# 环境
base_env = KeyFrameEnv(seq_length=SEQ_LENGTH, frame_dim=FRAME_DIM)
custom_env = CustomRewardEnv(base_env, reward_model=None, frame_dim=FRAME_DIM, seq_length=SEQ_LENGTH, device=device)  # 初始时reward_model为None

# 模型
reward_model = RewardModel(frame_dim=FRAME_DIM).to(device)
policy_model = ActorCritic(obs_dim=FRAME_DIM, action_dim=base_env.action_space.n).to(device)

# 优化器
reward_optimizer = optim.Adam(reward_model.parameters(), lr=LR_REWARD)

# 损失函数
bce_loss = nn.BCELoss()

# 存储用于奖励模型训练的数据
reward_dataset = []

# 使用SB3的PPO进行训练
# 这里我们需要先训练奖励模型，然后将其集成到CustomRewardEnv中
# 为了简化流程，先进行预训练奖励模型

# 预训练奖励模型（可选，根据需要进行）
# 在初始阶段，可以使用环境的奖励来训练奖励模型
# 例如，收集一些随机回合数据
pretrain_episodes = 100
for episode in range(pretrain_episodes):
    obs, _ = base_env.reset()
    done = False
    episode_obs = []
    episode_actions = []
    success = False

    while not done:
        # 随机动作
        action = base_env.action_space.sample()
        next_obs, env_reward, done, info = base_env.step(action)
        
        # 存储轨迹
        episode_obs.append(obs)
        episode_actions.append(action)
        
        obs = next_obs

    success = info.get('success', False)
    reward_dataset.append((episode_obs, episode_actions, success))

# 准备训练奖励模型
all_obs = []
all_actions = []
all_labels = []
for ep_obs, ep_actions, ep_success in reward_dataset:
    for t in range(len(ep_obs)):
        seq_obs = ep_obs[:t+1]
        seq_actions = ep_actions[:t+1]
        seq_length = min(len(seq_obs), SEQ_LENGTH)
        
        # 填充序列
        padded_obs = np.zeros((SEQ_LENGTH, FRAME_DIM), dtype=np.float32)
        padded_obs[:seq_length] = np.array(seq_obs[-SEQ_LENGTH:])
        
        padded_actions = np.zeros((SEQ_LENGTH, 1), dtype=np.float32)
        padded_actions[:seq_length] = np.array(seq_actions[-SEQ_LENGTH:]).reshape(-1, 1)
        
        all_obs.append(padded_obs)
        all_actions.append(padded_actions)
        all_labels.append(float(ep_success))  # 1.0 表示成功，0.0 表示失败

all_obs = torch.FloatTensor(all_obs).to(device)
all_actions = torch.FloatTensor(all_actions).to(device)
all_labels = torch.FloatTensor(all_labels).to(device)
lengths = torch.LongTensor([min(len(seq), SEQ_LENGTH) for seq in [ep_obs for ep_obs, _, _ in reward_dataset] for _ in range(len(seq))]).to(device)

dataset = TensorDataset(all_obs, all_actions, all_labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 训练奖励模型
reward_model.train()
for epoch in range(5):  # 训练5个epoch
    epoch_loss = 0.0
    for batch_obs, batch_actions, batch_labels in dataloader:
        pred_probs = reward_model(batch_obs, batch_actions, lengths[:batch_obs.size(0)])
        loss = bce_loss(pred_probs, batch_labels)
        reward_optimizer.zero_grad()
        loss.backward()
        reward_optimizer.step()
        epoch_loss += loss.item()
    print(f"Pretrain Epoch {epoch + 1}: Loss {epoch_loss / len(dataloader):.4f}")

# 将训练好的奖励模型集成到CustomRewardEnv
custom_env.reward_model = reward_model

# 检查环境
check_env(custom_env)

# 创建SB3的PPO模型
ppo_sb3_model = PPO("MlpPolicy", custom_env, verbose=1, device=device)

# 训练PPO模型
ppo_sb3_model.learn(total_timesteps=100000)

print("Training completed.")
