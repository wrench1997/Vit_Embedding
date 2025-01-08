import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

# ------------------------------
# 1. 自定义环境定义
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
# 2. 策略网络定义
# ------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, frame_dim=128, hidden_dim=128, num_layers=2, bidirectional=True):
        super(PolicyNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # 多层双向 LSTM
        self.lstm = nn.LSTM(
            input_size=frame_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 全连接层，输出动作概率
        self.fc = nn.Linear(hidden_dim * self.num_directions, 4)  # 4个动作
        
        # Softmax 层，用于多分类
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, hidden=None):
        """
        x: [batch_size, seq_length, frame_dim]
        hidden: Tuple of (h0, c0)
        """
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: [batch, seq_len, hidden_dim * num_directions]
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim * num_directions]
        logits = self.fc(last_output)  # [batch, 4]
        probs = self.softmax(logits)  # [batch, 4]
        return probs, hidden

# ------------------------------
# 3. 训练循环
# ------------------------------
def train_policy_gradient(env, policy_net, optimizer, num_episodes=1000, gamma=1.0):
    """
    使用 REINFORCE 算法训练策略网络。
    
    参数:
    - env: 自定义环境
    - policy_net: 策略网络
    - optimizer: 优化器
    - num_episodes: 训练回合数
    - gamma: 折扣因子
    """
    policy_net.train()
    for episode in range(1, num_episodes + 1):
        observation, _ = env.reset()
        done = False
        rewards = []
        log_probs = []
        actions = []
        state = observation[np.newaxis, :]  # [1, frame_dim]
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # [1, 1, frame_dim]
            # 前向传播
            probs, _ = policy_net(state_tensor)  # probs: [1, 4]
            m = distributions.Categorical(probs)
            action = m.sample()  # [1]
            log_prob = m.log_prob(action)  # [1]
            log_probs.append(log_prob)
            actions.append(action.item())
            # 环境交互
            observation, reward, done, info = env.step(action.item())
            rewards.append(reward)
            state = observation[np.newaxis, :]  # [1, frame_dim]
        
        # 计算累计折扣奖励
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        # 标准化 returns
        if returns.std() != 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns = returns - returns.mean()
        
        # 计算损失
        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)
        policy_loss = torch.cat(policy_loss).sum()
        
        # 优化策略网络
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # 日志记录
        if episode % 100 == 0:
            success = info.get('success', False)
            print(f"Episode {episode}\tReward: {info.get('success', False)}\tActions: {actions}\tSuccess: {success}")
    print("Training completed.")

# ------------------------------
# 4. 评估函数
# ------------------------------
def evaluate_policy(env, policy_net, num_episodes=10):
    """
    评估训练好的策略网络。
    
    参数:
    - env: 自定义环境
    - policy_net: 策略网络
    - num_episodes: 评估回合数
    """
    policy_net.eval()
    success = 0
    with torch.no_grad():
        for episode in range(1, num_episodes + 1):
            observation, _ = env.reset()
            done = False
            actions = []
            state = observation[np.newaxis, :]  # [1, frame_dim]
            while not done:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)  # [1, 1, frame_dim]
                probs, _ = policy_net(state_tensor)  # probs: [1, 4]
                action = torch.argmax(probs, dim=1).item()
                actions.append(action)
                # 环境交互
                observation, reward, done, info = env.step(action)
                state = observation[np.newaxis, :]  # [1, frame_dim]
            # 检查是否成功
            if info.get('success', False):
                success += 1
            print(f"Episode {episode}\tActions: {actions}\tSuccess: {info.get('success', False)}")
    print(f"Success rate: {success}/{num_episodes}")

# ------------------------------
# 5. 主函数
# ------------------------------
if __name__ == "__main__":
    # 设置随机种子以确保可复现性
    np.random.seed(42)
    torch.manual_seed(42)

    # 初始化环境和策略网络
    env = KeyFrameEnv(seq_length=4, frame_dim=128)
    policy_net = PolicyNetwork(frame_dim=128, hidden_dim=128, num_layers=2, bidirectional=True)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    # 训练策略网络
    print("开始训练策略网络...")
    train_policy_gradient(env, policy_net, optimizer, num_episodes=10000, gamma=0.99)

    # 评估策略网络
    print("\n评估训练好的策略网络...")
    evaluate_policy(env, policy_net, num_episodes=20)