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
    动作是离散的,需要按正确顺序执行才能结束游戏。
    """
    def __init__(self, seq_length=10, frame_dim=128):
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
            # 获取最近的n个动作,其中n是目标序列长度
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
    def __init__(self, frame_dim=128, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.hidden_size = hidden_dim
        self.lstm = nn.LSTMCell(frame_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)  # 输出选择概率
    
    def forward(self, x, hidden):
        """
        x: [batch_size, frame_dim]
        hidden: Tuple of (hx, cx) each of shape [batch_size, hidden_dim]
        """
        hx, cx = self.lstm(x, hidden)
        logits = self.fc(hx)  # [batch_size, 1]
        probs = torch.sigmoid(logits)  # [batch_size, 1]
        return probs, (hx, cx)

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
    - gamma: 折扣因子（由于奖励仅在回合结束时给予，gamma=1.0）
    """
    for episode in range(num_episodes):
        observation,_ = env.reset()
        log_probs = []
        selected_actions = []
        done = False
        total_reward = 0
        # 初始化隐藏状态
        hx = torch.zeros(1, policy_net.hidden_size)
        cx = torch.zeros(1, policy_net.hidden_size)
        hidden = (hx, cx)
        while not done:
            # 将观察转换为张量
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)  # [1, frame_dim]
            # 获取动作概率
            probs, hidden = policy_net(obs_tensor, hidden)
            prob = probs.squeeze(0)  # [1]
            # 采样动作
            m = distributions.Bernoulli(prob)
            action = m.sample()
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)
            selected_actions.append(action.item())
            # 执行动作
            observation, reward, done, info = env.step(action.item())
        # 获取回合总奖励
        total_reward = reward  # 标量
        # 计算策略损失
        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * total_reward)
        policy_loss = torch.stack(policy_loss).sum()
        # 更新策略网络
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        # 日志记录
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}\tReward: {total_reward}\tSelected: {selected_actions}")
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
    success = 0
    for episode in range(num_episodes):
        observation,_ = env.reset()
        selected_actions = []
        done = False
        # 初始化隐藏状态
        hx = torch.zeros(1, policy_net.hidden_size)
        cx = torch.zeros(1, policy_net.hidden_size)
        hidden = (hx, cx)
        while not done:
            # 将观察转换为张量
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)  # [1, frame_dim]
            # 获取动作概率
            probs, hidden = policy_net(obs_tensor, hidden)
            prob = probs.squeeze(0)  # [1]
            # 选择动作（贪婪策略）
            action = (prob > 0.5).float()
            selected_actions.append(action.item())
            # 执行动作
            observation, reward, done, info = env.step(action.item())
        # 检查是否成功
        if (info['true_positives'] == env.num_key_frames) and (info['false_positives'] == 0):
            success +=1
        print(f"Episode {episode+1}\tReward: {reward}\tSelected: {selected_actions}\tKey Frames: {info['key_frames']}")
    print(f"Success rate: {success}/{num_episodes}")

# ------------------------------
# 5. 主函数
# ------------------------------
if __name__ == "__main__":
    # 设置随机种子以确保可复现性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 初始化环境和策略网络
    env = KeyFrameEnv(seq_length=10, frame_dim=128)
    policy_net = PolicyNetwork(frame_dim=128, hidden_dim=64)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    
    # 训练策略网络
    print("开始训练策略网络...")
    train_policy_gradient(env, policy_net, optimizer, num_episodes=10000, gamma=1.0)
    
    # 评估策略网络
    print("\n评估训练好的策略网络...")
    evaluate_policy(env, policy_net, num_episodes=20)