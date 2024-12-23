import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import argparse

# ============================
# 1. 环境: SimpleGridEnv
# 只在 episode 结束时给终止奖励(1或0)，中间步无reward
# ============================
class SimpleGridEnv(gym.Env):
    """
    - 离散动作: 0=上,1=下,2=左,3=右
    - Observation: (3, 16, 16) 的图像, [0,1]之间
    - 只有当智能体到达目标时, 终止且 reward=1, 否则超时结束reward=0
    - 这里将网格缩小到16x16, 方便演示
    """
    def __init__(self, max_steps=50):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3,16,16), dtype=np.float32
        )
        self.max_steps = max_steps
        self.grid_size = 16

        self.reset()

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # 随机放置agent和目标
        self.agent_pos = self._sample_pos()
        while True:
            self.target_pos = self._sample_pos()
            if self.target_pos != self.agent_pos:
                break
        obs = self._get_obs()
        info = {}
        return (obs, info) if return_info else obs

    def step(self, action):
        self.step_count += 1

        # 移动
        x, y = self.agent_pos
        if action == 0: # up
            x = max(0, x-1)
        elif action == 1: # down
            x = min(self.grid_size-1, x+1)
        elif action == 2: # left
            y = max(0, y-1)
        elif action == 3: # right
            y = min(self.grid_size-1, y+1)
        self.agent_pos = (x,y)

        # 判断是否到达目标
        reward = 0.0
        terminated = False
        truncated = False
        if self.agent_pos == self.target_pos:
            terminated = True
            reward = 1.0
        elif self.step_count >= self.max_steps:
            truncated = True
            reward = 0.0

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # 3通道, 16x16
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        ax, ay = self.agent_pos
        tx, ty = self.target_pos
        # agent: 用白色(1,1,1)
        obs[0, ax, ay] = 1.0
        obs[1, ax, ay] = 1.0
        obs[2, ax, ay] = 1.0
        # target: 用红色(1,0,0)
        obs[0, tx, ty] = 1.0
        obs[1, tx, ty] = 0.0
        obs[2, tx, ty] = 0.0
        return obs

    def _sample_pos(self):
        x = np.random.randint(0, self.grid_size)
        y = np.random.randint(0, self.grid_size)
        return (x,y)


# ============================
# 2. 序列奖励模型: 输入一整条轨迹 (T,3,H,W), 输出每时刻的reward shape=(T,)
# 目标: sum_{t=1 to T}(r_t) ~= final_reward
# ============================
class SequenceRewardModel(nn.Module):
    def __init__(self, c=3, h=16, w=16, hidden_size=128):
        super().__init__()
        self.c, self.h, self.w = c, h, w

        # CNN特征
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # 16x16 -> stride=2 -> 8x8 -> stride=2-> 4x4
        # => conv输出 64*(4*4)=64*16=1024
        feature_dim = 1024

        # 用一个简单的GRU (或LSTM都行)
        self.rnn = nn.GRU(feature_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, obs_seq):
        """
        obs_seq: (B, T, c, h, w)
        return: (B, T) step-wise reward
        """
        B, T, c, h, w = obs_seq.shape
        # (B*T, c, h, w)
        flatten_in = obs_seq.view(B*T, c, h, w)
        feat = self.conv(flatten_in)  # (B*T, feature_dim)
        feat_dim = feat.shape[-1]
        feat = feat.view(B, T, feat_dim)  # (B, T, feature_dim)

        out, _ = self.rnn(feat)  # (B, T, hidden_size)
        # fc => (B, T, 1)
        rewards = self.fc(out).squeeze(-1)  # (B, T)
        return rewards

def train_seq_reward_model(model, optimizer, obs_seq_batch, final_rewards_batch, device):
    """
    obs_seq_batch: (batch, T, 3, 16, 16)
    final_rewards_batch: (batch,)  # 轨迹最终奖励(0 or 1)
    => sum_{t=1 to T} predicted_r_t ~= final_reward
    """
    model.train()
    obs_seq_batch = obs_seq_batch.to(device)
    final_rewards_batch = final_rewards_batch.to(device)

    pred_step_rewards = model(obs_seq_batch)  # (batch, T)
    pred_final = pred_step_rewards.sum(dim=1) # (batch,)
    loss = F.mse_loss(pred_final, final_rewards_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ============================
# 3. Policy (Actor-Critic) + PPO
# ============================
class PolicyNet(nn.Module):
    def __init__(self, c=3, h=16, w=16, num_actions=4, hidden_size=128):
        super().__init__()
        # CNN提特征
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # conv输出: 64*(4*4)=1024
        feature_dim = 1024
        self.fc = nn.Linear(feature_dim, hidden_size)
        self.relu = nn.ReLU()

        # 策略头
        self.policy_head = nn.Linear(hidden_size, num_actions)
        # 价值头
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: (batch, 3, 16, 16)
        return: logits, value
        """
        feat = self.conv(x)
        feat = self.relu(self.fc(feat))
        logits = self.policy_head(feat)
        value = self.value_head(feat).squeeze(-1)
        return logits, value

    def select_action(self, obs, device="cpu"):
        with torch.no_grad():
            logits, value = self.forward(obs.to(device))
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()


class PPOAgent:
    def __init__(self, policy_net, optimizer, device, clip_epsilon=0.2, gamma=0.99, gae_lambda=0.95):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.reset_storage()

    def reset_storage(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def store_transition(self, obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def compute_gae(self, next_value=0.0):
        """
        计算returns, advantages
        self.values要多append一个next_value
        """
        values = self.values + [next_value]
        rewards = self.rewards
        dones = self.dones

        gae = 0.0
        returns = []
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma*values[step+1]*(1 - dones[step]) - values[step]
            gae = delta + self.gamma*self.gae_lambda*(1 - dones[step])*gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        return returns, advantages

    def update(self, returns, advantages, old_log_probs, epochs=4, batch_size=32):
        obs_tensor = torch.stack([torch.FloatTensor(o) for o in self.obs]).to(self.device)  # (T, 3,16,16)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long, device=self.device)  # (T,)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float, device=self.device)

        advantages_tensor = (advantages_tensor - advantages_tensor.mean())/(advantages_tensor.std()+1e-8)

        dataset = torch.utils.data.TensorDataset(obs_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for ob, ac, old_lp, ret, adv in loader:
                logits, values = self.policy_net(ob)
                dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
                new_lp = dist.log_prob(ac)

                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio*adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)*adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, ret)
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5*value_loss - 0.01*entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optimizer.step()


# ============================
# 4. 主逻辑
# - 先收集一批episode(只拿到最终reward=0或1)
# - 训练SequenceRewardModel, 让 sum of predicted step rewards ~= final_reward
# - 用训练好的model对轨迹做 step-wise reward => PPO更新
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=200, help='Training iterations')
    parser.add_argument('--episodes_per_iter', type=int, default=10, help='Num of episodes to collect each iteration')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    env = SimpleGridEnv(max_steps=50)

    # 初始化 SequenceRewardModel
    seq_reward_model = SequenceRewardModel().to(device)
    seq_reward_optimizer = optim.Adam(seq_reward_model.parameters(), lr=1e-4)

    # 初始化 Policy + PPO
    policy_net = PolicyNet().to(device)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    ppo_agent = PPOAgent(policy_net, policy_optimizer, device)

    # ============ 训练循环 ============
    for it in range(1, args.iterations+1):
        # 1) 收集轨迹(episode)
        obs_seq_list = []
        final_rewards_list = []
        # also store transitions for PPO
        ppo_agent.reset_storage()

        for _ in range(args.episodes_per_iter):
            obs = env.reset()
            done = False
            truncated = False
            ep_obs = []
            step_count = 0

            while not (done or truncated):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)  # (1,3,16,16)
                action, logp, val = ppo_agent.policy_net.select_action(obs_t, device)
                obs_next, reward, done, truncated, _ = env.step(action)

                ppo_agent.store_transition(
                    obs, 
                    action,
                    logp,
                    val,
                    0.0,  # 暂时先把环境中步进reward设为0; 最终会用 seq model
                    float(done or truncated)
                )
                ep_obs.append(obs)
                obs = obs_next
                step_count+=1

            # episode结束后, final reward=1(成功) or 0(超时)
            final_r = 1.0 if done else 0.0
            # 将episode存储到 "训练sequence model" 的数据
            # => shape (T, 3,16,16)
            ep_obs_arr = np.array(ep_obs, dtype=np.float32)
            obs_seq_list.append(ep_obs_arr)
            final_rewards_list.append(final_r)

        # 2) 训练SequenceRewardModel: sum of step-wise = final reward
        #    将多条episode打包
        max_len = max([arr.shape[0] for arr in obs_seq_list])
        # pad sequences to same length if needed, or just batch them with dataloader
        # 这里做简单: 取subset. (真实实现需更完善)
        # 仅演示思路:
        batch_obs_seq = []
        batch_final_rewards = []
        for (ep_obs_arr, ep_r) in zip(obs_seq_list, final_rewards_list):
            # shape = (T,3,16,16)
            batch_obs_seq.append(ep_obs_arr)
            batch_final_rewards.append(ep_r)

        # 拼接 => (B, T, 3,16,16)
        # 需要对不等长做pad, 这里示例就假设都差不多(episode长度类似)
        # or take a subset min_len, etc.
        # simplified: take partial or full
        min_len = min([arr.shape[0] for arr in batch_obs_seq])
        # 截断到min_len
        batch_obs_seq = [arr[:min_len] for arr in batch_obs_seq]
        # => (B, T, c, h, w)
        obs_seq_tensor = torch.stack([torch.from_numpy(arr) for arr in batch_obs_seq], dim=0)
        final_rewards_tensor = torch.tensor(batch_final_rewards, dtype=torch.float)

        seq_loss = train_seq_reward_model(seq_reward_model, seq_reward_optimizer, obs_seq_tensor, final_rewards_tensor, device)

        # 3) 用训练好的 seq_reward_model 给 PPO 的 transitions 计算 step-wise reward
        #    ppo_agent里储存了( obs, action, logp, value, reward=0.0, done ) => 这里重新赋值
        with torch.no_grad():
            # 先把每个episode里的obs做 forward => (T,) reward
            # 需要把 transitions 拆分成episodes
            # 这里假设 episodes_per_iter=1 for simplicity, 真实要按episode done拆分
            # -----------
            # 演示: 直接对agent.obs做一次批量inference, 见下
            # (N, 3,16,16)
            all_obs_t = torch.stack([torch.FloatTensor(o) for o in ppo_agent.obs]).to(device)
            # 需要知道episode边界. 这里简单假设只有1个episode => step wise
            # 真实情况需要逐episode处理
            rewards_pred = seq_reward_model(
                all_obs_t.unsqueeze(0)
            )  # => shape (1, T)
            rewards_pred = rewards_pred.squeeze(0).cpu().numpy()

        # 4) 替换 PPO 中的 reward
        ppo_agent.rewards = rewards_pred.tolist()

        # 5) 计算GAE => PPO update
        next_value = 0.0  # assume terminal
        returns, advantages = ppo_agent.compute_gae(next_value=next_value)
        old_log_probs = ppo_agent.log_probs.copy()
        ppo_agent.update(returns, advantages, old_log_probs)

        # reset for next iteration
        ppo_agent.reset_storage()

        if it % 10 == 0:
            print(f"Iter={it}, SeqReward Loss={seq_loss:.4f}")

    print("Done training.")


if __name__=="__main__":
    main()
