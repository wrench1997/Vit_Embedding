import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

###############################################
# 1. FrameStackWrapper
###############################################
class FrameStackWrapper(gym.Wrapper):
    """
    在环境层面堆叠若干历史帧，使得在 step() 时返回多帧合并后的观测。
    可以灵活选择最简单的(channel 拼接)返回方式，也可保持 shape=(stack, c, h, w)。
    """
    def __init__(self, env, num_stack=4, stack_axis="channel"):
        """
        :param env: gym.Env
        :param num_stack: 堆叠的帧数
        :param stack_axis: "channel" 表示把帧拼到通道维度(C变成 C*num_stack)；
                           "stack" 表示保留stack维度 => shape (num_stack, C, H, W)
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.stack_axis = stack_axis
        self.frames = deque([], maxlen=num_stack)

        # 修改观察空间 (只在 channel 方式下能轻松确定 shape)
        if isinstance(self.observation_space, gym.spaces.Box):
            c, h, w = self.observation_space.shape
            if stack_axis=="channel":
                new_shape = (c * num_stack, h, w)
            else:
                new_shape = (num_stack, c, h, w)
            self.observation_space = gym.spaces.Box(
                low=0, high=1.0, shape=new_shape, dtype=np.float32
            )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, truncated, info

    def _get_ob(self):
        """
        返回多帧拼接后的观测
        """
        if self.stack_axis == "channel":
            # shape: (num_stack, c, h, w) => (c*num_stack, h, w)
            return np.concatenate(self.frames, axis=0)  # frames已是 (c,h,w) 的list
        else:
            # shape: (num_stack, c, h, w)
            return np.stack(self.frames, axis=0)


###############################################
# 2. MultiFrameEmbedder
###############################################
class MultiFrameEmbedder(nn.Module):
    """
    用来将“多帧观测” => “embedding向量”。
    可配置:
      - method="stack": 将多帧按channel堆叠后，用2D CNN或MLP等处理
      - method="sequential": 对每帧独立encode，再用RNN/Attention组合
      - 或你可以灵活扩展
    """
    def __init__(self, input_shape, embed_dim=128, method="stack", use_rnn=False):
        """
        :param input_shape: 多帧数据的shape, 例如:
               if stack_axis="channel", shape=(c*num_stack, h, w)
               if stack_axis="stack", shape=(num_stack, c, h, w)
        :param embed_dim: 输出embedding的维度
        :param method: "stack" 或 "sequential"
        :param use_rnn: 在 sequential 时, 是否用RNN(=True) 或简单MLP(=False)
        """
        super().__init__()
        self.method = method
        self.use_rnn = use_rnn
        self.embed_dim = embed_dim

        # 分析 input_shape
        if self.method=="stack":
            # e.g. (c*num_stack, h, w)
            c, h, w = input_shape
            self.encoder = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2,2)),
                nn.Flatten(),
            )
            # 64*(2*2)=256 => linear => embed_dim
            self.fc = nn.Linear(256, embed_dim)
        else:
            # method=="sequential"
            # e.g. input_shape=(num_stack, c, h, w)
            stack, c, h, w = input_shape
            self.cnn_per_frame = nn.Sequential(
                nn.Conv2d(c, 32, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2,2)),
                nn.Flatten(),
            )
            # single frame output dim => 64*(2*2)=256
            self.per_frame_outdim = 256
            if use_rnn:
                self.rnn = nn.GRU(self.per_frame_outdim, embed_dim, batch_first=True)
            else:
                # simpler: average all frames or MLP combine
                self.fc = nn.Sequential(
                    nn.Linear(self.per_frame_outdim, embed_dim),
                    nn.ReLU(),
                )
                # 这里会在 forward 里再pooling/平均

    def forward(self, obs_batch):
        """
        obs_batch: shape=(B, c*num_stack, h, w) if method="stack"
                   or shape=(B, num_stack, c, h, w) if method="sequential"
        return: embedding shape => (B, embed_dim)
        """
        if self.method=="stack":
            # 1) 2D CNN
            feat = self.encoder(obs_batch)  # (B, 256)
            emb = self.fc(feat)             # (B, embed_dim)
            return emb

        else:
            # method="sequential"
            B = obs_batch.shape[0]
            # obs_batch => (B, stack, c, h, w)
            # reshape => (B*stack, c,h,w)
            # run each frame => 256-dim
            stack, c, h, w = obs_batch.shape[1], obs_batch.shape[2], obs_batch.shape[3], obs_batch.shape[4]
            x = obs_batch.view(B*stack, c, h, w)
            frame_feat = self.cnn_per_frame(x)  # => (B*stack, 256)

            # reshape =>(B, stack, 256)
            frame_feat = frame_feat.view(B, stack, -1)

            if self.use_rnn:
                # RNN => take last hidden as embedding (or sum/mean?)
                out, _ = self.rnn(frame_feat)  # (B, stack, embed_dim)
                emb = out[:, -1, :]           # 取最后时刻
            else:
                # simpler: average all frames then linear
                # or just fc per frame + average
                # 这里演示: fc后平均
                # fc =>(B*stack, embed_dim)
                out2 = self.fc(frame_feat.view(B*stack, -1)) # =>(B*stack, embed_dim)
                out2 = out2.view(B, stack, self.embed_dim)   # =>(B, stack, embed_dim)
                emb = out2.mean(dim=1)                       # =>(B, embed_dim)

            return emb


###############################################
# 3. EmbeddingPolicy (一个简单的Actor-Critic Policy)
###############################################
class EmbeddingPolicy(nn.Module):
    """
    接受embedding向量 => 输出动作分布 + state value
    """
    def __init__(self, embed_dim, hidden_size=128, num_actions=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, emb):
        """
        emb: (B, embed_dim)
        return: (logits, value)
        """
        feat = self.mlp(emb)
        logits = self.policy_head(feat)          # (B, num_actions)
        value = self.value_head(feat).squeeze(-1)# (B,)
        return logits, value

    def select_action(self, emb_single):
        """
        对单条数据 => (embed_dim,) => forward => action
        """
        self.eval()
        with torch.no_grad():
            emb_input = emb_single.unsqueeze(0)   # => (1, embed_dim)
            logits, value = self.forward(emb_input)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action.item(), logp.item(), value.item()


###############################################
# 4. main_demo
###############################################
def main_demo():
    # =========== 1) 创建环境并包上多帧堆叠 ===========
    env = gym.make("CartPole-v1", render_mode=None)  # 仅示例, shape=(4,) 而非图像
    # 如果是图像环境, shape=(c,h,w) => frame stack => shape=(num_stack*c,h,w) or (num_stack,c,h,w)
    # 这里只是演示:
    env = FrameStackWrapper(env, num_stack=4, stack_axis="channel")
    obs = env.reset()

    # 现在 obs.shape 可能是(4,) 变成 ??? => CartPole只是示例, 这里真正要在Atari等图像env中才更明显

    # =========== 2) 创建多帧embedding模块 ===========
    # 假设stack_axis="channel" => input_shape=(c*num_stack, h, w)
    # 这里CartPole并非图像, 所以只是演示形状. 假设 (12,84,84) for example
    # method="stack":  (B, c*num_stack, h,w) => 2D CNN
    # or method="sequential":(B, num_stack, c, h,w)=> RNN
    input_shape = (12, 84, 84)  # 仅举例
    embed_dim = 128

    embedder = MultiFrameEmbedder(input_shape, embed_dim=embed_dim,
                                  method="stack", use_rnn=False)

    # =========== 3) 创建一个EmbeddingPolicy ===========
    policy = EmbeddingPolicy(embed_dim=embed_dim, hidden_size=64, num_actions=env.action_space.n)

    # =========== 4) 交互 (伪示例) ===========
    # obs = env.reset()
    done = False
    truncated = False

    # 这里的问题是: env返回的 obs 其实 shape不一定是图像
    # 你需要一个 'obs => embed' 流程
    # let's assume we already convert obs to shape (B=1, c*num_stack, h,w)
    # 这只是演示. 实际中应在Atari/MinAtar之类才对.
    while not done and not truncated:
        # obs shape => e.g. (c*num_stack, h, w), convert => (1,c*num_stack,h,w)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)

        # => feed embedder
        emb = embedder(obs_t)  # =>(1, embed_dim)
        emb_single = emb.squeeze(0)  # =>(embed_dim,)

        # => policy
        action, logp, val = policy.select_action(emb_single)
        obs_next, rew, done, truncated, info = env.step(action)
        obs = obs_next

    env.close()
    print("Demo finished.")


if __name__ == "__main__":
    main_demo()
