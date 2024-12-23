import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import argparse

class SimpleGridEnv(gym.Env):
    """
    一个简单的64x64网格世界:
    - 状态以3通道图像表示: shape = (3,64,64)
    - Agent在64x64的场景中移动
    - 目标随机生成在与Agent不同的位置
    - 动作空间: 0=上, 1=下, 2=左, 3=右
    - 只有当Agent到达目标时才reward=1并terminated=True
    - 若超过max_steps仍未到达目标则truncated=True且reward=0
    - 中间步骤reward=0
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, max_steps=200, render_mode=None):
        super(SimpleGridEnv, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,64,64), dtype=np.float32
        )
        
        self.max_steps = max_steps
        self.step_count = 0

        self.grid_size = 64
        self.agent_pos = None
        self.target_pos = None
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        # 随机生成agent和target位置
        self.agent_pos = self._sample_position()
        while True:
            self.target_pos = self._sample_position()
            if self.target_pos != self.agent_pos:
                break

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        # 根据动作移动agent
        ax, ay = self.agent_pos
        if action == 0:   # 上
            ax = max(0, ax-1)
        elif action == 1: # 下
            ax = min(self.grid_size-1, ax+1)
        elif action == 2: # 左
            ay = max(0, ay-1)
        elif action == 3: # 右
            ay = min(self.grid_size-1, ay+1)
        
        self.agent_pos = (ax, ay)

        # 检查是否到达目标
        terminated = False
        truncated = False
        reward = 0.0
        if self.agent_pos == self.target_pos:
            terminated = True
            reward = 1.0
        elif self.step_count >= self.max_steps:
            # 超过最大步数还没到达目标则截断
            truncated = True
            reward = 0.0

        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human":
            obs = self._get_obs()
            img = (obs.transpose(1,2,0)*255).astype(np.uint8)
            cv2.imshow("SimpleGridEnv", img)
            cv2.waitKey(1)
        elif self.render_mode == "rgb_array":
            obs = self._get_obs()
            img = (obs.transpose(1,2,0)*255).astype(np.uint8)
            return img

    def close(self):
        if self.render_mode == "human":
            cv2.destroyAllWindows()

    def _get_obs(self):
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)

        # Agent用白色 (1,1,1)
        ax, ay = self.agent_pos
        obs[0, ax, ay] = 1.0
        obs[1, ax, ay] = 1.0
        obs[2, ax, ay] = 1.0

        # Target用红色 (1,0,0)
        tx, ty = self.target_pos
        obs[0, tx, ty] = 1.0
        obs[1, tx, ty] = 0.0
        obs[2, tx, ty] = 0.0

        return obs

    def _sample_position(self):
        x = np.random.randint(0, self.grid_size)
        y = np.random.randint(0, self.grid_size)
        return (x, y)


def main():
    parser = argparse.ArgumentParser(description="运行SimpleGridEnv环境。")
    parser.add_argument('--render', action='store_true', help='启用环境渲染界面。')
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = SimpleGridEnv(max_steps=10000, render_mode=render_mode)
    obs, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample() # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        if args.render:
            env.render()
    env.close()


if __name__ == "__main__":
    main()
