import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# 设置画布大小
fig, ax = plt.subplots(figsize=(6, 6))

# 设置方块初始位置
start_pos = (0, 0)  # 起始位置
square1 = plt.Rectangle(start_pos, 1, 1, color="red", animated=True)  # 红色方块

# 将方块添加到图中
ax.add_patch(square1)

# 设置坐标轴限制
ax.set_xlim(0, 5)  # 5x5 网格
ax.set_ylim(0, 5)

# 定义方块的世界线路径
path_1_square1 = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3)]  # 红色方块路径1
path_2_square2 = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]  # 蓝色方块路径2

# 创建保存图像的文件夹
output_dir = "data/games"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 更新方块位置的函数
def update(frame):
    # 获取当前帧的路径位置
    pos1_path1 = path_1_square1[frame]
    pos2_path2 = path_2_square2[frame]

    # 更新方块的位置，路径1
    square1.set_xy((pos1_path1[0], pos1_path1[1]))  # 红色方块路径1

    # 保存当前帧图像，路径1
    plt.savefig(f"{output_dir}/frame_{frame}_path1.png")

    # 更新方块的位置，路径2
    square1.set_xy((pos2_path2[0], pos2_path2[1]))  # 红色方块路径2

    # 保存当前帧图像，路径2
    plt.savefig(f"{output_dir}/frame_{frame}_path2.png")

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=min(len(path_1_square1), len(path_2_square2)), interval=500)

# 显示动画
plt.show()

input('11111')
