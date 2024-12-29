import os
import torch
from torch import nn
from typing import Optional, List

def save_frames_as_gif(frames, gif_path, duration=200):
    """
    使用 PIL 将给定列表的帧保存为 gif。
    frames: List[ PIL.Image ], 存储多个帧的列表
    gif_path: str, 输出 gif 文件路径
    duration: int, 两帧之间的显示时间（毫秒）
    """
    if len(frames) == 0:
        print("No frames to save.")
        return
    # frames[0].save 是 PIL 的语法，指定 save_all=True 并 append_images 其他帧
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # 0 表示无限循环播放
    )
    print(f"已保存 GIF 到 {gif_path}")







def generate_comparison_html(input_png_path, z0_gif_path, z1_gif_path, html_output_path="compare.html"):
    """
    生成一个 HTML，内含三列：左边输入帧，中间z=0结果GIF，右边z=1结果GIF。
    """
    # 这里写了一段简易的表格HTML，让三张图并排放
    html_content = f'''
<html>
<head>
    <meta charset="utf-8" />
    <title>Diffusion Results Comparison</title>
</head>
<body>
    <table style="margin: auto; text-align: center; border-spacing: 20px;">
        <tr>
            <td>
                <h3>输入图片</h3>
                <img src="{input_png_path}" style="border:1px solid #ccc;" />
            </td>
            <td>
                <h3>z=0 预测未来视频1</h3>
                <img src="{z0_gif_path}" style="border:1px solid #ccc;" />
            </td>
            <td>
                <h3>z=1 预测未来视频2</h3>
                <img src="{z1_gif_path}" style="border:1px solid #ccc;" />
            </td>
        </tr>
    </table>
</body>
</html>
'''
    with open(html_output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"已生成 HTML 文件: {html_output_path}")
    print("请用浏览器打开这个文件，即可查看三张图并排，以及中间、右边的 GIF 正常播放。")








# 定义辅助函数
def patch_start_mask_from_entropy_with_monotonicity(entropies, t):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    """
    bs, seq_len = entropies.shape
    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = differences > t

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask


def patch_start_mask_global_and_monotonicity(entropies, t, t_add=0):
    """
    entropies: [bs, seq_len] torch tensor of entropies
    t: threshold
    returns [bs, seq_len] mask where True indicates the start of a patch
    """
    bs, seq_len = entropies.shape
    mask = torch.zeros_like(entropies, dtype=torch.bool)
    mask[:, 0] = True

    # Calculate differences between consecutive elements along the sequence length
    differences = entropies[:, 1:] - entropies[:, :-1]

    # Calculate conditions for all elements except the first one in each sequence
    condition = (differences > t_add) & (entropies[:, 1:] > t) & (~mask[:, :-1])

    # Update the mask based on the condition
    mask[:, 1:] = condition

    return mask


def patch_start_ids_from_patch_start_mask(patch_start_mask):
    bs, trunc_seq_len = patch_start_mask.shape
    max_patches = patch_start_mask.sum(dim=1).max()
    if max_patches == 0:
        patch_start_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
    else:
        patch_ids = (
            torch.arange(trunc_seq_len, device=patch_start_mask.device)
            .unsqueeze(0)
            .repeat(bs, 1)
        )
        extra_patch_ids = torch.full(
            (bs, trunc_seq_len),
            trunc_seq_len,
            dtype=torch.long,
            device=patch_start_mask.device,
        )
        all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
        patch_start_mask_padded = torch.cat(
            (patch_start_mask, ~patch_start_mask), dim=1
        )
        patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(
            bs, trunc_seq_len
        )[:, :max_patches]
    return patch_start_ids

def find_entropy_patch_start_ids(
    entropies,
    patch_size=None,
    threshold=None,
    threshold_add=None,
    monotonicity=False,
    include_next_token=True,
):
    """
    Use entropies to find the start ids of each patch.
    Use patch_size or threshold to figure out the total number of patches to allocate.

    When threshold is not None the number of patches is not constant between
    different sequences, but patches can be identified incrementally rather than
    decided globally using the entire sequence.
    """
    bs, seq_len = entropies.shape[:2]

    # Initialize first patch start ids
    first_ids = (
        torch.tensor([0, 1], dtype=torch.long, device=entropies.device)
        .unsqueeze(0)
        .repeat(bs, 1)
    )
    preds_truncation_len = first_ids.shape[1]  # remove the first preds because they will be start of patches.
    entropies = entropies[:, 1:]
    
    if threshold is None and patch_size is not None:
        num_patches = seq_len // patch_size
        patch_start_ids = entropies.topk(num_patches - 2, dim=1).indices
        patch_start_ids = patch_start_ids.sort(dim=1).values
    elif threshold is not None:
        # Assumes that there is at least one token going over the threshold
        if monotonicity:
            patch_start_mask = patch_start_mask_from_entropy_with_monotonicity(
                entropies, threshold
            )
        elif threshold_add is not None:
            patch_start_mask = patch_start_mask_global_and_monotonicity(
                entropies, threshold, threshold_add
            )
        else:
            patch_start_mask = entropies > threshold
        if not include_next_token:
            patch_start_mask = patch_start_mask[:, :-1]
        # Extract patch start ids from mask
        patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)
    else:
        raise ValueError("Either patch_size or threshold must be provided.")

    # Combine first_ids with patch_start_ids
    patch_start_ids = torch.cat(
        (first_ids, patch_start_ids + preds_truncation_len), dim=1
    )
    return patch_start_ids



def _compute_sliding_entropy_incremental(
        sequence: torch.Tensor, 
        window_size: int, 
        vocab_size: int,
        ignore_bos_eos: bool = True
    ) -> torch.Tensor:
        """
        对单条序列 (长度 n) 进行增量式滑动窗口熵计算，返回每个位置的局部熵。
        sequence: (n,)
        返回: shape (n,) 的张量 entropies。

        ignore_bos_eos=True 时，会将 BOS(256) / EOS(257) 的出现次数视为 0，不计入熵。
        """
        n = sequence.size(0)
        entropies = torch.zeros(n, device=sequence.device, dtype=torch.float)
        
        # 计数器: 每个符号出现多少次
        counts = torch.zeros(vocab_size, device=sequence.device, dtype=torch.float)
        
        # 辅助函数: 计算当前窗口的熵
        def calc_entropy():
            # 如果设置 ignore_bos_eos，需要把 BOS/EOS 的计数置0
            c = counts.clone()
            if ignore_bos_eos:
                c[256] = 0.0  # BOS
                c[257] = 0.0  # EOS
            
            total = c.sum()
            if total < 1e-12:
                return 0.0
            
            probs = c / total
            # 避免 log(0)
            probs = torch.clamp(probs, min=1e-12)
            H = -torch.sum(probs * torch.log2(probs))
            return H.item()
        
        # 1) 先把前 window_size 个符号放进计数器，如果不够 n，就取 min
        init_win = min(window_size, n)
        for i in range(init_win):
            counts[sequence[i]] += 1
        
        # 2) 第 0 个位置对应 [0..window_size-1] 的窗口熵
        entropies[0] = calc_entropy()
        
        # 3) 滑动窗口
        for i in range(1, n):
            start_idx = i
            end_idx = i + window_size - 1
            # 移除旧的
            old_symbol = sequence[i - 1]
            counts[old_symbol] -= 1
            
            if end_idx < n:
                # 窗口还够
                new_symbol = sequence[end_idx]
                counts[new_symbol] += 1
            else:
                # 不够则做简单处理: 用最后一个符号重复填充
                # 也可以选择 “窗口变小” 等其他策略
                new_symbol = sequence[-1]
                counts[new_symbol] += 1
            
            # 当前窗口熵
            entropies[i] = calc_entropy()
        
        return entropies