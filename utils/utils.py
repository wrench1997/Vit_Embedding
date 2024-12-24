import os


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
