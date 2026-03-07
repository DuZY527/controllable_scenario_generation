import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_multiscale_convolution():
    """绘制级联多尺度卷积示意图"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 依次尝试这些字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    fig, ax = plt.subplots(figsize=(14, 8))

    # 设置颜色方案
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    # 输入序列
    ax.text(0.1, 0.85, '输入序列\n(168小时)', ha='center', va='center',
            fontsize=10, fontweight='bold', transform=ax.transAxes)

    # 各层卷积示意图
    layers = [
        {'name': '3h卷积\n(局部波动)', 'kernel': 3, 'channels': 64, 'stride': 1, 'color': colors[0]},
        {'name': '24h卷积\n(日周期)', 'kernel': 24, 'channels': 128, 'stride': 2, 'color': colors[1]},
        {'name': '48h卷积\n(2日模式)', 'kernel': 48, 'channels': 256, 'stride': 2, 'color': colors[2]},
        {'name': '72h卷积\n(3日模式)', 'kernel': 72, 'channels': 512, 'stride': 2, 'color': colors[3]},
        {'name': '120h卷积\n(周中模式)', 'kernel': 120, 'channels': 1024, 'stride': 2, 'color': colors[4]}
    ]

    # 绘制卷积层
    for i, layer in enumerate(layers):
        x_pos = 0.2 + i * 0.15
        y_pos = 0.7

        # 卷积核示意图
        rect = patches.Rectangle((x_pos, y_pos), 0.08, 0.15,
                                 facecolor=layer['color'], alpha=0.7,
                                 edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        # 层信息
        ax.text(x_pos + 0.04, y_pos - 0.02,
                f"Kernel: {layer['kernel']}h\nChannels: {layer['channels']}\nStride: {layer['stride']}",
                ha='center', va='top', fontsize=8)

        # 层名称
        ax.text(x_pos + 0.04, y_pos + 0.17, layer['name'],
                ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 连接箭头
        if i > 0:
            ax.annotate('', xy=(x_pos - 0.02, y_pos + 0.075),
                        xytext=(x_pos - 0.07, y_pos + 0.075),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    # 输出层
    ax.text(0.95, 0.5, '潜在向量\n(128维)', ha='center', va='center',
            fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5',
                                                      facecolor=colors[5], alpha=0.7))

    # 添加时间尺度说明
    ax.text(0.1, 0.3, '时间尺度说明:', fontsize=11, fontweight='bold', transform=ax.transAxes)
    ax.text(0.1, 0.25, '• 3h卷积: 捕捉小时级波动、云遮挡等瞬时变化',
            fontsize=9, transform=ax.transAxes)
    ax.text(0.1, 0.22, '• 24h卷积: 捕捉完整的日周期（日出-日落模式）',
            fontsize=9, transform=ax.transAxes)
    ax.text(0.1, 0.19, '• 48h卷积: 捕捉周末效应（周六-周日模式差异）',
            fontsize=9, transform=ax.transAxes)
    ax.text(0.1, 0.16, '• 72h卷积: 捕捉周中模式（周一至周三）',
            fontsize=9, transform=ax.transAxes)
    ax.text(0.1, 0.13, '• 120h卷积: 捕捉周中工作日模式（周一至周五）',
            fontsize=9, transform=ax.transAxes)

    # 添加标题
    ax.set_title('级联多尺度卷积神经网络 (Cascaded Multi-Scale CNN)\n光伏出力序列特征提取',
                 fontsize=14, fontweight='bold', pad=20)

    # 设置坐标轴
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('multiscale_cnn_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()


# 运行绘图
plot_multiscale_convolution()