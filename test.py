import matplotlib.pyplot as plt
import numpy as np

# Cấu hình dữ liệu cho 24 Frames
num_frames = 24
frames = np.arange(1, num_frames + 1)

# Tạo dữ liệu giả lập 
attention_weights = np.random.uniform(0.005, 0.015, num_frames)

peak_indices = range(9, 16)
peak_values = [0.06, 0.10, 0.14, 0.165, 0.13, 0.09, 0.04]

for i, idx in enumerate(peak_indices):
    if idx < num_frames:
        attention_weights[idx] = peak_values[i]

# Thiết lập màu sắc
colors = []
edge_colors = []
threshold = 0.04

for val in attention_weights:
    if val > threshold:
        # Màu đỏ nhạt cho vùng bạo lực
        colors.append('#ffadad')      
        edge_colors.append('#ff4d4d')
    else:
        # Màu xanh nhạt cho vùng bình thường
        colors.append('#b3d9f2')      
        edge_colors.append('#4070a0')

# Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(12, 5))

# Vẽ cột
bars = ax.bar(frames, attention_weights, color=colors, edgecolor=edge_colors, width=0.8, linewidth=1)

# Vẽ đường nối đỉnh
ax.plot(frames, attention_weights, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

ax.set_title('Visualization of Attention Weights over 24 Frames', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Time Step ($t$)', fontsize=12)
ax.set_ylabel(r'Importance Weight ($\alpha_t$)', fontsize=12)

# Thiết lập giới hạn trục
ax.set_xlim(0, 25)
ax.set_ylim(0, 0.18)

# Lưới ngang
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

max_val = np.max(attention_weights)
max_idx = np.argmax(attention_weights) + 1

ax.text(max_idx, max_val + 0.005, 
        'Violence Occurs\n(High Importance)', 
        horizontalalignment='center', 
        verticalalignment='bottom', 
        color='#800000',
        fontsize=10, 
        fontweight='bold')

from matplotlib.patches import Patch
legend_element = [Patch(facecolor='#b3d9f2', edgecolor='#4070a0', label=r'Attention Score ($\alpha_t$)')]
ax.legend(handles=legend_element, loc='upper right')

plt.tight_layout()
plt.show()