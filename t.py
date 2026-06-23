import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# pn_saliency = np.load('mat-pn1-gradcam-L1-2874pcds-500drops-50steps.npz')
# pn2_saliency = np.load('tmp/mat-exp14-pn2-saliency-2874pcds-500drops-50steps.npz')

# pn_gradients = np.load('mat-My-pn22874pcds-500drops-50steps.npz')
# pn2_gradients = np.load('tmp/mat-exp13-pn2-gradients-absolute-2874pcds-500drops-50steps.npz')

# pn_grad = np.load('tmp/mat-pn1-gradcam-L1-2874pcds-500drops-50steps.npz')
# pn2_grad = np.load('tmp/mat-his_p2-2874pcds-500drops-50steps.npz')

# pn_ours = np.load('mat-My-pn22874pcds-500drops-50steps.npz')
# pn2_ours = np.load('tmp/mat-My-pn22874pcds-500drops-50steps.npz')

# gradients = pn_gradients
# saliency = pn_saliency
# grad = pn_grad
# ours = pn_ours

# num_drops = ours['num_drops'][0]
# num_classified = ours['num_classified'][0]

# # gradients_high_correct = gradients['high_correct'] / num_classified
# # gradients_low_correct = gradients['low_correct'] / num_classified

# saliency_high_correct = saliency['high_correct'] / num_classified
# saliency_low_correct = saliency['low_correct'] / num_classified

# grad_high_correct = grad['high_correct'] / num_classified
# grad_low_correct = grad['low_correct'] / num_classified

# ours_high_correct = ours['high_correct'] / num_classified
# ours_low_correct = ours['low_correct'] / num_classified


# x = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
# # grad_high_correct = grad_low_correct
# saliency_high_correct = saliency_low_correct

# ours_high_correct = ours_low_correct


# # 创建图像
# plt.figure(figsize=(10, 6))
# grad_high_correct[0] = saliency_high_correct[0] = ours_high_correct[0]
# from matplotlib.ticker import FormatStrFormatter
# plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))



# # 绘制每条线
# # plt.plot(x, gradients_high_correct[:11], label="Gradients", color='blue', linestyle='-', marker='o')
# plt.plot(x, saliency_high_correct[:11], label="PcSM", color='green', linestyle='--', marker='s')
# plt.plot(x, grad_high_correct[:11], label="Grad-Vis", color='purple', linestyle='-.', marker='^')
# plt.plot(x, ours_high_correct[:11], label="Ours", color='red', linestyle=':', marker='d')

# # 设置图例（左下角）
# plt.legend(loc='lower left')

# # 设置标题和轴标签
# plt.title("Comparison of High Dropping", fontsize=16)
# plt.xlabel("Number of Drops", fontsize=14)
# plt.ylabel("Accuracy", fontsize=14)

# # 动态调整 Y 轴范围
# min_y = min( saliency_high_correct[:11].min(), grad_high_correct[:11].min(), ours_high_correct[:11].min())
# max_y = max( saliency_high_correct[:11].max(), grad_high_correct[:11].max(), ours_high_correct[:11].max())
# padding = 0.05 * (max_y - min_y)  # 添加 5% 的上下边距
# # plt.ylim(0.91, 0.99)
# plt.ylim(min_y - padding, max_y + padding)

# # 设置 X 轴范围
# plt.xlim(0, 500)

# # 添加网格以便于观察
# plt.grid(alpha=0.3)

# # 显示图像
# plt.tight_layout()
# plt.savefig('./plt.png', dpi=300)
# plt.show()

data = {
    "PointNet": {
        "High_Drop": {
            "PcSM": [
                0.980, 0.9795, 0.980, 0.980,
                0.980, 0.980, 0.980, 0.980,
                0.983, 0.9834, 0.9836
            ],
            "Grad-Vis": [
                0.980, 0.970, 0.948, 0.907,
                0.875, 0.847, 0.809, 0.783,
                0.754, 0.735, 0.697
            ],
            "Ours": [
                0.980, 0.970, 0.965, 0.958,
                0.908, 0.858, 0.843, 0.809,
                0.788, 0.747, 0.712
            ],
            "FBI": [
                0.980, 0.971, 0.965, 0.9584,
                0.918, 0.864, 0.854, 0.813,
                0.798, 0.750, 0.730
            ],
            "MSXAI": [
                0.980, 0.974, 0.964, 0.911,
                0.880, 0.853, 0.840, 0.80,
                0.775, 0.753, 0.731
            ],
            "IIA" : [
                0.980, 0.975, 0.973, 0.970,
                0.948, 0.943, 0.936, 0.912,
                0.885, 0.876, 0.870
            ]

        },
        "Low_Drop": {
            "PcSM": [
                0.980, 0.973, 0.973, 0.972,
                0.972, 0.971, 0.9713, 0.971,
                0.971, 0.971, 0.9717
            ],
            "Grad-Vis": [
                0.980, 0.9766, 0.9766, 0.9768,
                0.976, 0.975, 0.9754, 0.971,
                0.969, 0.965, 0.961
            ],
            "Ours": [
                0.980, 0.9769, 0.9768, 0.976,
                0.973, 0.9724, 0.9714, 0.9709,
                0.9702, 0.9693, 0.963
            ],
            "FBI": [
                0.980, 0.976, 0.9763, 0.976,
                0.9734, 0.972, 0.971, 0.970,
                0.970, 0.967, 0.962
            ],
            "MSXAI": [
                0.980, 0.9750, 0.9747, 0.9740,
                0.9710, 0.9718, 0.9698, 0.9695,
                0.9697, 0.962, 0.9596
            ],
            "IIA": [
                0.980, 0.975, 0.973, 0.971,
                0.969, 0.967, 0.966, 0.965,
                0.964, 0.962, 0.960
            ]
        }
    },

    "PointNet++": {
        "High_Drop": {
            "PcSM": [
                0.96, 0.952, 0.947, 0.942,
                0.94, 0.937, 0.934, 0.931,
                0.932, 0.932, 0.931
            ],
            "Grad-Vis": [
                0.96, 0.939, 0.902, 0.854,
                0.854, 0.847, 0.834, 0.828,
                0.820, 0.806, 0.798
            ],
            "Ours": [
                0.96, 0.941, 0.914, 0.877,
                0.845, 0.804, 0.761, 0.718,
                0.678, 0.637, 0.595
            ],
            "MSXAI": [
                0.960, 0.950, 0.934, 0.911,
                0.894, 0.880, 0.867, 0.854,
                0.841, 0.828, 0.813
            ],
            "IIA":[
                0.960, 0.957, 0.955, 0.952,
                0.949, 0.946, 0.943, 0.941,
                0.938, 0.936, 0.933
            ]
        },

        "Low_Drop": {
            "PcSM": [
                0.963, 0.943, 0.932, 0.924, 0.914,
                0.904, 0.895, 0.888, 0.880, 0.873, 0.871
            ],
            "Grad-Vis": [
                0.963, 0.956, 0.950, 0.944, 0.932,
                0.921, 0.903, 0.882, 0.862, 0.833, 0.803
            ],
            "Ours": [
                0.963, 0.959, 0.957, 0.948, 0.940,
                0.926, 0.912, 0.891, 0.871, 0.850, 0.827
            ],
            "MSXAI": [
                0.963, 0.954, 0.946, 0.939, 0.929,
                0.919, 0.904, 0.893, 0.873, 0.851, 0.828
            ],
            "IIA":[
                0.963, 0.949, 0.937, 0.924,
                0.910, 0.896, 0.884, 0.871,
                0.856, 0.838, 0.820
            ]
        }
    }
}

x = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.xticks(fontname='Times New Roman', fontsize=12)
plt.yticks(fontname='Times New Roman', fontsize=12)

# 创建图像
plt.figure(figsize=(10, 6))
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
saliency_high_correct = data["PointNet"]["High_Drop"]["PcSM"]
grad_high_correct = data["PointNet"]["High_Drop"]["Grad-Vis"]
ours_high_correct = data["PointNet"]["High_Drop"]["Ours"]
fbi_high_correct = data["PointNet"]["High_Drop"]["FBI"]
msxai_high_correct = data["PointNet"]["High_Drop"]["MSXAI"]
iia_high_correct = data["PointNet"]["High_Drop"]["IIA"]

# saliency_high_correct = data["PointNet"]["Low_Drop"]["PcSM"]
# grad_high_correct = data["PointNet"]["Low_Drop"]["Grad-Vis"]
# ours_high_correct = data["PointNet"]["Low_Drop"]["Ours"]
# fbi_high_correct = data["PointNet"]["Low_Drop"]["FBI"]
# msxai_high_correct = data["PointNet"]["Low_Drop"]["MSXAI"]
# iia_high_correct = data["PointNet"]["Low_Drop"]["IIA"]

# 绘制每条线

plt.plot(x, saliency_high_correct[:11], label="PcSM", color='green', linestyle='--', marker='s')
plt.plot(x, grad_high_correct[:11], label="Grad-Vis", color='purple', linestyle='-.', marker='^')
plt.plot(x, fbi_high_correct[:11], label="FBI", color='blue', linestyle=(0, (3, 1, 1, 1)), marker='o')
plt.plot(x, msxai_high_correct[:11], label="MSXAI", color='orange', linestyle='-', marker='*')
plt.plot(x, iia_high_correct[:11], label="IIA", color='#17BECF', linestyle=(0, (5, 2)), marker='X')
plt.plot(x, ours_high_correct[:11], label="Ours", color='red', linestyle=':', marker='d')

# 设置图例（左下角）
plt.legend(loc='lower left')

# 设置标题和轴标签
plt.title("Comparison of High Dropping", fontsize=16)
# plt.title("Comparison of Low Dropping", fontsize=16)
plt.xlabel("Number of Drops", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)

# 动态调整 Y 轴范围
min_y = min( min(saliency_high_correct[:11]), min(grad_high_correct[:11]), min(ours_high_correct[:11]))
min_y = 0.91
max_y = max( max(saliency_high_correct[:11]), max(grad_high_correct[:11]), max(ours_high_correct[:11]))
padding = 0.05 * (max_y - min_y)  # 添加 5% 的上下边距
# plt.ylim(0.91, 0.99)
# plt.ylim(min_y - padding, max_y + padding)

# 设置 X 轴范围
plt.xlim(0, 500)

# 添加网格以便于观察
plt.grid(alpha=0.3)

# 显示图像
plt.tight_layout()
plt.savefig('./plt.png', dpi=300)
plt.show()