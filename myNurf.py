"""
这是一份 TinyNurf 的简单实现，目的是通过一系列不同方位拍摄的照片推测未拍摄方位的图像
训练过程整体可以分为 7 步：
1. 构造一个神经网络模型
2. 选取某一图片，使用 MLP 预测像素的值
3. 准备相机和图像资源
4. 获取相机射向图中所有像素的所有光线
5. 在这条光线上撒点，用于之后的训练
6. 把所有点的坐标进行位置编码，增加其在神经网络眼里的差异
7. 通过这 32 个点进行体渲染，得到MLS眼里该像素的颜色，算出 Loss 值，更新 MLP 参数，然后跳回第 2 步，直到你想停止
"""

import numpy as np
import torch

def get_ray_bundle(height: int, width: int, focal: float, pose: torch.Tensor):
    """
    函数功能：获得所有由摄像机发出，射向每一个像素的光线。它们的原点是摄像机，方向朝向每一个像素
    :param 由两个整数，一个浮点数和一个 4*4 矩阵。分别表示图片宽和高组成，相机焦距
    :return 由一个三维向量和一个矩阵构成。向量代表所有光线的原点（摄像机坐标），矩阵
    """

    device = pose.device

    # meshgrid 生成横纵坐标矩阵
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float, device=device),
        torch.arange(height, dtype=torch.float, device=device),
        indexing='xy'
    )


    # 创造一个包含所有像素的，相对相机坐标的矩阵
    dirs = torch.stack([
        (i - width * 0.5) / focal,
        -(j - height * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)

    # 提取旋转矩阵和位置向量
    rotation_matrix = pose[:3, :3]
    camera_position = pose[:3, 3]

    # 得到最终的，包含所有方向向量的矩阵
    ray_directions = torch.sum(dirs[..., None, :] * rotation_matrix, dim=-1)

    # 方向向量归一化
    norms = torch.norm(ray_directions, dim=-1, keepdim=True)
    ray_directions = ray_directions / norms


    # 所有的光线原点都是相机所在的位置
    ray_origins = camera_position.expand(ray_directions.shape)

    return ray_origins, ray_directions


def randomize_matrix(t_matrix : torch.Tensor) :
    """
    函数功能：将给定矩阵进行随机的扰动
    :param 一个 图片大小*32 的矩阵，用于表示 32 个变化向量
    :return 扰动后的矩阵
    """

    mids = 0.5 * (t_matrix[..., 1:] + t_matrix[..., :-1])
    upper = torch.cat([mids, t_matrix[..., -1:]], -1)
    lower = torch.cat([t_matrix[..., :1], mids], -1)

    t_rand = torch.rand_like(t_matrix)
    return lower + t_rand * (upper - lower)


def create_points(ray_origins: torch.Tensor, ray_directions: torch.Tensor, N: int, t_min: float, t_max: float,
                  randomize=False):
    """
    函数功能：通用版采点函数，同时支持 (H, W, 3) 的图片输入和 (Batch, 3) 的光线输入

    """
    device = ray_origins.device

    batch_shape = ray_origins.shape[:-1]

    t_line = torch.linspace(t_min, t_max, N, device=device)

    view_shape = [1] * len(batch_shape) + [N]

    t_matrix = t_line.view(*view_shape).expand(*batch_shape, N).clone()

    if randomize == True:
        t_matrix = randomize_matrix(t_matrix)

    t_matrix = t_matrix[..., :, None]

    ray_origins = ray_origins[..., None, :]
    ray_directions = ray_directions[..., None, :]

    return ray_origins + (t_matrix * ray_directions), t_matrix

def position_encode(point_position : torch.Tensor, D : int) :
    """
    函数功能：对采点得到的三维位置矩阵进行位置编码
    :param 一个 H*W*32*3 的采点得到的三维面积矩阵，以及一个代表编码后维数的整数 D
    :return 一个 H*W*32*(D*3) 的新矩阵，之后这个矩阵会将最后 D*3 拆成三部分作为 x,y,z 加上两个角度一起喂给 MLP
    """
    encode = [point_position]
    power_list = 2.0 ** torch.linspace(0, D - 1, D)

    for power in power_list:
        encode.append(torch.sin(point_position * power))
        encode.append(torch.cos(point_position * power))

    final_encode = torch.cat(encode, dim=-1)
    return final_encode



def volume_rendering(raw_data, z_vals, ray_directions):
    """
    函数功能：通过算出的颜色，密度和点之间的距离进行体渲染求出当前 MLP 眼里的图片
    :param MLP 返回的颜色和密度数组，点与点之间的距离数组
    :return MLP 眼里的图像
    """

    # 挤压 z_val 最后一维，保证维数一致
    z_vals = z_vals.squeeze(-1)

    # 拆分并激活数据
    rgb = torch.sigmoid(raw_data[..., :3])  # (H, W, N, 3)
    sigma = torch.nn.functional.relu(raw_data[..., 3])  # (H, W, N)

    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # 在最后一个点之后补上一个很大的距离
    last_dist = torch.tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)
    dists = torch.cat([dists, last_dist], dim=-1)  # (H, W, N)

    # 计算不透明度 Alpha
    alpha = 1.0 - torch.exp(-sigma * dists)

    # 计算透过率 Transmittance (T)
    transparency = 1.0 - alpha + 1e-10
    accum_prod = torch.cumprod(transparency, dim=-1)

    # 拼接一个全 1 的向量在最前面，扔掉最后一个
    ones = torch.ones_like(accum_prod[..., :1])
    # T: [1, t0, t0*t1, ...]
    T = torch.cat([ones, accum_prod[..., :-1]], dim=-1)

    # 计算每一个点的权重
    weights = alpha * T  # (H, W, N)

    # 加权求和得到最终像素颜色
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rgb_map, depth_map, acc_map

import torch.nn as nn
class TinyNeRF(nn.Module):
    def __init__(self, D=10, hidden_dim=256):
        """
        初始化 TinyNeRF 模型
        :param D: 位置编码的频率数，用于计算输入层维度
        :param hidden_dim: 隐藏层神经元数量，通常 128 或 256
        """
        super(TinyNeRF, self).__init__()

        self.input_dim = 3 + 3 * 2 * D

        self.layer1 = nn.Linear(self.input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)

        self.layer4 = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        """
        前向传播
        :param x: 经位置编码后的矩阵，形状为 (H, W, N, input_dim)
        :return: 预测的颜色和密度，形状为 (H, W, N, 4)
        """
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        out = self.layer4(x)
        return out

D = 10
hidden_dim = 256
learn_rate = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyNeRF(D=D, hidden_dim=hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)

data = np.load("tiny_nerf_data.npz")
# 获得图片，摄像机焦距和图片长宽
images = data["images"]
focal = float(data["focal"])
H = images.shape[1]
W = images.shape[2]

print("开始训练...")


all_rays = []
all_rgbs = []

# 遍历每一张图片
for i in range(images.shape[0]):
    img = images[i]  # (H, W, 3)
    pose = torch.tensor(data["poses"][i]).to(device)

    ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose)

    # 展平：(H, W, 3) -> (H*W, 3)
    ray_origins = ray_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)
    img = torch.tensor(img).reshape(-1, 3).to(device)

    all_rays.append(torch.cat([ray_origins, ray_directions], dim=1))
    all_rgbs.append(img)

all_rays = torch.cat(all_rays, dim=0)
all_rgbs = torch.cat(all_rgbs, dim=0)

print(f"预处理完成！总共有 {all_rays.shape[0]} 条光线。")

batch_size = 4096  # 每次随机取 4096 条光线 (2^12)
n_iters = 5000

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

print("开始 High-Quality 训练...")

for i in range(n_iters + 1):

    # 随机抽取 batch_size 条光线
    idxs = torch.randint(0, all_rays.shape[0], (batch_size,), device=device)

    batch_rays = all_rays[idxs]  # (batch_size, 6)
    target_rgb = all_rgbs[idxs]  # (batch_size, 3)

    ray_origins = batch_rays[:, :3]
    ray_directions = batch_rays[:, 3:]

    points, z_vals = create_points(ray_origins, ray_directions, N=64, t_min=2.0, t_max=6.0)

    points = points.to(device)
    z_vals = z_vals.to(device)

    # 编码
    encoded_points = position_encode(points, D=10)  # D改成了10

    # 预测
    raw_data = model(encoded_points)

    # 渲染
    rgb_map, _, _ = volume_rendering(raw_data, z_vals, ray_directions)

    # 计算 Loss
    loss = torch.mean((rgb_map - target_rgb) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 500 == 0:
        print(f"Iter: {i}, Loss: {loss.item()}")
print("训练结束")

print("开始画图")

import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt

def get_pose_spherical(theta, radius=4.0):
    """
    函数功能：生成绕原点旋转的相机
    :param theta:相位角
    :param radius: 旋转角度
    :return:
    """
    def look_at(eye, center, up):
        z = eye - center
        z = z / torch.norm(z)

        x = torch.cross(up, z, dim=-1)
        x = x / torch.norm(x)

        y = torch.cross(z, x, dim=-1)

        matrix = torch.eye(4)
        matrix[:3, 0] = x
        matrix[:3, 1] = y
        matrix[:3, 2] = z
        matrix[:3, 3] = eye
        return matrix

    camera_pos = torch.tensor([
        radius * np.cos(theta),
        radius * np.sin(theta),
        2.0
    ], dtype=torch.float32)

    center = torch.tensor([0., 0., 0.], dtype=torch.float32)
    up = torch.tensor([0., 1., 0.], dtype=torch.float32)

    pose = look_at(camera_pos, center, up)

    return pose


print("开始渲染 GIF 帧")

frames = []
data_poses = torch.tensor(data["poses"])
avg_radius = torch.norm(data_poses[:, :3, 3], dim=-1).mean().item()
print(f"计算得出的相机半径: {avg_radius:.2f}")

angles = np.linspace(0, 2 * np.pi, 60 + 1)[:-1]

model.eval()

with torch.no_grad():
    for idx, angle in enumerate(angles):
        # 生成位姿
        pose = get_pose_spherical(angle, radius=avg_radius).to(device)

        # 获取光线
        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose)
        ray_origins = ray_origins.to(device)
        ray_directions = ray_directions.to(device)

        # 3. 采样编码
        points, z_vals = create_points(ray_origins, ray_directions, N=32, t_min=2.0, t_max=6.0)

        points = points.to(device)
        z_vals = z_vals.squeeze(-1).to(device)

        encoded_points = position_encode(points, D=10)
        raw_data = model(encoded_points)

        # 体渲染
        rgb_map, _, _ = volume_rendering(raw_data, z_vals, ray_directions)

        # 图片后处理
        rgb_map = rgb_map.cpu().numpy()

        # 处理最终数据
        rgb_map = np.clip(rgb_map, 0, 1)
        rgb_image = (rgb_map * 255).astype(np.uint8)

        frames.append(rgb_image)

        if idx % 10 == 0:
            print(f"渲染进度: {idx}/60")

save_path = "nerf_result.gif"
imageio.mimsave(save_path, frames, fps=60)

print(f"动图已保存至 {save_path}")