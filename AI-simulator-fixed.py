import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.optim as optim 
from matplotlib.lines import Line2D
import os
from matplotlib.font_manager import FontProperties
import time
import matplotlib.collections as collections
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 全局变量，用于控制训练过程
stop_event = threading.Event()  # 使用 Event 替代布尔标志
training_thread = None
current_training_thread = None  # 当前运行的训练线程

# 全局变量，用于存储层配置
layer_entries = []  # 存储每一层的输入框
layer_delete_buttons = []  # 存储每一层的删除按钮

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

x_range = (-10, 10)  # 扩大 x 轴范围
y_range = (-10, 10)  # 扩大 y 轴范围

# 在程序开始部分（全局变量定义区域附近），增加损失函数和字典定义
# 在现有的全局变量之后添加，大约在第30行左右

# 在全局变量部分，添加优化器选择
optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
    "Adagrad": optim.Adagrad
}

# 定义损失函数
def mse_loss_func(x, y=None):
    """均方误差损失函数
    如果提供了y参数，计算神经网络的MSE损失
    如果只提供x参数(作为二维坐标)，计算等高线图上的MSE损失"""
    if y is not None:
        # 神经网络模式
        return F.mse_loss(x, y)
    else:
        # 等高线模式
        x_coord, y_coord = x
        return (x_coord - 1.0) ** 2 + (y_coord - 2.0) ** 2

def mae_loss_func(x, y=None):
    """平均绝对误差损失函数"""
    if y is not None:
        # 神经网络模式
        return F.l1_loss(x, y)
    else:
        # 等高线模式
        x_coord, y_coord = x
        return abs(x_coord - 1.0) + abs(y_coord - 2.0)

def huber_loss_func(x, y=None):
    """Huber损失函数"""
    if y is not None:
        # 神经网络模式
        return F.smooth_l1_loss(x, y)
    else:
        # 等高线模式
        x_coord, y_coord = x
        dx = abs(x_coord - 1.0)
        dy = abs(y_coord - 2.0)
        # Huber损失实现
        delta = 1.0
        if dx < delta:
            loss_x = 0.5 * dx * dx
        else:
            loss_x = delta * (dx - 0.5 * delta)
        if dy < delta:
            loss_y = 0.5 * dy * dy
        else:
            loss_y = delta * (dy - 0.5 * delta)
        return loss_x + loss_y

def log_cosh_loss_func(x, y=None):
    """Log-Cosh损失，对异常值不敏感且二阶可导"""
    if y is not None:
        # 神经网络模式
        diff = x - y
        return torch.mean(torch.log(torch.cosh(diff)))
    else:
        # 等高线模式
        x_coord, y_coord = x
        dx = x_coord - 1.0
        dy = y_coord - 2.0
        return np.log(np.cosh(dx)) + np.log(np.cosh(dy))

def rosenbrock_loss_func(x, y=None):
    """Rosenbrock函数: 100*(y-x^2)^2 + (1-x)^2"""
    if y is not None:
        # 神经网络模式 - 使用reshape将输入变为一维
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        # 确保有足够的元素
        if len(x_flat) > 1 and len(y_flat) > 1:
            return torch.mean(100.0 * (y_flat - x_flat**2)**2 + (1.0 - x_flat)**2)
        else:
            return F.mse_loss(x, y)  # 回退到MSE
    else:
        # 等高线模式
        x_coord, y_coord = x
        return 100.0 * (y_coord - x_coord**2)**2 + (1.0 - x_coord)**2

def himmelblau_loss_func(x, y=None):
    """Himmelblau函数: (x^2+y-11)^2 + (x+y^2-7)^2"""
    if y is not None:
        # 神经网络模式 - 使用reshape将输入变为一维
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        # 确保有足够的元素
        if len(x_flat) > 1 and len(y_flat) > 1:
            return torch.mean((x_flat**2 + y_flat - 11.0)**2 + (x_flat + y_flat**2 - 7.0)**2)
        else:
            return F.mse_loss(x, y)  # 回退到MSE
    else:
        # 等高线模式
        x_coord, y_coord = x
        return (x_coord**2 + y_coord - 11.0)**2 + (x_coord + y_coord**2 - 7.0)**2
    
def generate_points(n, distribution="linear", x_range=(0, 100), y_range=(0, 100)):
    """
    生成点数据
    :param n: 点的数量
    :param distribution: 生成点的方式，可选 "linear", "random", "vertical", "curve", "custom"
    :return: 生成的点数据
    """
    if distribution == "linear":
        x = np.linspace(0, 1, n)
        y = x + np.random.normal(0, 0.1, n)
    elif distribution == "random":
        x = np.random.rand(n)
        y = np.random.rand(n)
    elif distribution == "vertical":
        x = np.random.normal(0.5, 0.1, n)
        y = np.linspace(0, 1, n)
    elif distribution == 'curve':
        # 随机选择曲线形状
        curve_type = np.random.choice(['sin', 'cos', 'w', 'm'])
        
        # 生成曲线 y = a * sin(b * x) + c * x + d
        a = (y_range[1] - y_range[0]) / 4  # 振幅
        b = 2 * np.pi / (x_range[1] - x_range[0])  # 频率
        c = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])  # 线性斜率
        d = y_range[0]  # 偏移量
        x = np.linspace(x_range[0], x_range[1], n)
        
        if curve_type == 'sin':
            y = a * np.sin(b * x) + c * x + d
        elif curve_type == 'cos':
            y = a * np.cos(b * x) + c * x + d
        elif curve_type == 'w':
            y = a * np.sin(2 * b * x) + c * x + d  # 双倍频率，形成 w 形
        elif curve_type == 'm':
            y = -a * np.sin(2 * b * x) + c * x + d  # 双倍频率，负正弦，形成 M 形
        
        # 添加噪声
        y += np.random.normal(0, 0.1, n)
        points = np.column_stack((x, y))
    #elif distribution == 'custom':
        # 使用用户绘制的点生成数据
        """ if len(custom_curve_points) >= 1:
            points = []
            for _ in range(n):
                # 随机选择一个用户绘制的点
                base_point = custom_curve_points[np.random.randint(0, len(custom_curve_points))]
                # 在点附近随机生成数据
                x = base_point[0] + np.random.normal(0, 0.05)
                y = base_point[1] + np.random.normal(0, 0.05)
                points.append((x, y))
            points = np.array(points) """

    else:
        raise ValueError(f"不支持的生成点方式: {distribution}")
    return np.column_stack((x, y))
def plot_points(points):
    """
    在二维平面绘制坐标点
    :param points: 坐标点
    """
    n = len(points)
    colors = ['blue'] * (n // 2) + ['red'] * (n // 2)  # 生成等量的蓝色和红色
    np.random.shuffle(colors)  # 随机打乱颜色顺序
    plt.scatter(points[:, 0], points[:, 1], c=colors)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Randomly Generated Points')
    plt.grid(True)
    plt.show()

def split_data(points, colors):
    """
    根据颜色分割数据集和测试集
    :param points: 所有点
    :param colors: 颜色列表
    :return: 数据集 (蓝色), 测试集 (红色)
    """
    dataset = points[np.array(colors) == 'blue']
    testset = points[np.array(colors) == 'red']
    return dataset, testset
# 损失函数字典，统一管理所有损失函数
loss_functions = {
    "MSE损失": (mse_loss_func, np.array([1.0, 2.0]), 0.1),
    "MAE损失": (mae_loss_func, np.array([1.0, 2.0]), 0.05),
    "Huber损失": (huber_loss_func, np.array([1.0, 2.0]), 0.05),
    "Log-Cosh损失": (log_cosh_loss_func, np.array([1.0, 2.0]), 0.05),
    "Rosenbrock函数": (rosenbrock_loss_func, np.array([1.0, 1.0]), 0.001),
    "Himmelblau函数": (himmelblau_loss_func, np.array([3.0, 2.0]), 0.01)
}

# 当前选择的损失函数
current_loss_function = "MSE损失"
# 移除等高线损失函数的单独配置
# current_contour_loss_function = "MSE损失"

# 添加路径变量，用于跟踪梯度下降
path = []

# 全局变量用于3D视图状态
use_3d_view = False  # 默认使用2D视图

# 在全局变量部分添加
contour_path = []  # 用于存储梯度下降路径
optimal_point = None  # 用于存储最优点

# 在全局变量部分添加
selected_weight = "0-0"  # 默认显示第0层第0个权重

# 在全局定义 fig_contour
global fig_contour

class SimpleNN(nn.Module):
    """
    可配置的神经网络模型
    """
    def __init__(self, hidden_layers=[50, 100, 50], activation='tanh'):
        super(SimpleNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        
        # 输入层
        self.layers.append(nn.Linear(1, hidden_layers[0]))  # 输入是x值
        
        # 隐藏层
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        
        # 输出层
        self.layers.append(nn.Linear(hidden_layers[-1], 1))  # 输出是y值
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation == 'relu':
                x = torch.relu(x)
            elif self.activation == 'sigmoid':
                x = torch.sigmoid(x)
            elif self.activation == 'tanh':
                x = torch.tanh(x)
            else:
                raise ValueError("Unsupported activation function")
        x = self.layers[-1](x)
        return x

def draw_neural_net(ax, layer_sizes, weights, model,epoch):
    """
    绘制神经网络结构
    :param ax: matplotlib的axes对象
    :param layer_sizes: 每层的神经元数量，例如 [2, 3, 1]
    :param weights: 每层的权重矩阵
    :param model: 当前训练的模型
    """
    ax.clear()
    n_layers = len(layer_sizes)
    v_spacing = 1.0 / float(max(layer_sizes))
    h_spacing = 1.0 / float(n_layers - 1)  # 恢复默认水平间距
    
    # 绘制神经元
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2.0 + 0.5
        for j in range(layer_size):
            circle = plt.Circle((i * h_spacing, layer_top - j * v_spacing), v_spacing / 4.0, color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    
    # 绘制连接线
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2.0 + 0.5
        layer_top_b = v_spacing * (layer_size_b - 1) / 2.0 + 0.5
        weight_matrix = weights[i]  # 获取当前层的权重矩阵
        abs_weights = np.abs(weight_matrix)
        threshold = np.percentile(abs_weights, 90)  # 计算权重的前10%阈值
        
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                weight = weight_matrix[k, j]
                if abs(weight) >= threshold:  # 高权重使用蓝色
                    line = Line2D([i * h_spacing, (i + 1) * h_spacing],
                                  [layer_top_a - j * v_spacing, layer_top_b - k * v_spacing],
                                  c='blue', alpha=0.8, linewidth=abs(weight) * 2)
                else:  # 普通权重使用灰色
                    line = Line2D([i * h_spacing, (i + 1) * h_spacing],
                                  [layer_top_a - j * v_spacing, layer_top_b - k * v_spacing],
                                  c='gray', alpha=0.5, linewidth=abs(weight) * 2)
                line.set_label(f"{j}-{k}")  # 设置权重索引
                line.set_picker(True)  # 启用选择器
                ax.add_artist(line)
    
    ax.set_xlim(0, 1)  # 恢复默认水平范围
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # 绑定鼠标事件
    fig = ax.get_figure()
    fig.canvas.mpl_connect('pick_event', lambda event: on_weight_select(event, model,epoch))

def train_and_test(dataset, testset, epochs, lr, hidden_layers, activation, ax_loss, ax_learning, canvas, ax_neural_network):
    """
    训练和测试神经网络
    """
    global stop_event, contour_path
    
    # 初始化模型
    model = SimpleNN(hidden_layers=hidden_layers, activation=activation)
    
    # 清空路径
    contour_path = []
    
    # 获取用户选择的训练损失函数并创建损失函数
    train_loss_func_name = loss_function_var.get()
    train_loss_func, _, _ = loss_functions[train_loss_func_name]
    criterion = lambda input, target: train_loss_func(input, target)
    
    # 获取用户选择的优化器
    optimizer_name = optimizer_var.get()
    optimizer_class = optimizers[optimizer_name]
    optimizer = optimizer_class(model.parameters(), lr=lr)
    
    # 获取更新频率
    try:
        update_freq = int(update_frequency_entry.get())
        if update_freq <= 0:
            update_freq = 10  # 默认值
    except ValueError:
        update_freq = 10  # 默认值
    
    # 数据预处理
    dataset_x = torch.tensor(dataset[:, 0], dtype=torch.float32).unsqueeze(1)
    dataset_y = torch.tensor(dataset[:, 1], dtype=torch.float32).unsqueeze(1)
    testset_x = torch.tensor(testset[:, 0], dtype=torch.float32).unsqueeze(1)
    testset_y = torch.tensor(testset[:, 1], dtype=torch.float32).unsqueeze(1)
    
    # 记录训练和测试损失
    train_losses = []
    test_losses = []
    
    # 确保图表初始化
    ax_loss.clear()
    ax_learning.clear()
    
    # 绘制初始数据点
    ax_learning.scatter(dataset[:, 0], dataset[:, 1], c='blue', label='训练数据')
    ax_learning.scatter(testset[:, 0], testset[:, 1], c='red', label='测试数据')
    ax_learning.set_xlabel('输入特征')
    ax_learning.set_ylabel('输出值')
    ax_learning.set_title('学习曲线')
    ax_learning.legend()
    ax_learning.grid(True)
    
    # 初始刷新画布
    canvas.draw()
    root.update()
    
    for epoch in range(epochs):
        if stop_event.is_set():
            break
        
        # 前向传播
        outputs = model(dataset_x)
        loss = criterion(outputs, dataset_y)  # 使用之前定义的criterion
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录训练loss
        train_losses.append(loss.item())
        
        # 记录测试loss
        with torch.no_grad():
            test_outputs = model(testset_x)
            test_loss = criterion(test_outputs, testset_y)
            test_losses.append(test_loss.item())
        
        # 记录当前参数位置
        with torch.no_grad():
            # 计算当前参数在损失函数空间中的位置
            x = model.layers[0].weight.data.mean().item()
            y = model.layers[0].bias.data.mean().item()
            
            # 确保记录的点在合理范围内
            x = np.clip(x, x_range[0], x_range[1])
            y = np.clip(y, y_range[0], y_range[1])
            contour_path.append((x, y))
        
        # 更新图表显示
        if (epoch + 1) % update_freq == 0 or epoch == 0:
            # 更新loss曲线
            ax_loss.clear()
            ax_loss.plot(train_losses, label=f'训练损失 ({train_loss_func_name})')
            ax_loss.plot(test_losses, label=f'测试损失 ({train_loss_func_name})')
            ax_loss.set_xlabel('迭代次数')
            ax_loss.set_ylabel('损失值')
            ax_loss.set_title('损失曲线')
            ax_loss.legend()
            ax_loss.grid(True)
        
            # 更新学习曲线
            ax_learning.clear()
            ax_learning.scatter(dataset[:, 0], dataset[:, 1], c='blue', label='训练数据')
            ax_learning.scatter(testset[:, 0], testset[:, 1], c='red', label='测试数据')
            
            # 确保x值有序并覆盖数据的整个范围
            x_min = min(dataset[:, 0].min(), testset[:, 0].min()) - 0.1
            x_max = max(dataset[:, 0].max(), testset[:, 0].max()) + 0.1
            x_vals = np.linspace(x_min, x_max, 100)
            x_vals_tensor = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(1)
            
            with torch.no_grad():
                y_vals = model(x_vals_tensor).numpy()
            
            # 绘制学习曲线
            sort_idx = np.argsort(x_vals)
            ax_learning.plot(x_vals[sort_idx], y_vals[sort_idx], 'g-', label='学习曲线')
            
            ax_learning.set_xlabel('输入特征')
            ax_learning.set_ylabel('输出值')
            ax_learning.set_title('学习曲线')
            ax_learning.legend()
            ax_learning.grid(True)
        
            # 更新神经网络结构
            try:
                layer_sizes = [dataset_x.shape[1]] + hidden_layers + [dataset_y.shape[1]]
                weights = [param.data.numpy() for param in model.parameters() if len(param.shape) == 2]
                if len(weights) == len(layer_sizes) - 1:
                    draw_neural_net(ax_neural_network, layer_sizes, weights, model,epoch)
            except Exception as e:
                print(f"绘制神经网络结构时出错: {e}")
            
            # 更新等高线图
            update_contour_plot(model, dataset_x, dataset_y, epoch, loss.item())
            
            # 更新权重信息时传递当前选中的权重
            update_weight_info(epoch, epochs, weights, weight_info_text, model, selected_weight)
            
            # 刷新画布
            canvas.draw()
            root.update()
        
        update_weight_info(epoch, epochs, weights, weight_info_text,model)
    
    # 训练结束后恢复按钮状态
    start_button.config(text="启动训练", command=start_training)

def normalize_data(data, data_range):
    """
    归一化数据到0-1范围
    :param data: 输入数据
    :param data_range: 数据范围 (min, max)
    """
    return (data - data_range[0]) / (data_range[1] - data_range[0])

def denormalize_data(data, data_range):
    """
    反归一化数据
    :param data: 归一化后的数据
    :param data_range: 数据范围 (min, max)
    """
    return data * (data_range[1] - data_range[0]) + data_range[0]

def add_layer():
    """
    添加一层神经网络
    """
    layer_index = len(layer_entries) + 1
    ttk.Label(layer_frame, text=f"第 {layer_index} 层神经元数量:").grid(row=layer_index, column=0, sticky="w")
    entry = ttk.Entry(layer_frame, width=5)  # 调整宽度为5，适合输入3个数字
    entry.grid(row=layer_index, column=1, padx=5, pady=5)
    entry.insert(0, "10")  # 默认神经元数量为10
    layer_entries.append(entry)
    
    # 添加删除按钮
    delete_button = ttk.Button(layer_frame, text="-", width=2, command=lambda idx=layer_index-1: delete_layer(idx))  # 调整宽度为2
    delete_button.grid(row=layer_index, column=2, padx=5)
    layer_delete_buttons.append(delete_button)

def delete_layer(index):
    """
    删除指定层
    :param index: 层的索引
    """
    # 删除输入框和标签
    for widget in layer_frame.grid_slaves(row=index+1):
        widget.destroy()
    
    # 从列表中移除
    layer_entries.pop(index)
    layer_delete_buttons.pop(index)
    
    # 更新剩余层的标签和按钮
    for i in range(index, len(layer_entries)):
        for widget in layer_frame.grid_slaves(row=i+1):
            widget.grid(row=i+1)
        layer_delete_buttons[i].config(command=lambda idx=i: delete_layer(idx))

def start_training():
    """从界面获取参数并启动训练"""
    global training_thread, stop_event
    
    try:
        # 重新初始化等高线图
        fig_contour.clear()
        
        # 清除颜色条
        try:
            # 清除所有现有的颜色条
            for ax in fig_contour.axes:
                if ax.get_label() == 'colorbar':  # 通过标签识别颜色条
                    ax.remove()  # 删除颜色条轴
        except Exception as e:
            print(f"清除颜色条时出错: {e}")
        
        # 获取参数
        epochs = int(epochs_entry.get())
        lr = learning_rate_var.get()
        activation = activation_var.get()
        
        # 获取隐藏层配置
        hidden_layers = []
        for entry in layer_entries:
            try:
                neurons = int(entry.get())
                hidden_layers.append(neurons)
            except ValueError:
                # 处理非法输入
                neurons = 10  # 默认值
                entry.delete(0, tk.END)
                entry.insert(0, str(neurons))
                hidden_layers.append(neurons)
        
        # 生成数据
        distribution = data_gen_var.get()
        num_points = 200  # 可以设为可配置参数
        
        # 根据用户选择的方式生成数据
        if distribution == "custom":
            # 使用自定义曲线
            if len(custom_curve_points) >= 1:
                # 生成平滑的插值曲线
                x_vals = [p[0] for p in custom_curve_points]
                y_vals = [p[1] for p in custom_curve_points]
                
                # 确保点按x坐标排序，避免曲线交叉
                sorted_points = sorted(zip(x_vals, y_vals), key=lambda p: p[0])
                x_sorted = [p[0] for p in sorted_points]
                y_sorted = [p[1] for p in sorted_points]
                
                # 使用更多点来生成平滑曲线
                x_interp = np.linspace(0, 1, 200)
                y_interp = np.interp(x_interp, x_sorted, y_sorted)
                
                # 生成训练和测试数据
                n = num_points
                x_train = np.random.uniform(0, 1, n)
                y_train = np.interp(x_train, x_interp, y_interp) + np.random.normal(0, 0.05, n)
                
                # 确保坐标在合理范围内
                y_train = np.clip(y_train, 0, 1)
                
                # 创建数据点
                points = np.column_stack((x_train, y_train))
            else:
                # 如果没有绘制点，使用随机数据
                print("没有检测到自定义曲线，使用随机数据")
                n = num_points
                points = generate_points(n, distribution="random")
        else:
            # 使用其他生成方式
            points = generate_points(num_points, distribution=distribution, x_range=x_range, y_range=y_range)
        
        # 划分数据集
        n = len(points)
        colors = ['blue'] * (n // 2) + ['red'] * (n - n // 2)  # 确保颜色总数等于点的总数
        np.random.shuffle(colors)  # 随机打乱颜色顺序
        
        dataset, testset = split_data(points, colors)
        
        # 重置停止标志
        stop_event.clear()
        
        # 启动训练线程
        training_thread = threading.Thread(
            target=train_and_test, 
            args=(dataset, testset, epochs, lr, hidden_layers, activation, 
                  ax_loss, ax_learning, canvas, ax_neural_network)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # 更新按钮状态
        start_button.config(text="停止训练", command=stop_training_process)
        
    except Exception as e:
        print(f"启动训练时出错: {e}")
        import traceback
        traceback.print_exc()

def stop_training_process():
    """
    停止训练过程并重新初始化界面
    """
    global stop_event
    stop_event.set()  # 设置停止标志
    
    
    # 更新按钮状态
    start_button.config(text="启动训练", command=start_training)

def on_closing():
    """
    处理窗口关闭事件
    """
    global stop_event, training_thread
    stop_training_process()  # 调用停止训练的逻辑
    if training_thread and training_thread.is_alive():
        print("等待训练线程结束...")
        training_thread.join(timeout=1)  # 等待线程结束，最多等待1秒
        if training_thread.is_alive():
            print("训练线程未正常退出，强制终止程序...")
            os._exit(0)  # 强制退出
    root.destroy()  # 关闭窗口

def change_loss_function():
    # 更新当前选择的损失函数
    global current_loss_function
    current_loss_function = loss_function_var.get()
    print(f"当前损失函数已切换为: {current_loss_function}")

def update_contour_plot(model, dataset_x, dataset_y, epoch, current_loss):
    """更新等高线图"""
    global contour_path, path_line, current_point, fig_contour, ax_contour
    
    try:
        display_mode = contour_mode_var.get()
        layer_idx, weight_idx = map(int, selected_weight.split('-'))
        
        if epoch == 0:
            # 第一次调用时初始化等高线图
            fig_contour.clear()
            
            # 创建新的子图
            ax_contour = fig_contour.add_subplot(111)
            
            # 扩大坐标范围
            margin = 5  # 增大边界范围
            grid_size = 50
            x = np.linspace(-margin, margin, grid_size)
            y = np.linspace(-margin, margin, grid_size)
            X, Y = np.meshgrid(x, y)
            
            # 计算值
            if display_mode == "loss":
                Z = calculate_loss_surface(X, Y, model, dataset_x, dataset_y)
                label = 'Loss Value'
            else:
                Z = calculate_gradient_surface(X, Y, model, dataset_x, dataset_y)
                label = 'Gradient Magnitude'
            
            # 计算最优点
            min_idx = np.unravel_index(np.argmin(Z), Z.shape)
            optimal_x = X[min_idx]
            optimal_y = Y[min_idx]
            
            # 绘制等高线，减少等高线数量
            levels = np.linspace(Z.min(), Z.max(), 10)  # 减少等高线数量
            CS = ax_contour.contour(X, Y, Z, levels=levels, cmap='viridis')
            ax_contour.clabel(CS, inline=True, fontsize=8)
            
            # 添加颜色填充，降低透明度
            contour_filled = ax_contour.contourf(X, Y, Z, levels=levels, 
                                               cmap='viridis', alpha=0.2)  # 降低透明度
            cbar = plt.colorbar(contour_filled, ax=ax_contour)
            cbar.set_label(label, fontsize=8)
            cbar.ax.tick_params(labelsize=8)
            
            # 绘制最优点（增大标记大小）
            optimal_point = ax_contour.plot(optimal_x, optimal_y, 'g*', 
                                          markersize=15,  # 增大标记大小
                                          label='最优点')[0]
            
            # 初始化梯度下降路径
            path_line = ax_contour.plot([], [], 'r-', linewidth=1, label='梯度下降路径')[0]
            current_point = ax_contour.plot([], [], 'ro', markersize=5, label='当前位置')[0]
            
            # 设置坐标轴范围和标签
            ax_contour.set_xlim(-margin, margin)
            ax_contour.set_ylim(-margin, margin)
            ax_contour.set_xlabel(f'Weight (0-0)', fontsize=8)
            ax_contour.set_ylabel(f'Weight ({layer_idx}-{weight_idx})', fontsize=8)
            
            # 调整图形布局，给图例留更多空间
            plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.15)
            
            # 确保图例显示在合适位置
            ax_contour.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                            fontsize=8, framealpha=0.8)  # 增加图例背景透明度
            
            contour_path = []
            
        # 获取当前选中权重的值
        current_weight = model.layers[layer_idx].weight.data[weight_idx, 0].item()
        current_bias = model.layers[layer_idx].bias.data[weight_idx].item()
        
        # 记录路径
        contour_path.append((current_weight, current_bias))
        
        # 更新梯度下降路径
        if len(contour_path) > 1:
            path_x = [p[0] for p in contour_path]
            path_y = [p[1] for p in contour_path]
            path_line.set_data(path_x, path_y)
            current_point.set_data([path_x[-1]], [path_y[-1]])
        
        # 更新标题，只显示当前选中的权重信息
        title_prefix = "Loss" if display_mode == "loss" else "Gradient"
        ax_contour.set_title(
            f'{title_prefix}等高线图 (Epoch: {epoch}, Loss: {current_loss:.4f})', 
            fontsize=9
        )
        
        # 刷新画布
        fig_contour.canvas.draw()
        fig_contour.canvas.flush_events()  # 确保及时更新
        
    except Exception as e:
        print(f"更新等高线图时出错: {e}")

def calculate_loss_surface(X, Y, model, dataset_x, dataset_y):
    """计算损失函数曲面"""
    Z = np.zeros_like(X)
    train_loss_func_name = loss_function_var.get()
    train_loss_func, _, _ = loss_functions[train_loss_func_name]
    criterion = lambda input, target: train_loss_func(input, target)
    
    # 获取当前选中的层和权重索引
    layer_idx, weight_idx = map(int, selected_weight.split('-'))
    
    # 保存原始权重和偏置
    original_weight = model.layers[layer_idx].weight.data[weight_idx, 0].clone()
    original_bias = model.layers[layer_idx].bias.data[weight_idx].clone()
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # 临时修改权重和偏置
            model.layers[layer_idx].weight.data[weight_idx, 0] = torch.tensor(X[i, j])
            model.layers[layer_idx].bias.data[weight_idx] = torch.tensor(Y[i, j])
            
            # 计算损失
            outputs = model(dataset_x)
            loss = criterion(outputs, dataset_y)
            Z[i, j] = loss.item()
    
    # 恢复原始权重和偏置
    model.layers[layer_idx].weight.data[weight_idx, 0] = original_weight
    model.layers[layer_idx].bias.data[weight_idx] = original_bias
    
    return Z

def calculate_gradient_surface(X, Y, model, dataset_x, dataset_y):
    """计算梯度值曲面"""
    Z = np.zeros_like(X)
    train_loss_func_name = loss_function_var.get()
    train_loss_func, _, _ = loss_functions[train_loss_func_name]
    criterion = lambda input, target: train_loss_func(input, target)
    
    layer_idx, weight_idx = map(int, selected_weight.split('-'))
    original_weight = model.layers[layer_idx].weight.data[weight_idx, 0].clone()
    original_bias = model.layers[layer_idx].bias.data[weight_idx].clone()
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            model.layers[layer_idx].weight.data[weight_idx, 0] = torch.tensor(X[i, j])
            model.layers[layer_idx].bias.data[weight_idx] = torch.tensor(Y[i, j])
            
            # 计算梯度
            outputs = model(dataset_x)
            loss = criterion(outputs, dataset_y)
            loss.backward()
            
            # 获取梯度值
            grad_magnitude = torch.sqrt(
                model.layers[layer_idx].weight.grad[weight_idx, 0]**2 +
                model.layers[layer_idx].bias.grad[weight_idx]**2
            ).item()
            
            Z[i, j] = grad_magnitude
            
            # 清除梯度
            model.zero_grad()
    
    # 恢复原始权重和偏置
    model.layers[layer_idx].weight.data[weight_idx, 0] = original_weight
    model.layers[layer_idx].bias.data[weight_idx] = original_bias
    
    return Z

def on_weight_select(event, model, epoch):
    """
    处理用户选中权重事件
    """
    global selected_weight
    # 获取选中的权重
    selected_weight = event.artist.get_label()
    
    # 解析权重索引并更新输入框
    layer_idx, weight_idx = map(int, selected_weight.split('-'))
    layer_entry.delete(0, tk.END)
    layer_entry.insert(0, str(layer_idx))
    column_entry.delete(0, tk.END)
    column_entry.insert(0, str(weight_idx))
    
    # 立即更新显示
    update_weight_info(epoch, int(epochs_entry.get()), 
                      [param.data.numpy() for param in model.parameters() if len(param.shape) == 2],
                      weight_info_text, model, selected_weight)

def update_weight_info(epoch, epochs, weights, weight_info_text, model, selected_weight=None):
    weight_info_text.delete(1.0, tk.END)  # 清空当前内容
    weight_info_text.insert(tk.END, f"Epoch {epoch + 1}/{epochs}\n")
    
    # 获取用户输入的权重索引
    try:
        user_layer = int(layer_entry.get())
        user_column = int(column_entry.get())
        # 使用用户输入的权重索引
        layer_idx = user_layer
        weight_idx = user_column
    except (ValueError, NameError):
        # 如果用户输入无效或者是通过点击神经网络选择的权重
        if selected_weight:
            layer_idx, weight_idx = map(int, selected_weight.split('-'))
        else:
            layer_idx, weight_idx = 0, 0
    
    # 添加权重信息
    if weights and len(weights) > 0:
        # 只显示选中的权重信息
        if 0 <= layer_idx < len(weights) and 0 <= weight_idx < weights[layer_idx].shape[0]:
            weight_matrix = weights[layer_idx]
            weight_info_text.insert(tk.END, f"第 {layer_idx+1} 层权重矩阵:\n")
            weight_info_text.insert(tk.END, f"权重索引: {layer_idx}-{weight_idx}\n")
            weight_info_text.insert(tk.END, f"优化前权重值: {weight_matrix[weight_idx, 0]:.6f}\n")
            
            # 获取梯度值并初始化 grad 变量
            grad = None
            try:
                grad = model.layers[layer_idx].weight.grad[weight_idx, 0].item()
                weight_info_text.insert(tk.END, f"梯度值: {grad:.6f}\n")
            except:
                weight_info_text.insert(tk.END, "梯度值: 无\n")
            
            # 学习率和更新量
            lr = learning_rate_var.get()
            update_amount = grad * lr if grad is not None else 0  # 修改这里的条件判断
            weight_info_text.insert(tk.END, f"学习率: {lr}\n")
            weight_info_text.insert(tk.END, f"更新量: {update_amount:.10f}\n")
            weight_info_text.insert(tk.END, "-" * 40 + "\n")
        else:
            weight_info_text.insert(tk.END, f"无效的权重索引: {layer_idx}-{weight_idx}\n")
            weight_info_text.insert(tk.END, f"请确保层索引在0到{len(weights)-1}之间，\n")
            if len(weights) > 0:
                weight_info_text.insert(tk.END, f"列索引在0到{weights[0].shape[0]-1}之间\n")
    
    # 自动滚动到最新内容
    weight_info_text.see(tk.END)

# 创建主窗口
root = tk.Tk()
root.title("神经网络训练配置")

# 创建配置参数面板
config_frame = ttk.LabelFrame(root, text="配置参数", padding=10)
config_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# 生成点方式选择
data_gen_var = tk.StringVar(value="linear")  # 默认线性生成
data_gen_label = ttk.Label(config_frame, text="生成点方式:")
data_gen_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
data_gen_menu = ttk.Combobox(config_frame, textvariable=data_gen_var, values=["linear", "random", "vertical", "curve", "custom"], width=12)  # 缩小宽度
data_gen_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

# 创建参数输入框
ttk.Label(config_frame, text="训练次数:").grid(row=1, column=0, sticky="w")
epochs_entry = ttk.Entry(config_frame, width=12)  # 缩小宽度
epochs_entry.grid(row=1, column=1, padx=5, pady=5)
epochs_entry.insert(0, "10000")

# 去掉梯度下降学习率相关变量和UI组件
ttk.Label(config_frame, text="学习率:").grid(row=2, column=0, sticky="w")
# 创建一个框架包含滑块和显示值的标签
lr_frame = ttk.Frame(config_frame)
lr_frame.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

# 学习率变量
learning_rate_var = tk.DoubleVar(value=0.01)  # 修改默认值为0.01更适合梯度下降

# 学习率滑块
lr_slider = ttk.Scale(lr_frame, from_=0.0001, to=0.1, 
                     variable=learning_rate_var,
                     command=lambda v: update_lr_label())
lr_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

# 显示当前学习率的标签
lr_value_label = ttk.Label(lr_frame, text="0.01")
lr_value_label.pack(side=tk.RIGHT, padx=5)

def update_lr_label():
    """更新学习率标签"""
    value = learning_rate_var.get()
    lr_value_label.config(text=f"{value:.4f}")

ttk.Label(config_frame, text="激活函数:").grid(row=3, column=0, sticky="w")
activation_var = tk.StringVar(value="tanh")
activation_menu = ttk.Combobox(config_frame, textvariable=activation_var, values=["relu", "sigmoid", "tanh"], width=12)  # 缩小宽度
activation_menu.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

# 添加损失函数选择下拉框
ttk.Label(config_frame, text="损失函数:").grid(row=7, column=0, sticky="w")
loss_function_var = tk.StringVar(value="MSE损失")
loss_function_menu = ttk.Combobox(config_frame, textvariable=loss_function_var, 
                                    values=list(loss_functions.keys()), width=15)
loss_function_menu.grid(row=7, column=1, padx=5, pady=5, sticky="ew")
loss_function_menu.bind("<<ComboboxSelected>>", lambda e: change_loss_function())

# 添加优化器选择下拉框
ttk.Label(config_frame, text="优化器:").grid(row=9, column=0, sticky="w")
optimizer_var = tk.StringVar(value="Adam")
optimizer_menu = ttk.Combobox(config_frame, textvariable=optimizer_var, 
                             values=list(optimizers.keys()), width=15)
optimizer_menu.grid(row=9, column=1, padx=5, pady=5, sticky="ew")

# 添加在config_frame中的参数输入区域
ttk.Label(config_frame, text="更新频率:").grid(row=8, column=0, sticky="w")
update_frequency_entry = ttk.Entry(config_frame, width=5)
update_frequency_entry.grid(row=8, column=1, padx=5, pady=5)
update_frequency_entry.insert(0, "10")  # 默认每10个epoch更新一次

# 创建按钮面板
button_frame = ttk.Frame(config_frame)
button_frame.grid(row=4, column=0, columnspan=4, pady=10)

start_button = ttk.Button(button_frame, text="启动训练", command=start_training)
start_button.grid(row=0, column=0, padx=5)

add_layer_button = ttk.Button(button_frame, text="添加层", command=add_layer)
add_layer_button.grid(row=0, column=1, padx=5)

# 创建层配置面板
layer_frame = ttk.Frame(config_frame)
layer_frame.grid(row=5, column=0, columnspan=4, pady=10, sticky="w")

# 创建右侧主面板
right_frame = ttk.Frame(root)
right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# 创建右侧上部面板（包含三种曲线和自定义面板）
top_right_frame = ttk.Frame(right_frame)
top_right_frame.grid(row=0, column=0, sticky="nsew")

# 创建三种曲线面板
plot_frame = ttk.LabelFrame(top_right_frame, text="训练过程", padding=10)
plot_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

# 创建三种曲线绘图区域
fig = plt.figure(figsize=(12, 9.5))  # 增加整体高度
gs = fig.add_gridspec(3, 1, height_ratios=[1.8, 2, 2])  # 增加权重图的高度比例

# 创建子图
ax_loss = fig.add_subplot(gs[0, 0])  # Loss曲线
ax_loss.set_title("Loss Curve")
ax_learning = fig.add_subplot(gs[1, 0])  # 学习曲线
ax_learning.set_title("Learning Curve")
ax_neural_network = fig.add_subplot(gs[2, 0])  # 神经网络结构
ax_neural_network.set_title("Neural Network Layers")

canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# 创建自定义曲线面板
custom_curve_frame = ttk.LabelFrame(top_right_frame, text="Custom Curve", padding=10)
custom_curve_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

# 创建自定义曲线绘图区域
fig_custom = plt.figure(figsize=(6, 6))
ax_custom_curve = fig_custom.add_subplot(111)
ax_custom_curve.set_title("Draw Custom Curve")
ax_custom_curve.grid(True)

canvas_custom = FigureCanvasTkAgg(fig_custom, master=custom_curve_frame)
canvas_custom.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# 创建等高线面板
contour_frame = ttk.LabelFrame(right_frame, text="梯度下降轨迹", padding=5)
contour_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")  

# 创建等高线绘图区域
fig_contour = plt.figure(figsize=(12, 5))

# 清除之前的颜色条
for ax in fig_contour.axes:
    if ax.get_label() == 'colorbar':  # 通过标签识别颜色条
        ax.remove()  # 删除颜色条轴

ax_contour = fig_contour.add_subplot(111)
ax_contour.set_title("梯度下降轨迹", fontsize=9)
ax_contour.grid(True)

# 调整图形边距
plt.subplots_adjust(left=0.1, right=0.85,  # 调整right值给颜色条留出空间
                   top=0.9, bottom=0.15)

canvas_contour = FigureCanvasTkAgg(fig_contour, master=contour_frame)
canvas_contour.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# 设置窗口布局权重
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

right_frame.grid_rowconfigure(0, weight=2)  # 上半部分占比更大
right_frame.grid_rowconfigure(1, weight=1)  # 等高线图占比更小
right_frame.grid_columnconfigure(0, weight=1)

top_right_frame.grid_rowconfigure(0, weight=1)
top_right_frame.grid_columnconfigure(0, weight=1)
top_right_frame.grid_columnconfigure(1, weight=1)

# 绑定窗口关闭事件
root.protocol("WM_DELETE_WINDOW", on_closing)

# 默认增加3层神经网络
for _ in range(3):
    add_layer()

# 绑定鼠标事件
custom_curve_points = []  # 存储用户绘制的点
is_drawing = False  # 标记是否正在绘制

def on_mouse_press(event):
    """
    处理鼠标按下事件，开始绘制
    """
    global is_drawing
    if event.inaxes == ax_custom_curve and event.button == 1:  # 左键按下
        is_drawing = True
        custom_curve_points.append((event.xdata, event.ydata))
        ax_custom_curve.plot(event.xdata, event.ydata, 'ko')  # 绘制黑点
        canvas_custom.draw()

def on_mouse_move(event):
    """
    处理鼠标移动事件，连续绘制
    """
    if is_drawing and event.inaxes == ax_custom_curve:  # 左键按下且鼠标在绘图区域内
        custom_curve_points.append((event.xdata, event.ydata))
        ax_custom_curve.clear()  # 清除之前的绘图
        ax_custom_curve.set_xlim(0, 1)
        ax_custom_curve.set_ylim(0, 1)
        ax_custom_curve.set_title("Custom Curve")
        ax_custom_curve.grid(True)
        # 绘制所有点并连接成曲线
        x_vals = [p[0] for p in custom_curve_points]
        y_vals = [p[1] for p in custom_curve_points]
        ax_custom_curve.plot(x_vals, y_vals, 'k-')  # 绘制黑色曲线
        ax_custom_curve.plot(x_vals, y_vals, 'ko')  # 绘制黑点
        canvas_custom.draw()

def on_mouse_release(event):
    """
    处理鼠标释放事件，结束绘制
    """
    global is_drawing
    if event.button == 1:  # 左键释放
        is_drawing = False

canvas_custom.mpl_connect('button_press_event', on_mouse_press)
canvas_custom.mpl_connect('motion_notify_event', on_mouse_move)
canvas_custom.mpl_connect('button_release_event', on_mouse_release)

# 将权重信息相关内容移动到配置参数面板底部
weight_info_frame = ttk.Frame(config_frame)
weight_info_frame.grid(row=10, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

# 创建顶部控制面板
control_frame = ttk.Frame(weight_info_frame)
control_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

# 显示公式
formula_label = ttk.Label(control_frame, text="公式: W_new = W_old - η * (∂E/∂w)")
formula_label.pack(side=tk.LEFT, padx=5)

# 创建权重选择框架
weight_select_frame = ttk.Frame(control_frame)
weight_select_frame.pack(side=tk.RIGHT, padx=5)

# 添加层选择
ttk.Label(weight_select_frame, text="层:").pack(side=tk.LEFT)
layer_entry = ttk.Entry(weight_select_frame, width=3)
layer_entry.pack(side=tk.LEFT, padx=(0, 5))
layer_entry.insert(0, "0")

# 添加列选择
ttk.Label(weight_select_frame, text="列:").pack(side=tk.LEFT)
column_entry = ttk.Entry(weight_select_frame, width=3)
column_entry.pack(side=tk.LEFT, padx=(0, 5))
column_entry.insert(0, "0")

# 修改权重选择处理函数
def update_selected_weight(*args):
    """更新选中的权重"""
    global selected_weight
    try:
        layer = int(layer_entry.get())
        column = int(column_entry.get())
    except ValueError:
        layer = 0
        column = 0
        layer_entry.delete(0, tk.END)
        layer_entry.insert(0, "0")
        column_entry.delete(0, tk.END)
        column_entry.insert(0, "0")
    
    selected_weight = f"{layer}-{column}"

# 绑定输入框变化事件
layer_entry.bind('<KeyRelease>', update_selected_weight)
column_entry.bind('<KeyRelease>', update_selected_weight)

# 创建打印框
weight_info_text = tk.Text(weight_info_frame, height=20, width=40)
weight_info_text.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

# 添加滚动条
scrollbar = ttk.Scrollbar(weight_info_frame, orient="vertical", command=weight_info_text.yview)
scrollbar.grid(row=1, column=1, sticky="ns")
weight_info_text.configure(yscrollcommand=scrollbar.set)

# 在创建损失函数选择下拉框后，添加等高线显示模式选择
ttk.Label(config_frame, text="等高线显示:").grid(row=8, column=0, sticky="w")
contour_mode_var = tk.StringVar(value="loss")
contour_mode_menu = ttk.Combobox(config_frame, textvariable=contour_mode_var, 
                                values=["loss", "gradient"], width=15)
contour_mode_menu.grid(row=8, column=1, padx=5, pady=5, sticky="ew")

# 运行主循环
root.mainloop()
