import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False

def batch_rodrigues(rot_vecs):
    """
    Convert batch of axis-angle representations    # 测试2: 随机小幅度姿态
    print("\nTest 2 - Small random poses:")
    random_pose = torch.randn(batch_size, 16, 3, device=device) * 0.2  # 小幅度随机
    result_random = simple_forward_kinematics(random_pose, rest_joints_tensor, device)
    
    print(f"Random pose range: [{random_pose.min().item():.3f}, {random_pose.max().item():.3f}]")
    print(f"Result shape: {result_random.shape}")
    
    # 可视化随机姿态
    for i in range(min(2, batch_size)):
        visualize_hand(result_random[i], f"Random Pose {i+1}")
    
    # 测试2.5: 大范围随机姿态
    print("\nTest 2.5 - Large range random poses:")
    large_random_pose = torch.randn(batch_size, 16, 3, device=device) * 1.5  # 大幅度随机
    result_large_random = simple_forward_kinematics(large_random_pose, rest_joints_tensor, device)
    
    print(f"Large random pose range: [{large_random_pose.min().item():.3f}, {large_random_pose.max().item():.3f}]")
    print(f"Result shape: {result_large_random.shape}")
    
    # 可视化大范围随机姿态
    for i in range(min(2, batch_size)):
        visualize_hand(result_large_random[i], f"Large Random Pose {i+1}")n matrices using Rodrigues' formula
    
    Args:
        rot_vecs: [B, 3] axis-angle vectors
    Returns:
        rot_mats: [B, 3, 3] rotation matrices
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype
    
    # Get angle (magnitude of rotation vector)
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)  # [B, 1]
    
    # Get rotation axis (normalized)
    rot_dir = rot_vecs / angle  # [B, 3]
    
    cos = torch.cos(angle)  # [B, 1]
    sin = torch.sin(angle)  # [B, 1]
    
    # Outer product
    outer = torch.bmm(rot_dir.view(batch_size, 3, 1), rot_dir.view(batch_size, 1, 3))  # [B, 3, 3]
    
    # Cross product matrix
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
    K[:, 0, 1] = -rot_dir[:, 2]
    K[:, 0, 2] = rot_dir[:, 1]
    K[:, 1, 0] = rot_dir[:, 2]
    K[:, 1, 2] = -rot_dir[:, 0]
    K[:, 2, 0] = -rot_dir[:, 1]
    K[:, 2, 1] = rot_dir[:, 0]
    
    # Identity matrix
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K^2
    # But K^2 = outer - I, so: R = I + sin(θ)K + (1-cos(θ))(outer - I)
    # R = cos(θ)I + sin(θ)K + (1-cos(θ))outer
    rot_mat = cos.unsqueeze(-1) * ident + sin.unsqueeze(-1) * K + (1 - cos).unsqueeze(-1) * outer
    
    return rot_mat

def simple_forward_kinematics(pose_param_16x3: torch.Tensor, rest_joints: torch.Tensor, device=None):
    """
    MANO hand forward kinematics with 16D axis-angle parameters
    
    改动说明：
      - 把 16 维按 (root, per-finger: MCP, PIP, DIP) 映射到 joint indices:
          root -> 0
          thumb: 1,2,3
          index: 5,6,7
          middle:9,10,11
          ring:13,14,15
          little:17,18,19
      - 这样 pose[:,1,:] 会控制 joint 1（MCP），满足你的预期。
      - 位置更新使用 parent 的 global rotation： pos[j] = pos[parent] + parent_rot @ (rest[j]-rest[parent])
      - 全局旋转传播： global_rot[j] = parent_rot @ local_rot[j]
      
    Args:
      pose_param_16x3: [B,16,3] - 16维轴角参数
      rest_joints: [B,21,3] - 绝对rest坐标，函数会用差值得到相对偏移
      
    Returns:
      joint_positions: [B,21,3] - 21个关键点的3D坐标
    """
    if device is None:
        device = pose_param_16x3.device
    B = pose_param_16x3.shape[0]

    # 手部骨骼连接关系 (parent -> child)
    bones = [
        [0, 1], [1, 2], [2, 3], [3, 4],      # 拇指
        [0, 5], [5, 6], [6, 7], [7, 8],      # 食指
        [0, 9], [9, 10], [10, 11], [11, 12], # 中指
        [0, 13], [13, 14], [14, 15], [15, 16], # 无名指
        [0, 17], [17, 18], [18, 19], [19, 20]  # 小指
    ]
    
    # 构建父节点索引
    parent_idx = [-1] * 21  # -1 表示根节点
    for p, c in bones:
        parent_idx[c] = p

    # 关键映射：16维参数到21个关节的映射（已改为 MCP 可动）
    joint_to_param_idx = {
        0: 0,   # global root (腕关节)

        # 拇指（MCP, PIP, DIP）
        1: 1, 2: 2, 3: 3,
        # 食指
        5: 4, 6: 5, 7: 6,
        # 中指
        9: 7, 10: 8, 11: 9,
        # 无名指
        13: 10, 14: 11, 15: 12,
        # 小指
        17: 13, 18: 14, 19: 15,
        
        # 关节 4,8,12,16,20 (指尖) 保持不映射（无参数，跟随父关节）
    }

    # 初始化输出
    joint_positions = torch.zeros(B, 21, 3, device=device, dtype=rest_joints.dtype)
    transforms = [None] * 21  # 每项存 Bx3x3 的 global rotation

    # 根关节处理
    if 0 in joint_to_param_idx:
        root_local = pose_param_16x3[:, joint_to_param_idx[0], :]  # [B,3]
        root_rot = batch_rodrigues(root_local)  # [B,3,3]
    else:
        root_rot = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)

    # 根位置：旋转 rest_joints[0]
    joint_positions[:, 0, :] = torch.bmm(root_rot, rest_joints[:, 0, :].unsqueeze(-1)).squeeze(-1)
    transforms[0] = root_rot

    # 逐关节递归计算
    for j in range(1, 21):
        parent = parent_idx[j]
        parent_pos = joint_positions[:, parent, :]       # [B,3]
        parent_rot = transforms[parent]                  # [B,3,3]

        # 计算相对位置向量
        rel = rest_joints[:, j, :] - rest_joints[:, parent, :]  # [B,3]

        # 获取局部旋转
        if j in joint_to_param_idx:
            local_axis_angle = pose_param_16x3[:, joint_to_param_idx[j], :]  # [B,3]
            local_rot = batch_rodrigues(local_axis_angle)  # [B,3,3]
        else:
            # 指尖关节无参数，使用单位矩阵
            local_rot = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)

        # 计算全局旋转（用于子关节）
        global_rot = torch.bmm(parent_rot, local_rot)  # [B,3,3]
        transforms[j] = global_rot

        # 关节位置只受父关节全局旋转影响
        rotated_rel = torch.bmm(parent_rot, rel.unsqueeze(-1)).squeeze(-1)  # [B,3]
        joint_positions[:, j, :] = parent_pos + rotated_rel

    return joint_positions

def load_rest_hand_pose(file_path="../mano_hand_kpts/rest_hand_kpts_flat.xyz"):
    """
    从文件加载MANO手部静息姿态（21个关键点）
    
    Args:
        file_path: rest pose文件路径
        
    Returns:
        rest_joints: [21, 3] numpy数组，包含21个关键点的3D坐标
        
    关节顺序：
    0: 腕关节 (wrist)
    1-4: 拇指 (thumb): MCP, PIP, DIP, tip
    5-8: 食指 (index): MCP, PIP, DIP, tip  
    9-12: 中指 (middle): MCP, PIP, DIP, tip
    13-16: 无名指 (ring): MCP, PIP, DIP, tip
    17-20: 小指 (little): MCP, PIP, DIP, tip
    """
    try:
        # 读取xyz文件
        rest_joints = np.loadtxt(file_path)
        
        # 验证形状
        if rest_joints.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {rest_joints.shape}")
        
        print(f"Successfully loaded rest pose from {file_path}")
        print(f"Rest pose shape: {rest_joints.shape}")
        
        return rest_joints
        
    except Exception as e:
        print(f"Error loading rest pose from {file_path}: {e}")
        raise

def visualize_hand(joint_positions, title="Hand Pose", show_joint_labels=True):
    """
    可视化手部姿态
    
    Args:
        joint_positions: [21, 3] numpy array or [B, 21, 3] tensor (如果是batch，取第一个)
        title: 图像标题
        show_joint_labels: 是否显示关节标签
    """
    # 处理输入格式
    if isinstance(joint_positions, torch.Tensor):
        joint_positions = joint_positions.detach().cpu().numpy()
    
    if joint_positions.ndim == 3:
        joint_positions = joint_positions[0]  # 取第一个样本
    
    # 手部骨骼连接关系
    bones = [
        [0, 1], [1, 2], [2, 3], [3, 4],      # 拇指
        [0, 5], [5, 6], [6, 7], [7, 8],      # 食指
        [0, 9], [9, 10], [10, 11], [11, 12], # 中指
        [0, 13], [13, 14], [14, 15], [15, 16], # 无名指
        [0, 17], [17, 18], [18, 19], [19, 20]  # 小指
    ]
    
    # 手指颜色
    finger_colors = {
        'thumb': 'red',
        'index': 'blue', 
        'middle': 'green',
        'ring': 'orange',
        'little': 'purple'
    }
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制骨骼连接
    for i, (start, end) in enumerate(bones):
        # 确定手指类型和颜色
        if i < 4:  # 拇指
            color = finger_colors['thumb']
            finger_name = 'thumb'
        elif i < 8:  # 食指
            color = finger_colors['index']
            finger_name = 'index'
        elif i < 12:  # 中指
            color = finger_colors['middle']
            finger_name = 'middle'
        elif i < 16:  # 无名指
            color = finger_colors['ring']
            finger_name = 'ring'
        else:  # 小指
            color = finger_colors['little']
            finger_name = 'little'
        
        # 绘制骨骼
        ax.plot([joint_positions[start, 0], joint_positions[end, 0]],
                [joint_positions[start, 1], joint_positions[end, 1]], 
                [joint_positions[start, 2], joint_positions[end, 2]],
                color=color, linewidth=3, alpha=0.8, 
                label=finger_name if i in [0, 4, 8, 12, 16] else "")
    
    # 绘制关节点
    ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2],
               c='black', s=50, alpha=0.8, edgecolors='white', linewidth=1)
    
    # 添加关节标签
    if show_joint_labels:
        joint_names = [
            'Wrist',
            'T_MCP', 'T_PIP', 'T_DIP', 'T_tip',     # 拇指
            'I_MCP', 'I_PIP', 'I_DIP', 'I_tip',     # 食指
            'M_MCP', 'M_PIP', 'M_DIP', 'M_tip',     # 中指
            'R_MCP', 'R_PIP', 'R_DIP', 'R_tip',     # 无名指
            'L_MCP', 'L_PIP', 'L_DIP', 'L_tip'      # 小指
        ]
        
        for i, (pos, name) in enumerate(zip(joint_positions, joint_names)):
            ax.text(pos[0], pos[1], pos[2], f'  {name}', fontsize=8)
    
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 设置相等的坐标轴比例
    max_range = np.max(np.abs(joint_positions)) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    # 添加图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def test_forward_kinematics():
    """
    测试正运动学功能
    """
    print("=== Testing MANO Hand Forward Kinematics ===")
    
    # 加载静息姿态
    rest_joints = load_rest_hand_pose()
    print(f"Rest pose shape: {rest_joints.shape}")
    
    # 可视化静息姿态
    print("\n1. Visualizing rest pose...")
    visualize_hand(rest_joints, "Rest Hand Pose")
    
    # 创建一些测试姿态
    batch_size = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 转换为 torch tensor
    rest_joints_tensor = torch.FloatTensor(rest_joints).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    
    print(f"\n2. Testing with batch size {batch_size} on device: {device}")
    
    # 测试1: 零姿态（应该等于静息姿态）
    zero_pose = torch.zeros(batch_size, 16, 3, device=device)
    result_zero = simple_forward_kinematics(zero_pose, rest_joints_tensor, device)
    
    print("Test 1 - Zero pose:")
    print(f"Input pose shape: {zero_pose.shape}")
    print(f"Output shape: {result_zero.shape}")
    
    # 检查零姿态是否等于静息姿态
    diff = torch.abs(result_zero[0] - rest_joints_tensor[0]).max().item()
    print(f"Max difference from rest pose: {diff:.6f}")
    
    if diff < 1e-5:
        print("✓ Zero pose correctly produces rest pose")
    else:
        print("✗ Zero pose does not match rest pose")
    
    # 可视化零姿态结果
    visualize_hand(result_zero[0], "Zero Pose Result")
    
    # 测试2: 随机小幅度姿态
    print("\nTest 2 - Small random poses:")
    random_pose = torch.randn(batch_size, 16, 3, device=device) * 0.2  # 小幅度随机
    result_random = simple_forward_kinematics(random_pose, rest_joints_tensor, device)
    
    print(f"Random pose range: [{random_pose.min().item():.3f}, {random_pose.max().item():.3f}]")
    print(f"Result shape: {result_random.shape}")
    
    # 可视化随机姿态
    for i in range(min(2, batch_size)):
        visualize_hand(result_random[i], f"Random Pose {i+1}")
    
    # 测试2.5: 大范围随机姿态
    print("\nTest 2.5 - Large range random poses:")
    large_random_pose = torch.randn(batch_size, 16, 3, device=device) * 1.5  # 大幅度随机
    result_large_random = simple_forward_kinematics(large_random_pose, rest_joints_tensor, device)
    
    print(f"Large random pose range: [{large_random_pose.min().item():.3f}, {large_random_pose.max().item():.3f}]")
    print(f"Result shape: {result_large_random.shape}")
    
    # 可视化大范围随机姿态
    for i in range(min(2, batch_size)):
        visualize_hand(result_large_random[i], f"Large Random Pose {i+1}")
    
    # 测试3: 特定手势（握拳）
    print("\nTest 3 - Fist gesture:")
    fist_pose = torch.zeros(batch_size, 16, 3, device=device)
    
    # 设置握拳姿态 (所有手指弯曲)
    curl_angle = 1.2  # 约70度
    for finger_start in [1, 4, 7, 10, 13]:  # 每个手指的起始参数索引
        for joint_offset in [0, 1, 2]:  # MCP, PIP, DIP
            if finger_start + joint_offset < 16:
                fist_pose[:, finger_start + joint_offset, 0] = curl_angle  # 绕X轴弯曲
    
    result_fist = simple_forward_kinematics(fist_pose, rest_joints_tensor, device)
    visualize_hand(result_fist[0], "Fist Gesture")
    
    print("\n=== Forward Kinematics Test Completed ===")
    
    return rest_joints, result_zero, result_random, result_large_random, result_fist

if __name__ == "__main__":
    print("MANO Hand Forward Kinematics Implementation")
    print("==========================================")
    
    # 运行测试
    rest_joints, zero_result, random_result, large_random_result, fist_result = test_forward_kinematics()
    
    print("\nImplementation Summary:")
    print("- 21 joint hand model (MANO-style)")
    print("- 16D axis-angle parameter input")
    print("- Hierarchical forward kinematics")
    print("- 3D visualization with bone connections")
    print("- Batch processing support")
    print("\nNext steps: Generate training data and build MLP model")
