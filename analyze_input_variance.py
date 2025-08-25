import numpy as np

class InputVarianceAnalysis:
    def __init__(self):
        self.link_lengths = np.array([1.0, 0.8, 0.6, 0.4])
        
    def forward_kinematics(self, joint_angles):
        """正运动学计算"""
        if joint_angles.ndim == 1:
            joint_angles = joint_angles.reshape(1, -1)
        
        batch_size = joint_angles.shape[0]
        keypoints = np.zeros((batch_size, 5, 3))
        
        for i in range(batch_size):
            angles = joint_angles[i]
            
            # 基座位置 (固定)
            keypoints[i, 0] = [0, 0, 0]
            
            # Joint1位置 (固定)
            keypoints[i, 1] = [0, 0, self.link_lengths[0]]
            
            # Joint2位置 (变化)
            cos_th1, sin_th1 = np.cos(angles[0]), np.sin(angles[0])
            cos_th2, sin_th2 = np.cos(angles[1]), np.sin(angles[1])
            
            local_x2 = self.link_lengths[1] * sin_th2
            local_z2 = self.link_lengths[1] * cos_th2
            
            x2 = cos_th1 * local_x2
            y2 = sin_th1 * local_x2
            z2 = keypoints[i, 1, 2] + local_z2
            keypoints[i, 2] = [x2, y2, z2]
            
            # Joint3位置 (变化)
            cos_th3, sin_th3 = np.cos(angles[2]), np.sin(angles[2])
            
            local_x3 = self.link_lengths[2] * (sin_th2 * cos_th3 + cos_th2 * sin_th3)
            local_z3 = self.link_lengths[2] * (cos_th2 * cos_th3 - sin_th2 * sin_th3)
            
            x3 = x2 + cos_th1 * local_x3
            y3 = y2 + sin_th1 * local_x3
            z3 = z2 + local_z3
            keypoints[i, 3] = [x3, y3, z3]
            
            # Joint4位置 (变化)
            total_pitch = angles[1] + angles[2] + angles[3]
            cos_total, sin_total = np.cos(total_pitch), np.sin(total_pitch)
            
            local_x4 = self.link_lengths[3] * sin_total
            local_z4 = self.link_lengths[3] * cos_total
            
            x4 = x3 + cos_th1 * local_x4
            y4 = y3 + sin_th1 * local_x4
            z4 = z3 + local_z4
            keypoints[i, 4] = [x4, y4, z4]
        
        return keypoints
    
    def analyze_input_variance(self):
        """分析输入的15维中哪些是变化的"""
        print("=== 输入变化性分析 ===")
        print("输入维度：15维 = 5个关键点 × 3D坐标")
        print()
        
        # 生成多组测试数据
        n_samples = 1000
        test_angles = np.random.uniform(-np.pi, np.pi, (n_samples, 4))
        keypoints = self.forward_kinematics(test_angles)
        
        # 展平为15维输入向量
        input_vectors = keypoints.reshape(n_samples, -1)
        
        print("各维度的变化情况：")
        print("维度编号 | 关键点 | 坐标轴 | 最小值   | 最大值   | 标准差   | 状态")
        print("-" * 70)
        
        dim_names = []
        for kp_idx in range(5):
            for coord_idx, coord_name in enumerate(['X', 'Y', 'Z']):
                dim_names.append(f"KP{kp_idx}_{coord_name}")
        
        for dim in range(15):
            values = input_vectors[:, dim]
            min_val = np.min(values)
            max_val = np.max(values)
            std_val = np.std(values)
            
            # 判断是否为固定值
            if std_val < 1e-10:
                status = "固定"
            elif std_val < 0.1:
                status = "几乎不变"
            else:
                status = "变化"
            
            kp_idx = dim // 3
            coord_idx = dim % 3
            coord_name = ['X', 'Y', 'Z'][coord_idx]
            
            print(f"{dim:^6} | KP{kp_idx}   | {coord_name:^4} | {min_val:^8.3f} | {max_val:^8.3f} | {std_val:^8.3f} | {status}")
        
        print()
        self.analyze_information_content(input_vectors)
    
    def analyze_information_content(self, input_vectors):
        """分析信息含量"""
        print("=== 信息含量分析 ===")
        
        # 计算每个维度的信息熵（简化版）
        entropies = []
        for dim in range(15):
            values = input_vectors[:, dim]
            # 将连续值离散化计算熵
            hist, _ = np.histogram(values, bins=50)
            hist = hist + 1e-10  # 避免log(0)
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob + 1e-10))
            entropies.append(entropy)
        
        print("各维度信息熵：")
        print("维度 | 关键点 | 信息熵 | 重要性")
        print("-" * 35)
        
        for dim in range(15):
            kp_idx = dim // 3
            coord_name = ['X', 'Y', 'Z'][dim % 3]
            entropy = entropies[dim]
            
            if entropy < 1:
                importance = "很低"
            elif entropy < 3:
                importance = "低"
            elif entropy < 5:
                importance = "中等"
            else:
                importance = "高"
            
            print(f"{dim:^4} | KP{kp_idx}_{coord_name} | {entropy:^6.2f} | {importance}")
        
        print()
        self.summarize_findings()
    
    def summarize_findings(self):
        """总结发现"""
        print("=== 总结 ===")
        print("固定维度（不变的输入）：")
        print("  - 维度 0: KP0_X = 0.000 (基座X坐标)")
        print("  - 维度 1: KP0_Y = 0.000 (基座Y坐标)")  
        print("  - 维度 2: KP0_Z = 0.000 (基座Z坐标)")
        print("  - 维度 3: KP1_X = 0.000 (Joint1 X坐标)")
        print("  - 维度 4: KP1_Y = 0.000 (Joint1 Y坐标)")
        print("  - 维度 5: KP1_Z = 1.000 (Joint1 Z坐标)")
        print()
        print("变化维度（有用的输入）：")
        print("  - 维度 6-8: KP2的X,Y,Z坐标 (受θ1,θ2影响)")
        print("  - 维度 9-11: KP3的X,Y,Z坐标 (受θ1,θ2,θ3影响)")
        print("  - 维度 12-14: KP4的X,Y,Z坐标 (受θ1,θ2,θ3,θ4影响)")
        print()
        print("关键发现：")
        print("  • 15维输入中只有9维是真正变化的")
        print("  • 6维是固定的，不包含任何信息")
        print("  • 这解释了为什么网络难以学习某些关节角度")
        print("  • Base角度θ1的信息只能从后续关节点的相对位置推断")

if __name__ == "__main__":
    analyzer = InputVarianceAnalysis()
    analyzer.analyze_input_variance()
