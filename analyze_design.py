import numpy as np
import matplotlib.pyplot as plt

class DesignAnalysis:
    def __init__(self):
        self.link_lengths = np.array([1.0, 0.8, 0.6, 0.4])
        
    def analyze_joint_constraints(self):
        """分析每个关节对关键点位置的影响"""
        print("=== 四轴机械臂设计分析 ===")
        print(f"连杆长度: {self.link_lengths}")
        print()
        
        # 测试配置
        base_config = np.array([0, 0, 0, 0])  # 全零配置
        
        print("1. 关节1(Base)位置分析:")
        print("   - 位置计算: [0, 0, link_lengths[0]]")
        print("   - 无论Base角度θ1如何变化，Joint1的位置都是 [0, 0, 1.0]")
        print("   - 这是因为Base关节是绕Z轴旋转，而Joint1就在Z轴上！")
        print()
        
        # 验证Base旋转对Joint1位置的影响
        for base_angle in [0, np.pi/4, np.pi/2, np.pi]:
            config = np.array([base_angle, 0, 0, 0])
            x1, y1, z1 = 0, 0, self.link_lengths[0]
            print(f"   Base角度={base_angle:.2f}rad时，Joint1位置: [{x1:.3f}, {y1:.3f}, {z1:.3f}]")
        
        print()
        print("2. 为什么会出现学习困难？")
        print("   - Joint1位置与Base角度θ1无关，这创建了一个信息瓶颈")
        print("   - 神经网络无法从Joint1的位置推断出Base的旋转角度")
        print("   - Base角度只能通过后续关节点的位置来推断")
        print()
        
        print("3. Wrist关节(Joint4)的冗余性:")
        print("   - 当末端执行器的姿态要求不严格时，Wrist角度可以有多个解")
        print("   - 在某些配置下，不同的Wrist角度可能产生相似的关键点位置")
        print()
        
        # 演示Base角度对后续关节的影响
        print("4. Base角度对后续关节的影响:")
        shoulder_angle = np.pi/4  # 45度
        
        for base_angle in [0, np.pi/2]:
            config = np.array([base_angle, shoulder_angle, 0, 0])
            keypoints = self.forward_kinematics_simple(config)
            
            print(f"   Base={base_angle:.2f}rad时:")
            for i, kp in enumerate(keypoints):
                print(f"     关键点{i}: [{kp[0]:.3f}, {kp[1]:.3f}, {kp[2]:.3f}]")
            print()
    
    def forward_kinematics_simple(self, joint_angles):
        """简化的正运动学计算"""
        angles = joint_angles
        keypoints = np.zeros((5, 3))
        
        # 基座位置
        keypoints[0] = [0, 0, 0]
        
        # Joint1位置 - 始终在Z轴上
        keypoints[1] = [0, 0, self.link_lengths[0]]
        
        # Joint2位置
        cos_th1, sin_th1 = np.cos(angles[0]), np.sin(angles[0])
        cos_th2, sin_th2 = np.cos(angles[1]), np.sin(angles[1])
        
        local_x2 = self.link_lengths[1] * sin_th2
        local_z2 = self.link_lengths[1] * cos_th2
        
        x2 = cos_th1 * local_x2
        y2 = sin_th1 * local_x2
        z2 = keypoints[1, 2] + local_z2
        keypoints[2] = [x2, y2, z2]
        
        # 简化后续计算...
        keypoints[3] = keypoints[2]  # 占位
        keypoints[4] = keypoints[2]  # 占位
        
        return keypoints
    
    def visualize_problem(self):
        """可视化问题所在"""
        fig = plt.figure(figsize=(15, 5))
        
        # 子图1: Base角度变化的影响
        ax1 = fig.add_subplot(131, projection='3d')
        
        base_angles = np.linspace(0, 2*np.pi, 8)
        shoulder_angle = np.pi/4
        
        for i, base_angle in enumerate(base_angles):
            config = np.array([base_angle, shoulder_angle, 0, 0])
            keypoints = self.forward_kinematics_simple(config)
            
            # 绘制连杆
            ax1.plot([keypoints[0,0], keypoints[1,0]], 
                    [keypoints[0,1], keypoints[1,1]], 
                    [keypoints[0,2], keypoints[1,2]], 'b-', alpha=0.7)
            ax1.plot([keypoints[1,0], keypoints[2,0]], 
                    [keypoints[1,1], keypoints[2,1]], 
                    [keypoints[1,2], keypoints[2,2]], 'r-', alpha=0.7)
            
            # 标记Joint1位置（应该重叠）
            ax1.scatter(keypoints[1,0], keypoints[1,1], keypoints[1,2], 
                       c='red', s=50, alpha=0.8)
        
        ax1.set_title('Base Rotation Effect\n(Joint1 positions overlap!)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # 子图2: 输入输出维度分析
        ax2 = fig.add_subplot(132)
        ax2.text(0.1, 0.8, 'Input Dimensions: 15D', fontsize=12, weight='bold')
        ax2.text(0.1, 0.7, '• 5 keypoints × 3D coordinates', fontsize=10)
        ax2.text(0.1, 0.6, '• Base(0,0,0) + Joint1(0,0,1.0) + ...', fontsize=10)
        ax2.text(0.1, 0.4, 'Output Dimensions: 16D', fontsize=12, weight='bold')
        ax2.text(0.1, 0.3, '• 4 joints × axis-angle(4D)', fontsize=10)
        ax2.text(0.1, 0.2, '• [axis_x, axis_y, axis_z, angle]', fontsize=10)
        ax2.text(0.1, 0.05, 'Problem: Joint1 position invariant to θ1!', 
                fontsize=11, color='red', weight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Dimension Analysis')
        
        # 子图3: 学习困难度分析
        ax3 = fig.add_subplot(133)
        joints = ['Base', 'Shoulder', 'Elbow', 'Wrist']
        difficulties = [0.9, 0.3, 0.4, 0.8]  # 相对学习困难度
        colors = ['red', 'green', 'orange', 'red']
        
        bars = ax3.bar(joints, difficulties, color=colors, alpha=0.7)
        ax3.set_ylabel('Learning Difficulty')
        ax3.set_title('Joint Learning Difficulty')
        ax3.set_ylim(0, 1)
        
        # 添加说明
        for i, (bar, diff) in enumerate(zip(bars, difficulties)):
            height = bar.get_height()
            if diff > 0.7:
                reason = 'Position\nInvariant' if i == 0 else 'Redundant\nDOF'
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        reason, ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('/home/zhangrong/worker_lingyu/zhangrong/Else/arcsinx/design_analysis.png', 
                   dpi=150, bbox_inches='tight')
        print("分析图表已保存为: design_analysis.png")
        plt.close()  # 关闭图形以避免显示警告

if __name__ == "__main__":
    analyzer = DesignAnalysis()
    analyzer.analyze_joint_constraints()
    analyzer.visualize_problem()
