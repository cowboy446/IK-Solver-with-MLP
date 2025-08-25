import numpy as np
import matplotlib.pyplot as plt

class DesignAnalysis:
    def __init__(self):
        self.link_lengths = np.array([1.0, 0.8, 0.6, 0.4])
        
    def analyze_joint_constraints(self):
        """Analyze impact of each joint on keypoint positions"""
        print("=== Four-axis robot arm design analysis ===")
        print(f"Link lengths: {self.link_lengths}")
        print()
        
        base_config = np.array([0, 0, 0, 0])
        
        print("1. Joint1(Base) position analysis:")
        print("   - Position calculation: [0, 0, link_lengths[0]]")
        print("   - Regardless of Base angle θ1, Joint1 position is always [0, 0, 1.0]")
        print("   - This is because Base joint rotates around Z-axis, and Joint1 is on Z-axis!")
        print()
        
        for base_angle in [0, np.pi/4, np.pi/2, np.pi]:
            config = np.array([base_angle, 0, 0, 0])
            x1, y1, z1 = 0, 0, self.link_lengths[0]
            print(f"   Base angle={base_angle:.2f}rad, Joint1 position: [{x1:.3f}, {y1:.3f}, {z1:.3f}]")
        
        print()
        print("2. Why learning difficulties occur?")
        print("   - Joint1 position is independent of Base angle θ1, creating information bottleneck")
        print("   - Neural network cannot infer Base rotation angle from Joint1 position")
        print("   - Base angle can only be inferred from subsequent joint positions")
        print()
        
        print("3. Wrist joint (Joint4) redundancy:")
        print("   - When end-effector pose requirements are not strict, Wrist angle can have multiple solutions")
        print("   - In some configurations, different Wrist angles may produce similar keypoint positions")
        print()
        
        print("4. Base angle impact on subsequent joints:")
        shoulder_angle = np.pi/4
        
        for base_angle in [0, np.pi/2]:
            config = np.array([base_angle, shoulder_angle, 0, 0])
            keypoints = self.forward_kinematics_simple(config)
            
            print(f"   Base={base_angle:.2f}rad:")
            for i, kp in enumerate(keypoints):
                print(f"     Keypoint{i}: [{kp[0]:.3f}, {kp[1]:.3f}, {kp[2]:.3f}]")
            print()
    
    def forward_kinematics_simple(self, joint_angles):
        """Simplified forward kinematics calculation"""
        angles = joint_angles
        keypoints = np.zeros((5, 3))
        
        keypoints[0] = [0, 0, 0]
        
        keypoints[1] = [0, 0, self.link_lengths[0]]
        
        cos_th1, sin_th1 = np.cos(angles[0]), np.sin(angles[0])
        cos_th2, sin_th2 = np.cos(angles[1]), np.sin(angles[1])
        
        local_x2 = self.link_lengths[1] * sin_th2
        local_z2 = self.link_lengths[1] * cos_th2
        
        x2 = cos_th1 * local_x2
        y2 = sin_th1 * local_x2
        z2 = keypoints[1, 2] + local_z2
        keypoints[2] = [x2, y2, z2]
        
        keypoints[3] = keypoints[2]
        keypoints[4] = keypoints[2]
        
        return keypoints
    
    def visualize_problem(self):
        """Visualize the problem"""
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(131, projection='3d')
        
        base_angles = np.linspace(0, 2*np.pi, 8)
        shoulder_angle = np.pi/4
        
        for i, base_angle in enumerate(base_angles):
            config = np.array([base_angle, shoulder_angle, 0, 0])
            keypoints = self.forward_kinematics_simple(config)
            
            ax1.plot([keypoints[0,0], keypoints[1,0]], 
                    [keypoints[0,1], keypoints[1,1]], 
                    [keypoints[0,2], keypoints[1,2]], 'b-', alpha=0.7)
            ax1.plot([keypoints[1,0], keypoints[2,0]], 
                    [keypoints[1,1], keypoints[2,1]], 
                    [keypoints[1,2], keypoints[2,2]], 'r-', alpha=0.7)
        
        ax1.scatter([0], [0], [1.0], color='red', s=50, alpha=0.8)
        
        ax1.set_title('Base Rotation Effect\n(Joint1 positions overlap!)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
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
        
        ax3 = fig.add_subplot(133)
        joints = ['Base', 'Shoulder', 'Elbow', 'Wrist']
        difficulties = [0.9, 0.3, 0.4, 0.8]
        colors = ['red', 'green', 'orange', 'red']
        
        bars = ax3.bar(joints, difficulties, color=colors, alpha=0.7)
        ax3.set_ylabel('Learning Difficulty')
        ax3.set_title('Joint Learning Difficulty')
        ax3.set_ylim(0, 1)
        
        for i, (bar, diff) in enumerate(zip(bars, difficulties)):
            height = bar.get_height()
            if diff > 0.7:
                reason = 'Position\nInvariant' if i == 0 else 'Redundant\nDOF'
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        reason, ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('/home/zhangrong/worker_lingyu/zhangrong/Else/arcsinx/design_analysis.png', 
                   dpi=150, bbox_inches='tight')
        print("Analysis chart saved as: design_analysis.png")
        plt.close()

if __name__ == "__main__":
    analyzer = DesignAnalysis()
    analyzer.analyze_joint_constraints()
    analyzer.visualize_problem()
