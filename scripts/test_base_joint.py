import numpy as np

class TestRobotArm:
    def __init__(self):
        self.link_lengths = np.array([1.2, 1.0, 0.8, 0.5])
        
    def forward_kinematics(self, joint_angles):
        angles = joint_angles
        keypoints = np.zeros((5, 3))
        
        keypoints[0] = [0, 0, 0]
        
        x1 = 0
        y1 = 0
        z1 = self.link_lengths[0]
        keypoints[1] = [x1, y1, z1]
        
        cos_th1, sin_th1 = np.cos(angles[0]), np.sin(angles[0])
        cos_th2, sin_th2 = np.cos(angles[1]), np.sin(angles[1])
        
        local_x2 = self.link_lengths[1] * sin_th2
        local_y2 = 0
        local_z2 = self.link_lengths[1] * cos_th2
        x2 = x1 + cos_th1 * local_x2 - sin_th1 * local_y2
        y2 = y1 + sin_th1 * local_x2 + cos_th1 * local_y2
        z2 = z1 + local_z2
        keypoints[2] = [x2, y2, z2]
        
        return keypoints

robot = TestRobotArm()

# 测试Base关节影响 - 使用非零的Shoulder角度
print("=== Base关节影响测试（Shoulder=45°）===")
base_config = np.array([0, np.pi/4, 0, 0])  # Shoulder角度45度
base_keypoints = robot.forward_kinematics(base_config)

print("Base角度=0时的关键点:")
for i, kp in enumerate(base_keypoints[:3]):  # 只看前3个
    print(f"  关键点{i}: [{kp[0]:.3f}, {kp[1]:.3f}, {kp[2]:.3f}]")

# Base旋转90度，保持Shoulder=45°
rotated_config = np.array([np.pi/2, np.pi/4, 0, 0])
rotated_keypoints = robot.forward_kinematics(rotated_config)

print(f"\nBase角度=90°时的关键点:")
for i, kp in enumerate(rotated_keypoints[:3]):
    print(f"  关键点{i}: [{kp[0]:.3f}, {kp[1]:.3f}, {kp[2]:.3f}]")

print(f"\n变化量:")
for i, (orig, new) in enumerate(zip(base_keypoints[:3], rotated_keypoints[:3])):
    diff = np.linalg.norm(new - orig)
    print(f"  关键点{i}: {diff:.3f} units")
