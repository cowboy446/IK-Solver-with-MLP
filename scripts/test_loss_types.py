#!/usr/bin/env python3
"""
测试不同损失类型的脚本

演示如何使用两种不同的损失类型：
1. keypoints_only: 只有关键点损失 (forward kinematics loss)
2. combined: 轴损失 + 角度损失 + 0.1*关键点损失
"""

from hand_ik_clean import main

def test_keypoints_only():
    """测试只使用关键点损失"""
    print("=" * 60)
    print("Testing KEYPOINTS ONLY loss")
    print("This will only use forward kinematics loss")
    print("=" * 60)
    
    main(loss_type='keypoints_only')

def test_combined_loss():
    """测试组合损失"""
    print("=" * 60)
    print("Testing COMBINED loss")
    print("This will use: axis loss + angle loss + 0.1 * keypoint loss")
    print("=" * 60)
    
    main(loss_type='combined')

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        loss_type = sys.argv[1]
        if loss_type == 'keypoints_only':
            test_keypoints_only()
        elif loss_type == 'combined':
            test_combined_loss()
        else:
            print("Usage: python test_loss_types.py [keypoints_only|combined]")
            print("Available loss types:")
            print("  keypoints_only - Only forward kinematics loss")
            print("  combined       - Axis + angle + 0.1*keypoint loss")
    else:
        print("Usage: python test_loss_types.py [keypoints_only|combined]")
        print("\nAvailable loss types:")
        print("  keypoints_only - Only forward kinematics loss")
        print("  combined       - Axis + angle + 0.1*keypoint loss")
        print("\nExample:")
        print("  python test_loss_types.py keypoints_only")
        print("  python test_loss_types.py combined")
