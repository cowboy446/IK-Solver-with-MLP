Would it be equivalent for a neural network to generate a set of **spatial coordinates of keypoints**, and to generate corresponding joint-level **axis-angles** for its kinematic chain?

Since an axis-angle representation can be derived from vectors of kinematic keypoints with Rodrigues' rotation formula(consisting of merely trigonometric operations and vector normalizations), it seems that a neural network should be able to learn this mapping, so the above two representations should be equivalent in terms of learnability, with a reflection added to original MLP: $$mlp(\text{keypoints}) \rightarrow \text{axis-angles}$$


Here's a demo aiming to test the fitting capability of MLP for:

1. `mlp_arcsin_torch.py`: Fit arcsin function with MLP
2. `vector_to_axis_angle.py`: Single DOF **Robot Arm** Inverse Kinematics with MLP (xyz --> axis-angle)
3. `robot_ik_complex.py`: 3-DOF **Robot Arm** Inverse Kinematics with MLP (3*(xyz) --> 3*(axis-angle)), note that each arm has only one rotational DOF
4. `robot_ik_5dof.py`: 5-DOF **Robot Arm** Inverse Kinematics with MLP (5*(xyz) --> 5*(axis-angle)), note that each arm has only one rotational DOF
5. `robot_ik_spherical_joints.py`: Spherical Joint **Robot Arm** Inverse Kinematics with MLP (xyz --> axis-angle), each arm has three rotational DOFs

The results show that MLP can [fit 1 to 4 well](https://github.com/cowboy446/IK-Solver-with-MLP/blob/main/results/robot_5dof_results.png), for they are all single-DOF joints and each joint provides one sin or cos function to fit. But it [fails](https://github.com/cowboy446/IK-Solver-with-MLP/blob/main/results/spherical_robot_results.png) in case 5 for 3-DOF spherical joints.