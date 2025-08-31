# Learning_to_Coordinate_1
Learning to Coordinate (L2C) is a general framework that meta-learns key hyperparameters for ADMM-DDP-based multiagent distributed trajectory optimization and adapts to diverse tasks and team sizes. These hyperparameters are modeled using lightweight, agent-wise neural networks to achieve the adaptations, trained efficiently through analytically differentiating the ADMM-DDP pipeline end-to-end. We validate the effectiveness of L2C on a multilift system, a challenging multiagent system consisting of multiple quadrotors cooperatively transporting a cable-suspended load. Our method achieves faster gradient computation than state-of-the-art methods such as PDP [[1]](#1)., and exhibits strong generalizability to diverse system dynamics, tasks, and team sizes without extra tuning.

<img width="3599" height="1552" alt="diagram_github" src="https://github.com/user-attachments/assets/94cf529a-96d2-4e87-b3c5-f9783a90621c" />

## Table of contents

1. [Dependency Packages](#Dependency-Packages)
2. [How to Use](#How-to-Use)
3. [Contact Us](#Contact-Us)

## 1. Dependency Packages
Please make sure that the following packages have already been installed before running the source code.
* CasADi: version 3.5.5 Info: https://web.casadi.org/
* Numpy: version 1.23.0 Info: https://numpy.org/
* Pytorch: version 1.12.0+cu116 Info: https://pytorch.org/
* Matplotlib: version 3.3.0 Info: https://matplotlib.org/
* Python: version 3.9.12 Info: https://www.python.org/
* Scipy: version 1.8.1 Info: https://scipy.org/
* Pandas: version 1.4.2 Info: https://pandas.pydata.org/
* scikit-learn: version 1.0.2 Info: https://scikit-learn.org/stable/whats_new/v1.0.html

## 2. How to Use
The implementation of L2C for multilift systems is straightforward to setup.  Simply follow the steps outlined below, sequentially, after downloading all the necessary files and folders.
1. Run the Python file '**Meta_learning_cable_references.py**' to meta-learn collision-free cable references. Before running, you can specify the system's parameters such as the quadrotor count, the load mass, and the cable length in Line 34 '*sysm_para*' where these parameters from left to right denote the load mass, the load radius, the load's rotational inertia, the load's CoM offset vector, the quadrotor count, the cable length, the quadrotor radius, and the obstacle radius. When running the code, you will be asked to choose the training mode: 't' for training and 'e' for evaluate; 'n' for neural adaptive hyperparameters and 'f' for fixed hyperparameters.
2. Run the same Python file $M$ times, where $M$ denotes the task count, by selecting 'e' for evaluation.
3. Run the Python file '**Meta_learning_cable_trajectories.py**' to meta-learn dynamically feasible cable trajectories. Set '*sysm_para*' in Line 38 to be the same as that in the file '**Meta_learning_cable_references.py**'， except for the last parameter denoting the quadrotor mass.

|       Training loss       |      Episode 0     | Episode 33|
|-----------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|
<img width="870" height="835" alt="meanloss_comparison2" src="https://github.com/user-attachments/assets/6f7b6053-11b9-4b21-b5b6-4d4c4fec7719" /> | <img width="817" height="853" alt="multiagent007" src="https://github.com/user-attachments/assets/0c4824b2-0b60-4bff-ad23-63e68db2e75b" />  | <img width="817" height="853" alt="multiagent3307" src="https://github.com/user-attachments/assets/3677702e-e075-4165-942e-2f86cf962dc9" />

|     Three-quadrotor team | Six-quadrotor team | Seven-quadrotor team |
|-----------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|
![3_lift_-3_2_DDP](https://github.com/user-attachments/assets/540b1b1c-a35c-4dc6-bba5-f5c307c66c15) | ![6_lift_-3_2_DDP](https://github.com/user-attachments/assets/49a68a58-6f62-48fd-836a-3035bea2d15f) | ![7_lift_-3_2_DDP](https://github.com/user-attachments/assets/df62b0c8-aa98-4217-8cd2-4551f09a1973)


## 3. Contact Us
If you encounter a bug in your implementation of the code, please do not hesitate to inform me.
* Name: Dr. Bingheng Wang
* Email: wangbingheng@u.nus.edu



## References
<a id="1">[1]</a> 
Jin, Wanxin and Mou, Shaoshuai and Pappas, George J, "Safe pontryagin differentiable programming", Advances in Neural Information Processing Systems, 34, 16034--16050, 2021
