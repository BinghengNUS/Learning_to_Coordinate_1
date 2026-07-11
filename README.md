# DiffCoord
Differentiable Coordination (DiffCoord) is a unified meta-learning framework that differentiates the truncated ADMM-DDP pipeline end-to-end to jointly learn task-adaptive problem-level and solver-level parameters for efficient distributed multi-agent trajectory optimization. Its main feature is a structure-exploiting ADMM-LQR distributed gradient solver that mirrors the forward ADMM-DDP pipeline and reuses key DDP and ADMM computation results. Applied to multilift systems, DiffCoord enables task-adaptive formation reconfiguration, scalable deployment across different team sizes, and robust real-flight payload transport through constrained spaces. 

<img width="4490" height="1930" alt="diagram_github" src="https://github.com/user-attachments/assets/5fa35ed5-d6ff-46e9-a42b-12c977e1ee2d" />



## Citation
If you find this work helpful in your publications, we would appreciate citing our paper。

```
@misc{wang2026diffcoorddifferentiablecoordinationdistributed,
      title={DiffCoord: Differentiable Coordination for Distributed Multi-Agent Trajectory Optimization}, 
      author={Bingheng Wang and Yichao Gao and Tianchen Sun and Shanker Ajay and Lin Zhao},
      year={2026},
      eprint={2509.01630},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.01630}, 
}
```

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
* jax: version 0.9.0.1 Info: https://docs.jax.dev/en/latest/installation.html
* jaxlib: version 0.9.0.1 Info: https://docs.jax.dev/en/latest/jep/9419-jax-versioning.html
* JAXopt: version 0.8.5 Info: https://jaxopt.github.io/stable/ 

## 2. How to Use
The implementation of Diffoord for multilift systems is straightforward to setup.  Simply follow the steps outlined below, sequentially, after downloading all the necessary files and folders.
1. Run the Python file '**main_LoadPlanner_DDP_ADMM_quaternion_Meta_Learning_COM_Dyn.py**' to meta-learn collision-free cable references. When running the code, you will be asked to choose the training mode: 't' for training and 'e' for evaluate; 'n' for neural adaptive hyperparameters and 'f' for fixed hyperparameters.
2. Run the Python file '**main_load_kinodynamic_planner_ADMM_DDP_Meta_learning_2nd_COM_Dyn_true_parallel_best_backup.py**' to meta-learn dynamically feasible cable trajectories.

Stage |       Training loss       |      Untrained     | Trained |
------|-----------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|
Stage 1 |<img width="944" height="472" alt="stage1_training_meta_loss" src="https://github.com/user-attachments/assets/95106504-c36b-4211-96c0-00504053650d" /> | <img width="944" height="944" alt="four_quadrotor_stage1_episode0" src="https://github.com/user-attachments/assets/8b5c9994-e04d-4c28-85ec-0e331605f55d" /> |   <img width="944" height="944" alt="four_quadrotor_stage1_episode13" src="https://github.com/user-attachments/assets/51cebb15-1ac1-45aa-94cf-14bc97b29ea9" />
Stage 2 |<img width="944" height="472" alt="stage2_training_meta_loss" src="https://github.com/user-attachments/assets/e87b44ac-68c3-4c4c-8ae0-4db72e1d3640" /> | <img width="944" height="944" alt="four_quadrotor_stage2_episode0" src="https://github.com/user-attachments/assets/c8467ffc-b435-4fd5-a964-4e91d2fc11e7" /> |<img width="944" height="944" alt="four_quadrotor_stage2_episode52" src="https://github.com/user-attachments/assets/c0fc321b-81fe-4248-8f45-9c169ccd0889" />

We conduct real-flight experiments using two multilift systems with three and six quadrotors, respectively. The meta-learned networks, trained on the 4-quadrotor multilift system, are directly deployed to the two-stage ADMM-DDP pipelines for both systems without extra tuning.

https://github.com/user-attachments/assets/bf191b6d-0a06-452b-938f-161fa88ae77b

To reproduce the 6s (3-drone) and 5s (6-drone) flight trajectories used in the experiments, run the Python file '**main_load_kinodynamic_planner_ADMM_DDP_2nd_evaluation_COM_Dyn_HV**' and '**main_load_kinodynamic_planner_ADMM_DDP_2nd_evaluation_COM_Dyn_V_6**', respectively.


## 3. Contact Us
If you encounter a bug in your implementation of the code, please do not hesitate to inform me.
* Name: Dr. Bingheng Wang
* Email: wangbingheng@u.nus.edu


