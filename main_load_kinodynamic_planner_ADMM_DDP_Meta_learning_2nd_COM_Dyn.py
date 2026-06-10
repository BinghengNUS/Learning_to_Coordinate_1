"""
Main function of the load planner (Tension Dynamic Allocation)
------------------------------------------------------
1st version, Dr. Wang Bingheng, 07-Mar-2024
"""

from casadi import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import Dynamics_load_cable_autotuning_2nd_COM_Dyn
import Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn
import math
import time as TM
from scipy.spatial.transform import Rotation as Rot
import os
import Neural_network
import torch
import random

if not hasattr(torch.nn.modules.module, '_WrappedHook'):
    class _WrappedHook:
        def __init__(self, *args, **kwargs):
            self.hook = args[0] if args else None

        def __call__(self, *args, **kwargs):
            hook = getattr(self, 'hook', None)
            if hook is not None:
                return hook(*args, **kwargs)
            return None

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    torch.nn.modules.module._WrappedHook = _WrappedHook

def compatible_torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)

print("=============================================")
print("Main code for training or evaluating Automultilift")
print("Please choose mode")
mode = input("enter 't' or 'e' without the quotation mark, t: training; e: evaluation")
if mode == 't':
    print("Should we generate new initial models randomly?")
    initial_mode = input("enter 'y' or 'n' without the quotation mark, y: yes; n: no")
print("Please choose initial model for stage 1")
initial_model_1 = int(input("enter '0', '1', '2', '3' or '4' without the quotation mark"))
print("Please choose initial model for stage 2")
initial_model   = int(input("enter '0', '1', '2', '3' or '4' without the quotation mark"))
print("Please choose ADMM truncation number for stage 1")
max_iter_ADMM_1 = int(input("enter '2', '3', '4', or '5' without the quotation mark"))
print("Please choose ADMM truncation number for stage 2")
max_iter_ADMM   = int(input("enter '2', '3', '4', or '5' without the quotation mark"))
print("Please choose weight_mode")
weight_mode = input("enter 'n' or 'f' without the quotation mark, n: neural network; f: fixed")
# print("Please choose ADMM penalty mode")
# adaptiveADMM = input("enter 'a' or 'f' without the quotation mark, a: iteration-adaptive; f: iteration-fixed")
print("=============================================")

if not os.path.exists("trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)"):
    os.makedirs("trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)")

m1        = 0.45  # the load's net weight [kg], a circular basket with uniform mass distribution
m2        = 0.25  # the added mass [kg]
mtot      = m1+m2 # the total weight [kg]
nq        = 4     # the number of quadrotors
mq        = 0.25  # the quadrotor mass [kg]
fqmax     = 0.75*9.81 # the quadrotor maximum thrust [N]
cl0       = 1     # the cable length [m]
rq        = 0.15  # the radius of quadrotor [m]
rl        = 0.25  # the radius of the load [m]
ro        = 0.65  # the radius of obstacle [m]

"""--------------------------------------Load Environment---------------------------------------"""
sysm_para = np.array([m1, m2, 
                      1/4*m1*rl**2, 1/4*m1*rl**2, 1/2*m1*rl**2, 
                      rl, nq, rq, mq, fqmax,
                      cl0, ro])

dt        = 0.04
sysm      = Dynamics_load_cable_autotuning_2nd_COM_Dyn.multilift_model(sysm_para,dt)
rp0       = np.array([[0.05,0.05,0]]).T # the initial 
sysm.Rotational_Inertia(rp0)
sysm.model()
nxl       = sysm.nxl # dimension of the load's state
nul       = sysm.nul # dimension of the load's control
nxi       = sysm.nxi # dimension of the cable's state
nui       = sysm.nui # dimension of the cable's control

"""--------------------------------------Define Planner---------------------------------------"""
horizon   = 100
# pob1, pob2 = np.array([[1.7,1.3]]).T, np.array([[0.3,3.1]]).T # planar positions of the two obstacle in the world frame
pob1, pob2 = np.array([[1.7,1.15]]).T, np.array([[0.3,3.05]]).T
MPC_load  = Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn.MPC_Planner(sysm_para,dt,horizon)
MPC_load.Rotational_Inertia(rp0)
rg0       = m2/mtot*rp0
MPC_load.allocation_martrix(rg0)
MPC_load.SetStateVariables(sysm.xl,sysm.xi)
MPC_load.SetCtrlVariables(sysm.ul,sysm.ui)
MPC_load.SetDyns(sysm.model_l,sysm.model_i)
MPC_load.SetWeightPara()
MPC_load.SetPayloadCostDyn(max_iter_ADMM)
MPC_load.SetCableCostDyn(max_iter_ADMM)
MPC_load.SetConstriants(pob1,pob2)
MPC_load.SetADMMSubP2_SoftCost_k()
MPC_load.SetADMMSubP2_SoftCost_N()
MPC_load.ADMM_SubP2_Init()
MPC_load.ADMM_SubP2_N_Init()
MPC_load.Load_derivatives_DDP_ADMM()
MPC_load.Cable_derivatives_DDP_ADMM()
MPC_load.system_derivatives_SubP2_ADMM_k()
MPC_load.system_derivatives_SubP2_ADMM_N()
MPC_load.system_derivatives_SubP3_ADMM()

npl       = MPC_load.npl
npi       = MPC_load.npi
npauto    = MPC_load.n_Pauto

D_inl, D_h1l, D_h2l, D_outl = 2, 16, 32, MPC_load.npl # inputs: x, y, 13*2+6+4=36
def convert_nn_l(nn_l_outcolumn):
    # convert a column tensor to a row np.array
    nn_l_row = np.zeros((1,D_outl))
    for i in range(D_outl):
        nn_l_row[0,i] = nn_l_outcolumn[i,0]
    return nn_l_row

D_ini, D_h1i, D_h2i, D_outi = 2, 16, 32, MPC_load.npi # 14*2+4+4 = 36
def convert_nn_i(nn_i_outcolumn):
    # convert a column tensor to a row np.array
    nn_i_row = np.zeros((1,D_outi))
    for i in range(D_outi):
        nn_i_row[0,i] = nn_i_outcolumn[i,0]
    return nn_i_row

num_task   = 10
max_radius = 0.15  # reference length [m], same as in Stage1
# if mode == 't':
#     rp_task   = []
#     for _ in range(num_task):
#         rp        = np.random.uniform(0,max_radius) # in training, we do not make it very large for the concern of stable training
#         alpha     = np.random.uniform(0,2*np.pi)
#         random_rp = np.array([[rp*np.cos(alpha),rp*np.sin(alpha),0]]).T # unit: [m]
#         rp_task  += [random_rp] # in this stage, we CANNOT normalize it as it is needed in the load dynamics!
#     print('rp_task=',rp_task)
#     np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/rp_task',rp_task)
rp_task    = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/rp_task.npy') # unit: [m]
# parameters of ADAM
lr0       = 0.2 # 0.25 
lr_nn     = lr0 
epsilon   = 1e-8
m0        = np.zeros(MPC_load.n_Pauto)
v0        = np.zeros(MPC_load.n_Pauto)
mt0       = 0
vt0       = 0
mrp0      = 0
vrp0      = 0
beta1     = 0.94 # 0.95
beta2     = 0.999 
"""--------------------------------------Define Gradient Solver--------------------------------------"""
Grad_Solver   = Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn.Gradient_Solver(sysm_para, horizon, MPC_load.xl, MPC_load.ul, MPC_load.scxl, MPC_load.scul, MPC_load.xi, MPC_load.ui, MPC_load.scxi, MPC_load.scui, MPC_load.P_auto, MPC_load.para_l, MPC_load.para_i)


"""--------------------------------------Define Load Reference---------------------------------------"""
Coeffx        = np.zeros((2,8))
Coeffy        = np.zeros((2,8))
Coeffz        = np.zeros((2,8))
for k in range(2):
    Coeffx[k,:] = np.load('Reference_traj_4/coeffx'+str(k+1)+'.npy')
    Coeffy[k,:] = np.load('Reference_traj_4/coeffy'+str(k+1)+'.npy')
    Coeffz[k,:] = np.load('Reference_traj_4/coeffz'+str(k+1)+'.npy')

Time   = []
time   = 0
for k in range(horizon):
    Time  += [time]
    time += dt

# initial palyload's state (same as that used in the meta-learning of cable reference)
# x0         = np.random.normal(0,0.01)
# y0         = np.random.normal(0,0.01)
# z0         = np.random.normal(0.5,0.01)
# pl         = np.array([[x0,y0,z0]]).T
# vl         = np.reshape(np.random.normal(0,0.01,3),(3,1)) # initial velocity of CO in {I}
# Eulerl     = np.clip(np.reshape(np.random.normal(0,0.01,3),(3,1)),-5/57.3,5/57.3) # should be small
# Rl0        = sysm.dir_cosine(Eulerl)
# r          = Rot.from_matrix(Rl0)  
# # quaternion in the format of x, y, z, w 
# # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html)
# ql0        = r.as_quat() 
# ql         = np.array([[ql0[3], ql0[0], ql0[1], ql0[2]]]).T
# wl         = np.reshape(np.random.normal(0,0.01,3),(3,1))
# xl_init    = np.reshape(np.vstack((pl,vl,ql,wl)),nxl)
# np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/xl_init_'+str(max_iter_ADMM),xl_init)
xl_init    = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/xl_init_'+str(max_iter_ADMM)+'.npy') # 1-dim array
k_const    = 10 # dimensionless
# it has no physical meaning and only scales the numerical representation of the input data 
# it helps the neural network to learn the mapping between the input data and the ADMM-DDP hyperparameters.
# it can be tuned to be different from that in Stage 1. In evaluation, we can use the same value of k.
wt0, wrp0  = 1, 1 

# if mode == 't':
#     if initial_mode == 'y':
#         # Tunable_para0 = []
#         NN_l          = []
#         NN_i          = []
#         for _ in range(5): # generate 5 candidates
#             # Tunable_para0 += [np.random.normal(0,0.1,npauto)] # initialization
#             NN_l          += [Neural_network.Net(D_inl,D_h1l,D_h2l,D_outl)]
#             NN_i          += [Neural_network.Net(D_ini,D_h1i,D_h2i,D_outi)]
#         # np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Tunable_para0',Tunable_para0)
#         PATHl_init   = "trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/initial_NN_l.pt"
#         PATHi_init   = "trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/initial_NN_i.pt"
#         torch.save(NN_l,PATHl_init)
#         torch.save(NN_i,PATHi_init)
PATHl_init    = "trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/initial_NN_l.pt"
PATHi_init    = "trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/initial_NN_i.pt"
Tunable_para0 = np.load('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Tunable_para0.npy')
NN_l          = compatible_torch_load(PATHl_init)
NN_i          = compatible_torch_load(PATHi_init)

# cable references (waypoints) that need not achieve obstacle-avoidance, just for stabilizing the cable trajectories
# alpha  = 2*np.pi/nq
# beta   = np.random.normal(4/9*np.pi,0.01) # inter-robot collision-avoidance
# d_quad = (rl+cl0*math.cos(beta))*alpha
# while d_quad<(4*rq+0.05):
#     beta  -= 0.05
#     d_quad = (rl+cl0*math.cos(beta))*alpha
# DI         = []
# TI         = np.zeros((nq,horizon+1))
# for i in range(nq):
#     di     = np.zeros((3,horizon+1))
#     for k in range(horizon+1):
#         di_k        = np.array([[math.cos(beta)*math.cos(i*alpha),math.cos(beta)*math.sin(i*alpha),math.sin(beta)]]).T # world frame
#         TI[i,k]     = mtot*9.81/nq
#         di[:,k:k+1] = di_k # inertial frame
#     DI += [di]


def train(m0,v0,lr0,lr_nn,Tunable_para0,NN_l,NN_i,wt0,wrp0,max_iter_ADMM,initial_model):
    tunable_para   = Tunable_para0[initial_model]
    nn_l           = NN_l[initial_model]
    nn_i           = NN_i[initial_model]
    wt             = wt0
    wrp            = wrp0
    i_iter         = 1
    i_max          = 1e2
    delta_loss     = 1e2
    loss0          = 1e2
    epi            = 1e-1
    xl_train       = []
    ul_train       = []
    Kfbl_train     = []
    scxl_train     = []
    scul_train     = []
    xc_train       = []
    uc_train       = []
    scxc_train     = []
    scuc_train     = []
    loss_train     = []
    wloss_train    = []
    losst_train    = []
    lossrp_train   = []
    iter_train     = []
    Wt             = []
    Wrp            = []
    Kauto          = []
    gradtimeOur    = []
    gradtimeCaos   = []
    gradtimeCao    = []
    gradtimePDP    = []
    gradtimeOur_c  = []
    gradtimeCaos_c = []
    gradtimeCao_c  = []
    gradtimePDP_c  = []
    meanerrorCao   = []
    meanerrorPDP   = []
    meanerrorCao_c = []
    meanerrorPDP_c = []
    gMeanerror_load = []
    gMeanerror_cable = []
    start_time1  = TM.time()
    m            = m0
    v            = v0
    avg_loss     = 500
    optimizer_l  = torch.optim.Adam(nn_l.parameters(),lr=lr_nn,betas=(beta1, beta2),eps=1e-08,weight_decay=0) 
    optimizer_i  = torch.optim.Adam(nn_i.parameters(),lr=lr_nn,betas=(beta1, beta2),eps=1e-08,weight_decay=0)
    # optimizer_l= torch.optim.AdamW(nn_l.parameters(),lr=lr_nn,weight_decay=1e-4,betas=(beta1, beta2)) # can also lead to an increase of the loss
    # optimizer_i= torch.optim.AdamW(nn_i.parameters(),lr=lr_nn,weight_decay=1e-4,betas=(beta1, beta2))
    weight_mode1 = 'n'
    loss_train1  = np.load('trained_data_meta_COM_Dyn/loss_train_'+str(initial_model_1)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'.npy')
    i_train_1    = len(loss_train1)-1
    
    while delta_loss>epi and i_iter<=i_max:
        Wt           += [wt]
        Wrp          += [wrp]
        task_loss     = 0 # meta-loss
        task_losst    = 0 # tracking error loss
        task_lossrp   = 0 # ADMM residual loss
        task_grad     = 0 # grad w.r.t. ADMM-DDP hyperparameters
        task_loss_nnl = 0 # meta-loss for the load
        task_loss_nni = 0 # meta-loss for the cables
        task_glosst   = 0
        task_glossrp  = 0
        xl_task    = []
        ul_task    = []
        Kfbl_task  = []
        scxl_task  = []
        scul_task  = []
        xc_task    = []
        uc_task    = []
        scxc_task  = []
        scuc_task  = []
        nloss      = 0
        Loss_task  = []
        wLoss_task = []
        
        for task_idx in range(num_task):
            sysm.Rotational_Inertia(rp_task[task_idx]) # [m]
            sysm.model()
            MPC_load.Rotational_Inertia(rp_task[task_idx]) # [m]
            rg_task = m2/mtot*rp_task[task_idx]
            MPC_load.allocation_martrix(rg_task) # [m]
            start_time = TM.time()
            MPC_load.SetDyns(sysm.model_l,sysm.model_i)
            MPC_load.SetConstriants(pob1,pob2)
            MPC_load.SetADMMSubP2_SoftCost_k()
            MPC_load.SetADMMSubP2_SoftCost_N()
            MPC_load.ADMM_SubP2_Init()
            MPC_load.ADMM_SubP2_N_Init()
            MPC_load.Load_derivatives_DDP_ADMM()
            MPC_load.system_derivatives_SubP2_ADMM_k()
            MPC_load.system_derivatives_SubP2_ADMM_N()
            buildingtime    = (TM.time() - start_time)*1000
            print("building time:--- %s ms ---" % format(buildingtime,'.2f'))
            Ref_xl = np.zeros(nxl*(horizon+1))
            Ref_ul = np.zeros(nul*horizon)
            Time   = []
            time   = 0
            DI     = np.load('Planning_plots_meta_COM_Dyn/cable_direction_'+str(i_train_1)+'_'+str(task_idx)+'_'+str(initial_model_1)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'.npy')
            TI     = np.load('Planning_plots_meta_COM_Dyn/tension_magnitude_'+str(i_train_1)+'_'+str(task_idx)+'_'+str(initial_model_1)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'.npy')
            ref_xc = [np.zeros((horizon+1)*nxi) for _ in range(nq)]
            # the reference in the running stage
            for k in range(horizon):
                Time  += [time]
                ref_xl, ref_ul = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
                Ref_xl[k*nxl:(k+1)*nxl] = ref_xl
                Ref_ul[k*nul:(k+1)*nul] = ref_ul
                time += dt
                for i in range(nq):
                    ref_di_k = np.reshape(DI[i][:,k],(3,1))
                    ref_wi_k = np.zeros((3,1))
                    ref_ai_k = np.zeros((3,1))
                    ref_ji_k = np.zeros((3,1))
                    ref_ti_k = np.reshape(TI[i,k],(1,1))
                    ref_dti_k= np.zeros((1,1))
                    ref_xi_k = np.reshape(np.vstack((ref_di_k,ref_wi_k,ref_ai_k,ref_ji_k,ref_ti_k,ref_dti_k)),nxi)
                    ref_xc[i][k*nxi:(k+1)*nxi] = ref_xi_k
            ref_uq = np.zeros(int(nq)*nui)
            # the reference in the terminal stage
            ref_xl, ref_ul = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
            for i in range(nq):
                ref_di_N = np.reshape(DI[i][:,-1],(3,1))
                ref_wi_N = np.zeros((3,1))
                ref_ai_N = np.zeros((3,1))
                ref_ji_N = np.zeros((3,1))
                ref_ti_N = np.reshape(TI[i,-1],(1,1))
                ref_dti_N= np.zeros((1,1))
                ref_xi_N = np.reshape(np.vstack((ref_di_N,ref_wi_N,ref_ai_N,ref_ji_N,ref_ti_N,ref_dti_N)),nxi)
                ref_xc[i][horizon*nxi:(horizon+1)*nxi]=ref_xi_N
            Ref_xl[horizon*nxl:(horizon+1)*nxl] = ref_xl
            # the initial cable state (world frame)
            xq_init = np.zeros(nq*nxi) 
            for i in range(nq):
                di_init   = np.reshape(DI[i][:,0],(3,1)) 
                wi_init   = np.zeros((3,1))
                ai_init   = np.zeros((3,1))
                ji_init   = np.zeros((3,1))
                ti_init   = np.array([[TI[i,0]]])
                dti_init  = np.zeros((1,1))
                xi_init   = np.reshape(np.vstack((di_init,wi_init,ai_init,ji_init,ti_init,dti_init)),nxi)
                xq_init[i*nxi:(i+1)*nxi] = xi_init
            # generate the corresponding hyperparameters, given the task rg
            if weight_mode == 'n':
                # radius           = np.sqrt(rg_task[0]**2+rg_task[1]**2)# m
                nn_input         = np.reshape(rg_task[0:2]/max_radius*k_const,(2,1)) #dimensionless
                nn_l_output_task = convert_nn_l(nn_l(nn_input))
                P_weight1        = Grad_Solver.Set_Parameters_nn_l(nn_l_output_task)
                nn_i_output_task = convert_nn_i(nn_i(nn_input))
                P_weight2        = Grad_Solver.Set_Parameters_nn_i(nn_i_output_task)
                print('iter_train=',i_iter,'task_idx=',task_idx,'rg_task [m]:',rg_task,'weightmode:',weight_mode,'initial_model_1=',initial_model_1,'initial_model=',initial_model)
            else:
                # if norm_dldw>=loss_max:
                #     p_min  = np.clip(p_min+1e-3,Grad_Solver.p_min,1e-1)
                weight     = Grad_Solver.Set_Parameters(tunable_para)
                P_weight1  = weight[0:npl]
                P_weight2  = weight[npl:npauto]
                print('iter_train=',i_iter,'task_idx=',task_idx,'rg_task [m]:',rg_task,'weightmode:',weight_mode,'initial_model_1=',initial_model_1,'initial_model=',initial_model)
           

            print('iter_train=',i_iter,'task_idx=',task_idx,'Ql=',P_weight1[0:nxl],'QlN=',P_weight1[nxl:2*nxl],'Rl=',P_weight1[2*nxl:2*nxl+nul],'px=',P_weight1[-4],'gammax=',P_weight1[-2],'pu=',P_weight1[-3],'gammau=',P_weight1[-1])
            print('iter_train=',i_iter,'task_idx=',task_idx,'Qi=',P_weight2[0:nxi],'QiN=',P_weight2[nxi:2*nxi],'Ri=',P_weight2[2*nxi:2*nxi+nui],'pix=',P_weight2[-4],'gammaix=',P_weight2[-2],'piu=',P_weight2[-3],'gammaiu=',P_weight2[-1])
            xl_init_task    = np.zeros(nxl)
            xl_init_task[0] = xl_init[0] + rg_task[0]
            xl_init_task[1] = xl_init[1] + rg_task[1]
            xl_init_task[2:nxl] = xl_init[2:nxl]
            start_time = TM.time()
            opt_sol, Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3 = MPC_load.ADMM_forward_MPC(Ref_xl,Ref_ul,ref_xc,ref_uq,xl_init_task,xq_init,P_weight1,P_weight2,max_iter_ADMM)
            mpctime    = (TM.time() - start_time)*1000
            print("forward mpc:--- %s ms ---" % format(mpctime,'.2f'))
            start_time = TM.time()
            Grad_Out1l, Grad_Out1c, Grad_Out2, Grad_Out3, GradTime, GradTimeCaos, GradTimeCao,  GradTimePDP,  GradTime_c, GradTimeCaos_c, GradTimeCao_c, GradTimePDP_c,  MeanerrorCao, MeanerrorPDP, MeanerrorCao_c, MeanerrorPDP_c, gMeanerror_l, gMeanerror_c   = MPC_load.ADMM_Gradient_Solver(Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3, Ref_xl, Ref_ul, ref_xc, ref_uq, P_weight1, P_weight2)
            gradtime    = (TM.time() - start_time)*1000
            print("backward:--- %s ms ---" % format(gradtime,'.2f'))
            gradtimeOur    += [GradTime[-1]]
            gradtimeCaos   += [GradTimeCaos[-1]]
            gradtimeCao    += [GradTimeCao[-1]]
            gradtimePDP    += [GradTimePDP[-1]]
            gradtimeOur_c  += [GradTime_c[-1]]
            gradtimeCaos_c += [GradTimeCaos_c[-1]]
            gradtimeCao_c  += [GradTimeCao_c[-1]]
            gradtimePDP_c  += [GradTimePDP_c[-1]]
            meanerrorCao   += [MeanerrorCao[-1]]
            meanerrorPDP   += [MeanerrorPDP[-1]]
            meanerrorCao_c += [MeanerrorCao_c[-1]]
            meanerrorPDP_c += [MeanerrorPDP_c[-1]]
            gMeanerror_load += [gMeanerror_l]
            gMeanerror_cable += [gMeanerror_c]
            dldw, loss, loss_track, loss_resid, gloss_t, gloss_rp  = Grad_Solver.ChainRule(Opt_Sol1_l,Opt_Sol1_c,Opt_Sol2,Ref_xl,ref_xc,Grad_Out1l,Grad_Out1c,Grad_Out2,wt,wrp)
            print('iter_train=',i_iter,'task_idx=',task_idx,'loss_tp=',loss_track,'loss_rp=',loss_resid, 'loss_tp+loss_rp=', loss_track+loss_resid,'wloss=',loss)
            Loss_task      += [loss_track[0]+loss_resid[0]]
            wLoss_task     += [loss[0]]
            
            if i_iter >1:
                if (loss_track[0]+loss_resid[0]) <2e3*epi: #only stable loss can be used
                    nloss          += 1
                    task_loss      += loss[0]
                    task_losst     += loss_track[0]
                    task_lossrp    += loss_resid[0]
                    task_glosst    += gloss_t
                    task_glossrp   += gloss_rp
                    if weight_mode == 'n':
                        dwdpl        = Grad_Solver.ChainRule_Gradient_nn_l(nn_l_output_task)
                        dldwl        = np.reshape(dldw[0,0:npl],(1,npl))
                        dldpl        = np.reshape(dldwl@dwdpl,(1,npl))
                        loss_nn_l    = nn_l.myloss(nn_l(nn_input),dldpl)
                        task_loss_nnl += loss_nn_l
                        dldpl        = np.reshape(dldpl,npl)
                        dwdpi        = Grad_Solver.ChainRule_Gradient_nn_i(nn_i_output_task)
                        dldwi        = np.reshape(dldw[0,npl:npauto],(1,npi))
                        dldpi        = np.reshape(dldwi@dwdpi,(1,npi))
                        loss_nn_i    = nn_i.myloss(nn_i(nn_input),dldpi)
                        task_loss_nni += loss_nn_i
                        dldpi        = np.reshape(dldpi,npi)
                        dldp         = np.append(dldpl,dldpi)
                        task_grad  += dldp
                    else:
                        dwdp        = Grad_Solver.ChainRule_Gradient(tunable_para)
                        dldp        = np.reshape(dldw@dwdp,npauto)
                        task_grad  += dldp
            else:
                nloss          += 1
                task_loss      += loss[0]
                task_losst     += loss_track[0]
                task_lossrp    += loss_resid[0]
                task_glosst    += gloss_t
                task_glossrp   += gloss_rp
                if weight_mode == 'n':
                    dwdpl        = Grad_Solver.ChainRule_Gradient_nn_l(nn_l_output_task)
                    dldwl        = np.reshape(dldw[0,0:npl],(1,npl))
                    dldpl        = np.reshape(dldwl@dwdpl,(1,npl))
                    loss_nn_l    = nn_l.myloss(nn_l(nn_input),dldpl)
                    task_loss_nnl += loss_nn_l
                    dldpl        = np.reshape(dldpl,npl)
                    dwdpi        = Grad_Solver.ChainRule_Gradient_nn_i(nn_i_output_task)
                    dldwi        = np.reshape(dldw[0,npl:npauto],(1,npi))
                    dldpi        = np.reshape(dldwi@dwdpi,(1,npi))
                    loss_nn_i    = nn_i.myloss(nn_i(nn_input),dldpi)
                    task_loss_nni += loss_nn_i
                    dldpi        = np.reshape(dldpi,npi)
                    dldp         = np.append(dldpl,dldpi)
                    task_grad  += dldp
                else:
                    dwdp        = Grad_Solver.ChainRule_Gradient(tunable_para)
                    dldp        = np.reshape(dldw@dwdp,npauto)
                    task_grad  += dldp
            xl_task    += [opt_sol['xl_traj']]
            ul_task    += [opt_sol['ul_traj']]
            Kfbl_task  += [opt_sol['Kfbl_traj']]
            scxl_task  += [opt_sol['scxl_traj']]
            scul_task  += [opt_sol['scul_traj']]
            xc_task    += [opt_sol['xc_traj']]
            uc_task    += [opt_sol['uc_traj']]
            scxc_task  += [opt_sol['scxc_traj']]
            scuc_task  += [opt_sol['scuc_traj']]

        if weight_mode == 'n':
            optimizer_l.zero_grad()
            avg_loss_nn_l = task_loss_nnl/(nloss+1e-8)
            avg_loss_nn_l.backward()
            optimizer_l.step()
            optimizer_i.zero_grad()
            avg_loss_nn_i = task_loss_nni/(nloss+1e-8)
            avg_loss_nn_i.backward()
            optimizer_i.step()
            avg_grad    = task_grad/(nloss+1e-8)
            
        else:
            avg_grad    = task_grad/(nloss+1e-8)
            # ADAM adaptive learning
            for k in range(int(npauto)):
                m[k]    = beta1*m[k] + (1-beta1)*avg_grad[k]
                m_hat   = m[k]/(1-beta1**i_iter)
                v[k]    = beta2*v[k] + (1-beta2)*avg_grad[k]**2
                v_hat   = v[k]/(1-beta2**i_iter)
                lr      = lr0/(np.sqrt(v_hat)+epsilon)
                tunable_para[k] = tunable_para[k] - lr*m_hat
        
        avg_losst   = task_losst/(nloss+1e-8)
        avg_lossrp  = task_lossrp/(nloss+1e-8)
        avg_loss    = task_loss/(nloss+1e-8)
        avg_glosst  = task_glosst/(nloss+1e-8)
        avg_glossrp = task_glossrp/(nloss+1e-8)
        # update the weights in the meta-loss
        wt, wrp, kauto = Grad_Solver.adaptive_meta_loss_weights(avg_losst,avg_lossrp,avg_glosst,avg_glossrp,wt)
        
        loss_train += [avg_losst+avg_lossrp]
        wloss_train += [avg_loss]
        losst_train += [avg_losst]
        lossrp_train += [avg_lossrp]
        xl_train   += [xl_task]
        ul_train   += [ul_task]
        Kfbl_train += [Kfbl_task]
        scxl_train += [scxl_task]
        scul_train += [scul_task]
        xc_train   += [xc_task]
        uc_train   += [uc_task]
        scxc_train += [scxc_task]
        scuc_train += [scuc_task]
        iter_train += [i_iter]
        Kauto      += [kauto]
        if i_iter==1:
            epi = 1e-3*(avg_losst+avg_lossrp)
        if i_iter>40: # enough training
            delta_loss = abs(avg_losst+avg_lossrp-loss0)
        loss0      = avg_losst+avg_lossrp
        dldp1      = avg_grad[0:npl]
        dldp2      = avg_grad[npl:npauto]
        print('iter_train=',i_iter,'loss=',avg_losst+avg_lossrp,'loss_t=',avg_losst,'loss_rp=',avg_lossrp,'loss_train=',loss_train,'wt=',wt,'wrp=',wrp,'kauto=',kauto,'medain of Loss_task=',np.median(Loss_task),'std of Loss_task=',np.std(Loss_task),'wloss_train=',wloss_train,'median of wloss_task=',np.median(wLoss_task),'std of wLoss_task=',np.std(wLoss_task))
        print('iter_train=',i_iter,'dldpQl=',dldp1[0:nxl],'dldpRl=',dldp1[2*nxl:2*nxl+nul],'dldppx=',dldp1[-4],'dldpgammax=',dldp1[-2],'dldppu=',dldp1[-3],'dldpgammau=',dldp1[-1])
        print('iter_train=',i_iter,'dldpQi=',dldp2[0:nxi],'dldpRi=',dldp2[2*nxi:2*nxi+nui],'dldppix=',dldp2[-4],'dldpgammaix=',dldp2[-2],'dldppiu=',dldp2[-3],'dldpgammaiu=',dldp2[-1],'weightmode:',weight_mode,'initial_model1=',initial_model_1,'initial_model=',initial_model)
        i_iter += 1 # comment this for comparing gradient computation time
        
        # below is the code for saving the trajectory optimization results using the last updated neural network (through lines 380 & 384)
        if delta_loss <=epi or i_iter>i_max:
            nloss         = 0
            task_loss     = 0
            task_losst    = 0 # tracking error loss
            task_lossrp   = 0 # ADMM residual loss
            task_grad     = 0
            task_loss_nnl = 0
            task_loss_nni = 0
            xl_task    = []
            ul_task    = []
            Kfbl_task  = []
            scxl_task  = []
            scul_task  = []
            xc_task    = []
            uc_task    = []
            scxc_task  = []
            scuc_task  = []
            Loss_task  = []
            wLoss_task = []
            Losst_task = []
            Lossrp_task = []
         
            for task_idx in range(num_task):
                sysm.Rotational_Inertia(rp_task[task_idx])
                sysm.model()
                MPC_load.Rotational_Inertia(rp_task[task_idx])
                rg_task = m2/mtot*rp_task[task_idx]
                MPC_load.allocation_martrix(rg_task)
                MPC_load.SetDyns(sysm.model_l,sysm.model_i)
                MPC_load.SetConstriants(pob1,pob2)
                MPC_load.SetADMMSubP2_SoftCost_k()
                MPC_load.SetADMMSubP2_SoftCost_N()
                MPC_load.ADMM_SubP2_Init()
                MPC_load.ADMM_SubP2_N_Init()
                MPC_load.Load_derivatives_DDP_ADMM()
                MPC_load.system_derivatives_SubP2_ADMM_k()
                MPC_load.system_derivatives_SubP2_ADMM_N()
                Ref_xl = np.zeros(nxl*(horizon+1))
                Ref_ul = np.zeros(nul*horizon)
                Time   = []
                time   = 0
                DI     = np.load('Planning_plots_meta_COM_Dyn/cable_direction_'+str(i_train_1)+'_'+str(task_idx)+'_'+str(initial_model_1)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'.npy')
                TI     = np.load('Planning_plots_meta_COM_Dyn/tension_magnitude_'+str(i_train_1)+'_'+str(task_idx)+'_'+str(initial_model_1)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'.npy')
                ref_xc = [np.zeros((horizon+1)*nxi) for _ in range(nq)]
                # the reference in the running stage
                for k in range(horizon):
                    Time  += [time]
                    ref_xl, ref_ul = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
                    Ref_xl[k*nxl:(k+1)*nxl] = ref_xl
                    Ref_ul[k*nul:(k+1)*nul] = ref_ul
                    time += dt
                    for i in range(nq):
                        ref_di_k = np.reshape(DI[i][:,k],(3,1))
                        ref_wi = np.zeros((3,1))
                        ref_ai = np.zeros((3,1))
                        ref_ji = np.zeros((3,1))
                        ref_ti_k = np.reshape(TI[i,k],(1,1))
                        ref_dti_k= np.zeros((1,1))
                        ref_xi_k = np.reshape(np.vstack((ref_di_k,ref_wi,ref_ai,ref_ji,ref_ti_k,ref_dti_k)),nxi)
                        ref_xc[i][k*nxi:(k+1)*nxi] = ref_xi_k
                ref_uq = np.zeros(int(nq)*nui)
                # the reference in the terminal stage
                ref_xl, ref_ul = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
                for i in range(nq):
                    ref_di_N = np.reshape(DI[i][:,-1],(3,1))
                    ref_wi_N = np.zeros((3,1))
                    ref_ai_N = np.zeros((3,1))
                    ref_ji_N = np.zeros((3,1))
                    ref_ti_N = np.reshape(TI[i,-1],(1,1))
                    ref_dti_N= np.zeros((1,1))
                    ref_xi_N = np.reshape(np.vstack((ref_di_N,ref_wi_N,ref_ai_N,ref_ji_N,ref_ti_N,ref_dti_N)),nxi)
                    ref_xc[i][horizon*nxi:(horizon+1)*nxi]=ref_xi_N
                Ref_xl[horizon*nxl:(horizon+1)*nxl] = ref_xl 
                # the initial cable state (world frame)
                xq_init = np.zeros(nq*nxi) 
                for i in range(nq):
                    di_init   = np.reshape(DI[i][:,0],(3,1)) 
                    wi_init   = np.zeros((3,1))
                    ai_init   = np.zeros((3,1))
                    ji_init   = np.zeros((3,1))
                    ti_init   = np.array([[TI[i,0]]])
                    dti_init  = np.zeros((1,1))
                    xi_init   = np.reshape(np.vstack((di_init,wi_init,ai_init,ji_init,ti_init,dti_init)),nxi)
                    xq_init[i*nxi:(i+1)*nxi] = xi_init
            
                if weight_mode == 'n':
                    # radius           = np.sqrt(rg_task[0]**2+rg_task[1]**2)# m
                    nn_input         = np.reshape(rg_task[0:2]/max_radius*k_const,(2,1)) #dimensionless
                    nn_l_output_task = convert_nn_l(nn_l(nn_input))
                    P_weight1  = Grad_Solver.Set_Parameters_nn_l(nn_l_output_task)
                    nn_i_output_task = convert_nn_i(nn_i(nn_input))
                    P_weight2  = Grad_Solver.Set_Parameters_nn_i(nn_i_output_task)
                else:
                    weight     = Grad_Solver.Set_Parameters(tunable_para)
                    P_weight1  = weight[0:npl]
                    P_weight2  = weight[npl:npauto]
                

                print('iter_train=',i_iter,'task_idx=',task_idx,'weightmode:',weight_mode,'initial_model_1=',initial_model_1,'initial_model=',initial_model)  
                print('iter_train=',i_iter,'task_idx=',task_idx,'Ql=',P_weight1[0:nxl],'QlN=',P_weight1[nxl:2*nxl],'Rl=',P_weight1[2*nxl:2*nxl+nul],'px=',P_weight1[-4],'gammax=',P_weight1[-2],'pu=',P_weight1[-3],'gammau=',P_weight1[-1])
                print('iter_train=',i_iter,'task_idx=',task_idx,'Qi=',P_weight2[0:nxi],'QiN=',P_weight2[nxi:2*nxi],'Ri=',P_weight2[2*nxi:2*nxi+nui],'pix=',P_weight2[-4],'gammaix=',P_weight2[-2],'piu=',P_weight2[-3],'gammaiu=',P_weight2[-1])
                xl_init_task    = np.zeros(nxl)
                xl_init_task[0] = xl_init[0] + rg_task[0]
                xl_init_task[1] = xl_init[1] + rg_task[1]
                xl_init_task[2:nxl] = xl_init[2:nxl]
                start_time = TM.time()
                opt_sol, Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3 = MPC_load.ADMM_forward_MPC(Ref_xl,Ref_ul,ref_xc,ref_uq,xl_init_task,xq_init,P_weight1,P_weight2,max_iter_ADMM)
                mpctime    = (TM.time() - start_time)*1000
                print("forward mpc:--- %s ms ---" % format(mpctime,'.2f'))
                start_time = TM.time()
                Grad_Out1l, Grad_Out1c, Grad_Out2, Grad_Out3, GradTime, GradTimeCaos, GradTimeCao,  GradTimePDP,  GradTime_c, GradTimeCaos_c, GradTimeCao_c, GradTimePDP_c,  MeanerrorCao, MeanerrorPDP, MeanerrorCao_c, MeanerrorPDP_c, gMeanerror_l, gMeanerror_c   = MPC_load.ADMM_Gradient_Solver(Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3, Ref_xl, Ref_ul, ref_xc, ref_uq, P_weight1, P_weight2)
                gradtime    = (TM.time() - start_time)*1000
                print("backward:--- %s ms ---" % format(gradtime,'.2f'))
                dldw, loss, loss_track, loss_resid, gloss_t, gloss_rp  = Grad_Solver.ChainRule(Opt_Sol1_l,Opt_Sol1_c,Opt_Sol2,Ref_xl,ref_xc,Grad_Out1l,Grad_Out1c,Grad_Out2,wt,wrp)
                print('iter_train=',i_iter,'task_idx=',task_idx,'loss_t=',loss_track,'loss_rp=',loss_resid,'loss_t+loss_rp=',loss_track+loss_resid)
                Loss_task      += [loss_track[0]+loss_resid[0]]
                wLoss_task     += [loss[0]]
                Losst_task     += [loss_track[0]]
                Lossrp_task    += [loss_resid[0]]
                if (loss_track[0]+loss_resid[0]) <2e3*epi: # only stable loss can be used
                    nloss          += 1
                    task_loss      += loss[0]
                    task_losst     += loss_track[0]
                    task_lossrp    += loss_resid[0]
                    if weight_mode == 'n':
                        dwdpl        = Grad_Solver.ChainRule_Gradient_nn_l(nn_l_output_task)
                        dldwl        = np.reshape(dldw[0,0:npl],(1,npl))
                        dldpl        = np.reshape(dldwl@dwdpl,(1,npl))
                        loss_nn_l    = nn_l.myloss(nn_l(nn_input),dldpl)
                        task_loss_nnl += loss_nn_l
                        dldpl        = np.reshape(dldpl,npl)
                        dwdpi        = Grad_Solver.ChainRule_Gradient_nn_i(nn_i_output_task)
                        dldwi        = np.reshape(dldw[0,npl:npauto],(1,npi))
                        dldpi        = np.reshape(dldwi@dwdpi,(1,npi))
                        loss_nn_i    = nn_i.myloss(nn_i(nn_input),dldpi)
                        task_loss_nni += loss_nn_i
                        dldpi        = np.reshape(dldpi,npi)
                        dldp         = np.append(dldpl,dldpi)
                        task_grad  += dldp
                    else:
                        dwdp        = Grad_Solver.ChainRule_Gradient(tunable_para)
                        dldp        = np.reshape(dldw@dwdp,npauto)
                        task_grad  += dldp
                
                xl_task    += [opt_sol['xl_traj']]
                ul_task    += [opt_sol['ul_traj']]
                Kfbl_task  += [opt_sol['Kfbl_traj']]
                scxl_task  += [opt_sol['scxl_traj']]
                scul_task  += [opt_sol['scul_traj']]
                xc_task    += [opt_sol['xc_traj']]
                uc_task    += [opt_sol['uc_traj']]
                scxc_task  += [opt_sol['scxc_traj']]
                scuc_task  += [opt_sol['scuc_traj']]
            
            avg_loss    = task_loss/(nloss+1e-8)
            avg_losst   = task_losst/(nloss+1e-8)
            avg_lossrp  = task_lossrp/(nloss+1e-8)
            loss_train += [avg_losst+avg_lossrp]
            wloss_train += [avg_loss]
            losst_train += [avg_losst]
            lossrp_train += [avg_lossrp]
            xl_train   += [xl_task]
            ul_train   += [ul_task]
            Kfbl_train += [Kfbl_task]
            scxl_train += [scxl_task]
            scul_train += [scul_task]
            xc_train   += [xc_task]
            uc_train   += [uc_task]
            scxc_train += [scxc_task]
            scuc_train += [scuc_task]
            iter_train += [i_iter]
            if nloss == num_task: # all tasks are stable
                delta_loss = abs(avg_losst+avg_lossrp - loss0)
            else:
                delta_loss = 1e2
            loss0      = avg_losst+avg_lossrp
            if delta_loss >epi:
                if weight_mode == 'n':
                    optimizer_l.zero_grad()
                    avg_loss_nn_l = task_loss_nnl/(nloss+1e-8)
                    avg_loss_nn_l.backward()
                    optimizer_l.step()
                    optimizer_i.zero_grad()
                    avg_loss_nn_i = task_loss_nni/(nloss+1e-8)
                    avg_loss_nn_i.backward()
                    optimizer_i.step()
                    avg_grad    = task_grad/(nloss+1e-8)
                else:
                    avg_grad    = task_grad/(nloss+1e-8)
                    # ADAM adaptive learning
                    for k in range(int(npauto)):
                        m[k]    = beta1*m[k] + (1-beta1)*avg_grad[k]
                        m_hat   = m[k]/(1-beta1**i_iter)
                        v[k]    = beta2*v[k] + (1-beta2)*avg_grad[k]**2
                        v_hat   = v[k]/(1-beta2**i_iter)
                        lr      = lr0/(np.sqrt(v_hat)+epsilon)
                        tunable_para[k] = tunable_para[k] - lr*m_hat
            print('iter_train=',i_iter,'loss=',avg_losst+avg_lossrp,'loss_t=',avg_losst,'loss_rp=',avg_lossrp,'loss_train=',loss_train,'wt=',wt,'wrp=',wrp,'weightmode:',weight_mode,'initial_model1=',initial_model_1,'initial_model=',initial_model,'medain of Loss_task=',np.median(Loss_task),'std of Loss_task=',np.std(Loss_task),'wloss_train=',wloss_train,'median of wloss_task=',np.median(wLoss_task),'std of wLoss_task=',np.std(wLoss_task))
            i_iter += 1 

    traintime    = (TM.time() - start_time1)
    print("train:--- %s s ---" % format(traintime,'.2f'))
    # save the trained network models
    if weight_mode == 'n':
        PATH2   = "trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/trained_nn_l_"+str(initial_model)+'_'+str(max_iter_ADMM)+"_"+str(weight_mode)+".pt"
        torch.save(nn_l,PATH2)
        PATH3   = "trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/trained_nn_i_"+str(initial_model)+'_'+str(max_iter_ADMM)+"_"+str(weight_mode)+".pt"
        torch.save(nn_i,PATH3)
    else:   
        np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/tunable_para_trained_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),tunable_para)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/loss_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),loss_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/std_wloss_task_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),np.std(wLoss_task))
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/wloss_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),wloss_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/losst_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),losst_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/lossrp_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),lossrp_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Losst_task_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Losst_task)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Lossrp_task_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Lossrp_task)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Wt_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Wt)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Wrp_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Wrp)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Kauto_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Kauto)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/xl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),xl_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/ul_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),ul_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Kfbl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Kfbl_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/scxl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),scxl_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/scul_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),scul_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/xc_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),xc_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/uc_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),uc_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/scxc_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),scxc_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/scuc_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),scuc_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/train_num_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),iter_train)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gradtimeOur_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimeOur)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gradtimeOur_c_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimeOur_c)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gradtimeCaos_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimeCaos)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gradtimeCaos_c_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimeCaos_c)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gradtimeCao_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimeCao)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gradtimeCao_c_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimeCao_c)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gradtimePDP_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimePDP)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gradtimePDP_c_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimePDP_c)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/meanerrorCao_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),meanerrorCao)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/meanerrorPDP_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),meanerrorPDP)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/meanerrorCao_c_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),meanerrorCao_c)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/meanerrorPDP_c_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),meanerrorPDP_c)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gMeanerror_l_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gMeanerror_load)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gMeanerror_c_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gMeanerror_cable)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Final_loss_task_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Loss_task)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Final_Gradout1l_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Grad_Out1l)
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Final_Gradout1c_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Grad_Out1c)
    
    plt.figure(1,figsize=(6,4),dpi=400)
    plt.plot(wloss_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('wLoss')
    plt.grid()
    plt.savefig('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/wloss_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(2,figsize=(6,4),dpi=400)
    plt.plot(loss_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/loss_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(3,figsize=(6,4),dpi=400)
    plt.plot(losst_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Loss_track')
    plt.grid()
    plt.savefig('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/losst_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(4,figsize=(6,4),dpi=400)
    plt.plot(lossrp_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Loss_residuals')
    plt.grid()
    plt.savefig('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/lossrp_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(5,figsize=(6,4),dpi=400)
    plt.plot(Wt, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Wt')
    plt.grid()
    plt.savefig('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Wt_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(6,figsize=(6,4),dpi=400)
    plt.plot(Wrp, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Wrp')
    plt.grid()
    plt.savefig('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Wrp_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(7,figsize=(6,4),dpi=400)
    plt.plot(Kauto, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Kauto')
    plt.grid()
    plt.savefig('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Kauto_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()


    


def evaluate(i_train,task_idx,max_iter_ADMM,initial_model):
    if not os.path.exists("Planning_plots_multiagent_meta_COM_Dyn"):
        os.makedirs("Planning_plots_multiagent_meta_COM_Dyn")
    rp_task = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/rp_task.npy')
    rg_task = m2/mtot*rp_task[task_idx] # [m]
    MPC_load.allocation_martrix(rg_task)
    weight_mode1 = 'n'

  
    loss_train1  = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/loss_train_'+str(initial_model_1)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'.npy')
    i_train_1    = len(loss_train1)-1
    
    if weight_mode == 'n':
        PATH2            = "trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/trained_nn_l_"+str(initial_model)+'_'+str(max_iter_ADMM)+"_"+str(weight_mode)+".pt"
        PATH3            = "trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/trained_nn_i_"+str(initial_model)+'_'+str(max_iter_ADMM)+"_"+str(weight_mode)+".pt"
        nn_l             = compatible_torch_load(PATH2)
        nn_i             = compatible_torch_load(PATH3)
        # radius           = np.sqrt(rg_task[0]**2+rg_task[1]**2)# m
        nn_input         = np.reshape(rg_task[0:2]/max_radius*k_const,(2,1)) #dimensionless
        nn_l_output_task = convert_nn_l(nn_l(nn_input))
        P_weight1        = Grad_Solver.Set_Parameters_nn_l(nn_l_output_task)
        nn_i_output_task = convert_nn_i(nn_i(nn_input))
        P_weight2        = Grad_Solver.Set_Parameters_nn_i(nn_i_output_task)
    else:
        tunable_para_trained = np.load('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/tunable_para_trained_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
        weight           = Grad_Solver.Set_Parameters(tunable_para_trained)
        P_weight1        = weight[0:npl]
        P_weight2        = weight[npl:npauto]
    

    print('task_idx=',task_idx,'rg_task[m]=',rg_task,'Ql=',P_weight1[0:nxl],'QlN=',P_weight1[nxl:2*nxl],'Rl=',P_weight1[2*nxl:2*nxl+nul],'px=',P_weight1[-4],'gammax=',P_weight1[-2],'pu=',P_weight1[-3],'gammau=',P_weight1[-1])
    print('task_idx=',task_idx,'Qi=',P_weight2[0:nxi],'QiN=',P_weight2[nxi:2*nxi],'Ri=',P_weight2[2*nxi:2*nxi+nui],'pix=',P_weight2[-4],'gammaix=',P_weight2[-2],'piu=',P_weight2[-3],'gammaiu=',P_weight2[-1])
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Qi_'+str(initial_model)+'_'+str(task_idx),P_weight2[0:nxi])
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/QiN_'+str(initial_model)+'_'+str(task_idx),P_weight2[nxi:2*nxi])
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/Ri_'+str(initial_model)+'_'+str(task_idx),P_weight2[2*nxi:2*nxi+nui])
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/pix_'+str(initial_model)+'_'+str(task_idx),P_weight2[-4])
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gammaix_'+str(initial_model)+'_'+str(task_idx),P_weight2[-2])
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/piu_'+str(initial_model)+'_'+str(task_idx),P_weight2[-3])
    np.save('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/gammaiu_'+str(initial_model)+'_'+str(task_idx),P_weight2[-1])
    xl_train    = np.load('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/xl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    ul_train    = np.load('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/ul_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    scxl_train  = np.load('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/scxl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    scul_train  = np.load('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/scul_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    xc_train    = np.load('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/xc_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    uc_train    = np.load('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/uc_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    scxc_train  = np.load('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/scxc_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    scuc_train  = np.load('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/scuc_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    xl_traj     = xl_train[i_train]
    ul_traj     = ul_train[i_train]
    scxl_traj   = scxl_train[i_train]
    scul_traj   = scul_train[i_train]
    xq_traj     = xc_train[i_train]
    uq_traj     = uc_train[i_train]
    scxq_traj   = scxc_train[i_train]
    scuq_traj   = scuc_train[i_train]
    # System open-loop predicted trajectories
    Ref_xl      = np.zeros((nxl,horizon))
    Pl          = np.zeros((3,horizon))
    scPl        = np.zeros((3,horizon))
    Eulerl      = np.zeros((3,horizon))
    norm_2_Ql   = np.zeros(horizon)
    time        = 0
    for k in range(horizon):
        ref_xl, ref_ul  = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
        Ref_xl[:,k:k+1] = np.reshape(ref_xl,(nxl,1))
        Pl[:,k:k+1]     = np.reshape(xl_traj[task_idx][k,0:3],(3,1))
        scPl[:,k:k+1]   = np.reshape(scxl_traj[task_idx][k,0:3],(3,1))
        ql_k            = np.reshape(xl_traj[task_idx][k,6:10],(4,1))
        norm_2_Ql[k]    = LA.norm(ql_k)
        Rl_k            = sysm.q_2_rotation(ql_k)
        rl_k            = Rot.from_matrix(Rl_k)
        eulerl_k        = np.reshape(rl_k.as_euler('zyx',degrees=True),(3,1))
        Eulerl[:,k:k+1] = eulerl_k
        time           += dt

    Xq         = [] # list that stores all quadrotors' predicted trajectories
    Aq         = [] # list that stores all cable attachments' trajectories in the world frame
    scXq       = [] # list that stores all quadrotors' safe copy predicted trajectories
    refXq      = [] # list that stores all quadrotors' reference trajectories
    Tq         = np.zeros((nq,horizon))
    scTq       = np.zeros((nq,horizon))

    DI     = np.load('Planning_plots_meta_COM_Dyn/cable_direction_'+str(i_train_1)+'_'+str(task_idx)+'_'+str(initial_model_1)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'.npy')
 
    for i in range(nq):
        Pi     = np.zeros((3,horizon))
        scPi   = np.zeros((3,horizon))
        refPi  = np.zeros((3,horizon))
        ri     = np.reshape(MPC_load.ra[:,i],(3,1))
        ai     = np.zeros((3,horizon))
        for k in range(horizon):
            pl_k   = np.reshape(xl_traj[task_idx][k,0:3],(3,1))
            ql_k   = np.reshape(xl_traj[task_idx][k,6:10],(4,1))
            Rl_k   = sysm.q_2_rotation(ql_k)
            di_k   = np.reshape(xq_traj[task_idx][i][k,0:3],(3,1))
            ti_k   = xq_traj[task_idx][i][k,12]
            scdi_k = np.reshape(scxq_traj[task_idx][i][k,0:3],(3,1))
            scti_k = scxq_traj[task_idx][i][k,12]
            ai_k   = pl_k + Rl_k@ri
            pi_k   = ai_k + cl0*di_k
            scpi_k = ai_k + cl0*scdi_k
            ref_plk= np.reshape(Ref_xl[0:3,k],(3,1)) + ri
            refpi_k= ref_plk + cl0*np.reshape(DI[i][:,k],(3,1))
            ai[:,k:k+1] = ai_k
            Tq[i,k]= ti_k
            scTq[i,k] = scti_k
            Pi[:,k:k+1] = pi_k
            scPi[:,k:k+1] = scpi_k
            refPi[:,k:k+1]= refpi_k
        Xq    += [Pi]
        scXq  += [scPi]
        refXq += [refPi]
        Aq    += [ai]
    

    # Plots
    fig1, ax1 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax1.add_patch(obs1)
    ax1.add_patch(obs2)
    ax1.plot(Xq[0][0,:],Xq[0][1,:],label='1st quadrotor',linewidth=1)
    ax1.plot(scXq[0][0,:],scXq[0][1,:],label='1st quadrotor_safe copy',color='black',marker='.',markersize=1,linewidth=1)
    ax1.plot(refXq[0][0,:],refXq[0][1,:],label='1st quadrotor_ref',color='orange',marker='.',markersize=1,linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,color='blue',fill=False)
        ax1.add_patch(quad)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True)
    fig1.savefig('Planning_plots_multiagent_meta_COM_Dyn/quadrotor1_traj_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax2.add_patch(obs1)
    ax2.add_patch(obs2)
    ax2.plot(Xq[1][0,:],Xq[1][1,:],label='2nd quadrotor',linewidth=1)
    ax2.plot(scXq[1][0,:],scXq[1][1,:],label='2nd quadrotor_safe copy',color='black',marker='.',markersize=1,linewidth=1)
    ax2.plot(refXq[1][0,:],refXq[1][1,:],label='2nd quadrotor_ref',color='orange',marker='.',markersize=1,linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,color='blue',fill=False)
        ax2.add_patch(quad)
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True)
    fig2.savefig('Planning_plots_multiagent_meta_COM_Dyn/quadrotor2_traj_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax3.add_patch(obs1)
    ax3.add_patch(obs2)
    ax3.plot(Xq[2][0,:],Xq[2][1,:],label='3rd quadrotor',linewidth=1)
    ax3.plot(scXq[2][0,:],scXq[2][1,:],label='3rd quadrotor_safe copy',color='black',marker='.',markersize=1,linewidth=1)
    ax3.plot(refXq[2][0,:],refXq[2][1,:],label='3rd quadrotor_ref',color='orange',marker='.',markersize=1,linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,color='blue',fill=False)
        ax3.add_patch(quad)
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True)
    fig3.savefig('Planning_plots_multiagent_meta_COM_Dyn/quadrotor3_traj_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax4.add_patch(obs1)
    ax4.add_patch(obs2)
    ax4.plot(Xq[3][0,:],Xq[3][1,:],label='4th quadrotor',linewidth=1)
    ax4.plot(scXq[3][0,:],scXq[3][1,:],label='4th quadrotor_safe copy',color='black',marker='.',markersize=1,linewidth=1)
    ax4.plot(refXq[3][0,:],refXq[3][1,:],label='4th quadrotor_ref',color='orange',marker='.',markersize=1,linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,color='blue',fill=False)
        ax4.add_patch(quad)
    ax4.set_xlabel('x [m]')
    ax4.set_ylabel('y [m]')
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.grid(True)
    fig4.savefig('Planning_plots_multiagent_meta_COM_Dyn/quadrotor4_traj_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    
    fig5, ax5 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax5.add_patch(obs1)
    ax5.add_patch(obs2)
    ax5.plot(Ref_xl[0,:],Ref_xl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax5.plot(Pl[0,:],Pl[1,:],label='Planned',linewidth=1)
    ax5.plot(scPl[0,:],scPl[1,:],label='Planned_safe_copy',color='black',marker='.',markersize=1,linewidth=1)
    kt= 42
    for k in range(horizon):
        if k==2 or k==kt or k==98:
            #6 quadrotors
            quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
            ax5.add_patch(quad1)
            quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
            ax5.add_patch(quad2)
            quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
            ax5.add_patch(quad3)
            quad4  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,fill=False)
            ax5.add_patch(quad4)
            ax5.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
            ax5.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax5.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax5.plot([Xq[3][0,k],Aq[3][0,k]],[Xq[3][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax5.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax5.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax5.plot([Aq[2][0,k],Aq[3][0,k]],[Aq[2][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax5.plot([Aq[3][0,k],Aq[0][0,k]],[Aq[3][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
    
    ax5.set_xlabel('x [m]')
    ax5.set_ylabel('y [m]')
    ax5.set_aspect('equal')
    ax5.legend()
    ax5.grid(True)
    fig5.savefig('Planning_plots_multiagent_meta_COM_Dyn/system_traj_quadrotor_num6_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(kt)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig6, ax6 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax6.add_patch(obs1)
    ax6.add_patch(obs2)
    ax6.plot(Ref_xl[0,:],Ref_xl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax6.plot(Pl[0,:],Pl[1,:],label='Planned',linewidth=1)
    ax6.plot(scPl[0,:],scPl[1,:],label='Planned_safe_copy',color='black',marker='.',markersize=1,linewidth=1)
    kt= 44
    for k in range(horizon):
        if k==2 or k==kt or k==98:
            #6 quadrotors
            quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
            ax6.add_patch(quad1)
            quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
            ax6.add_patch(quad2)
            quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
            ax6.add_patch(quad3)
            quad4  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,fill=False)
            ax6.add_patch(quad4)
            ax6.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
            ax6.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax6.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax6.plot([Xq[3][0,k],Aq[3][0,k]],[Xq[3][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax6.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax6.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax6.plot([Aq[2][0,k],Aq[3][0,k]],[Aq[2][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax6.plot([Aq[3][0,k],Aq[0][0,k]],[Aq[3][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
    
    ax6.set_xlabel('x [m]')
    ax6.set_ylabel('y [m]')
    ax6.set_aspect('equal')
    ax6.legend()
    ax6.grid(True)
    fig6.savefig('Planning_plots_multiagent_meta_COM_Dyn/system_traj_quadrotor_num6_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(kt)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig7, ax7 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax7.add_patch(obs1)
    ax7.add_patch(obs2)
    ax7.plot(Ref_xl[0,:],Ref_xl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax7.plot(Pl[0,:],Pl[1,:],label='Planned',linewidth=1)
    ax7.plot(scPl[0,:],scPl[1,:],label='Planned_safe_copy',color='black',marker='.',markersize=1,linewidth=1)
    kt= 46
    for k in range(horizon):
        if k==2 or k==kt or k==98:
            #6 quadrotors
            quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
            ax7.add_patch(quad1)
            quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
            ax7.add_patch(quad2)
            quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
            ax7.add_patch(quad3)
            quad4  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,fill=False)
            ax7.add_patch(quad4)
            ax7.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Xq[3][0,k],Aq[3][0,k]],[Xq[3][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Aq[2][0,k],Aq[3][0,k]],[Aq[2][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax7.plot([Aq[3][0,k],Aq[0][0,k]],[Aq[3][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
    
    ax7.set_xlabel('x [m]')
    ax7.set_ylabel('y [m]')
    ax7.set_aspect('equal')
    ax7.legend()
    ax7.grid(True)
    fig7.savefig('Planning_plots_multiagent_meta_COM_Dyn/system_traj_quadrotor_num6_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(kt)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig8, ax8 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax8.add_patch(obs1)
    ax8.add_patch(obs2)
    ax8.plot(Ref_xl[0,:],Ref_xl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax8.plot(Pl[0,:],Pl[1,:],label='Planned',linewidth=1)
    ax8.plot(scPl[0,:],scPl[1,:],label='Planned_safe_copy',color='black',marker='.',markersize=1,linewidth=1)
    kt= 48
    for k in range(horizon):
        if k==2 or k==kt or k==98:
            #6 quadrotors
            quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
            ax8.add_patch(quad1)
            quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
            ax8.add_patch(quad2)
            quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
            ax8.add_patch(quad3)
            quad4  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,fill=False)
            ax8.add_patch(quad4)
            ax8.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
            ax8.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax8.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax8.plot([Xq[3][0,k],Aq[3][0,k]],[Xq[3][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax8.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax8.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax8.plot([Aq[2][0,k],Aq[3][0,k]],[Aq[2][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax8.plot([Aq[3][0,k],Aq[0][0,k]],[Aq[3][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
    
    ax8.set_xlabel('x [m]')
    ax8.set_ylabel('y [m]')
    ax8.set_aspect('equal')
    ax8.legend()
    ax8.grid(True)
    fig8.savefig('Planning_plots_multiagent_meta_COM_Dyn/system_traj_quadrotor_num6_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(kt)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig9, ax9 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax9.add_patch(obs1)
    ax9.add_patch(obs2)
    ax9.plot(Ref_xl[0,:],Ref_xl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax9.plot(Pl[0,:],Pl[1,:],label='Planned',linewidth=1)
    ax9.plot(scPl[0,:],scPl[1,:],label='Planned_safe_copy',color='black',marker='.',markersize=1,linewidth=1)
    kt= 50
    for k in range(horizon):
        if k==2 or k==kt or k==98:
            #6 quadrotors
            quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
            ax9.add_patch(quad1)
            quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
            ax9.add_patch(quad2)
            quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
            ax9.add_patch(quad3)
            quad4  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,fill=False)
            ax9.add_patch(quad4)
            ax9.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
            ax9.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax9.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax9.plot([Xq[3][0,k],Aq[3][0,k]],[Xq[3][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax9.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax9.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax9.plot([Aq[2][0,k],Aq[3][0,k]],[Aq[2][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax9.plot([Aq[3][0,k],Aq[0][0,k]],[Aq[3][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
    
    ax9.set_xlabel('x [m]')
    ax9.set_ylabel('y [m]')
    ax9.set_aspect('equal')
    ax9.legend()
    ax9.grid(True)
    fig9.savefig('Planning_plots_multiagent_meta_COM_Dyn/system_traj_quadrotor_num6_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(kt)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig10, ax10 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax10.add_patch(obs1)
    ax10.add_patch(obs2)
    ax10.plot(Ref_xl[0,:],Ref_xl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax10.plot(Pl[0,:],Pl[1,:],label='Planned',linewidth=1)
    ax10.plot(scPl[0,:],scPl[1,:],label='Planned_safe_copy',color='black',marker='.',markersize=1,linewidth=1)
    kt= 52
    for k in range(horizon):
        if k==2 or k==kt or k==98:
            #6 quadrotors
            quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
            ax10.add_patch(quad1)
            quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
            ax10.add_patch(quad2)
            quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
            ax10.add_patch(quad3)
            quad4  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,fill=False)
            ax10.add_patch(quad4)
            ax10.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
            ax10.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax10.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax10.plot([Xq[3][0,k],Aq[3][0,k]],[Xq[3][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax10.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax10.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax10.plot([Aq[2][0,k],Aq[3][0,k]],[Aq[2][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax10.plot([Aq[3][0,k],Aq[0][0,k]],[Aq[3][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
    
    ax10.set_xlabel('x [m]')
    ax10.set_ylabel('y [m]')
    ax10.set_aspect('equal')
    ax10.legend()
    ax10.grid(True)
    fig10.savefig('Planning_plots_multiagent_meta_COM_Dyn/system_traj_quadrotor_num6_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(kt)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig11, ax11 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax11.add_patch(obs1)
    ax11.add_patch(obs2)
    ax11.plot(Ref_xl[0,:],Ref_xl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax11.plot(Pl[0,:],Pl[1,:],label='Planned',linewidth=1)
    ax11.plot(scPl[0,:],scPl[1,:],label='Planned_safe_copy',color='black',marker='.',markersize=1,linewidth=1)
    kt= 54
    for k in range(horizon):
        if k==2 or k==kt or k==98:
            #6 quadrotors
            quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
            ax11.add_patch(quad1)
            quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
            ax11.add_patch(quad2)
            quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
            ax11.add_patch(quad3)
            quad4  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,fill=False)
            ax11.add_patch(quad4)
            ax11.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
            ax11.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax11.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax11.plot([Xq[3][0,k],Aq[3][0,k]],[Xq[3][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax11.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax11.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax11.plot([Aq[2][0,k],Aq[3][0,k]],[Aq[2][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax11.plot([Aq[3][0,k],Aq[0][0,k]],[Aq[3][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
    
    ax11.set_xlabel('x [m]')
    ax11.set_ylabel('y [m]')
    ax11.set_aspect('equal')
    ax11.legend()
    ax11.grid(True)
    fig11.savefig('Planning_plots_multiagent_meta_COM_Dyn/system_traj_quadrotor_num6_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(kt)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig12, ax12 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax12.add_patch(obs1)
    ax12.add_patch(obs2)
    ax12.plot(Ref_xl[0,:],Ref_xl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax12.plot(Pl[0,:],Pl[1,:],label='Planned',linewidth=1)
    ax12.plot(scPl[0,:],scPl[1,:],label='Planned_safe_copy',color='black',marker='.',markersize=1,linewidth=1)
    kt= 56
    for k in range(horizon):
        if k==2 or k==kt or k==98:
            #6 quadrotors
            quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
            ax12.add_patch(quad1)
            quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
            ax12.add_patch(quad2)
            quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
            ax12.add_patch(quad3)
            quad4  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,fill=False)
            ax12.add_patch(quad4)
            ax12.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
            ax12.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax12.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax12.plot([Xq[3][0,k],Aq[3][0,k]],[Xq[3][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax12.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax12.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax12.plot([Aq[2][0,k],Aq[3][0,k]],[Aq[2][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax12.plot([Aq[3][0,k],Aq[0][0,k]],[Aq[3][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
    
    ax12.set_xlabel('x [m]')
    ax12.set_ylabel('y [m]')
    ax12.set_aspect('equal')
    ax12.legend()
    ax12.grid(True)
    fig12.savefig('Planning_plots_multiagent_meta_COM_Dyn/system_traj_quadrotor_num6_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(kt)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig13, ax13 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax13.add_patch(obs1)
    ax13.add_patch(obs2)
    ax13.plot(Ref_xl[0,:],Ref_xl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax13.plot(Pl[0,:],Pl[1,:],label='Planned',linewidth=1)
    ax13.plot(scPl[0,:],scPl[1,:],label='Planned_safe_copy',color='black',marker='.',markersize=1,linewidth=1)
    kt= 58
    for k in range(horizon):
        if k==2 or k==kt or k==98:
            #6 quadrotors
            quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
            ax13.add_patch(quad1)
            quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
            ax13.add_patch(quad2)
            quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
            ax13.add_patch(quad3)
            quad4  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,fill=False)
            ax13.add_patch(quad4)
            ax13.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
            ax13.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax13.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax13.plot([Xq[3][0,k],Aq[3][0,k]],[Xq[3][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax13.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
            ax13.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
            ax13.plot([Aq[2][0,k],Aq[3][0,k]],[Aq[2][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
            ax13.plot([Aq[3][0,k],Aq[0][0,k]],[Aq[3][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
    
    ax13.set_xlabel('x [m]')
    ax13.set_ylabel('y [m]')
    ax13.set_aspect('equal')
    ax13.legend()
    ax13.grid(True)
    fig13.savefig('Planning_plots_multiagent_meta_COM_Dyn/system_traj_quadrotor_num6_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(kt)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(14,figsize=(6,4),dpi=300)
    plt.plot(Time,Tq[0,:],linewidth=1,label='1st cable')
    plt.plot(Time,Tq[1,:],linewidth=1,label='2nd cable')
    plt.plot(Time,Tq[2,:],linewidth=1,label='3rd cable')
    plt.plot(Time,Tq[3,:],linewidth=1,label='4th cable')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('MPC tension force [N]')
    plt.grid()
    plt.savefig('Planning_plots_multiagent_meta_COM_Dyn/cable_MPC_tensions_6_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(15,figsize=(6,4),dpi=300)
    plt.plot(Time,Eulerl[0,:],linewidth=1,label='roll')
    plt.plot(Time,Eulerl[1,:],linewidth=1,label='pitch')
    plt.plot(Time,Eulerl[2,:],linewidth=1,label='yaw')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Euler angle [deg]')
    plt.grid()
    plt.savefig('Planning_plots_multiagent_meta_COM_Dyn/euler_6_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(16,figsize=(6,4),dpi=300)
    plt.plot(Time,Pl[2,:],linewidth=1,label='actual')
    plt.plot(Time,Ref_xl[2,:],linewidth=0.5,linestyle='--',label='desired')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.grid()
    plt.savefig('Planning_plots_multiagent_meta_COM_Dyn/height_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(17,figsize=(6,4),dpi=300)
    plt.plot(Time,uq_traj[task_idx][0][:,0],linewidth=1,label='actual dddwx')
    plt.plot(Time,scuq_traj[task_idx][0][:,0],linewidth=1,label='safe dddwx')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('uc_x')
    plt.grid()
    plt.savefig('Planning_plots_multiagent_meta_COM_Dyn/uc_x_quadrotor_1_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(18,figsize=(6,4),dpi=300)
    plt.plot(Time,uq_traj[task_idx][0][:,1],linewidth=1,label='actual dddwy')
    plt.plot(Time,scuq_traj[task_idx][0][:,1],linewidth=1,label='safe dddwy')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('uc_y')
    plt.grid()
    plt.savefig('Planning_plots_multiagent_meta_COM_Dyn/uc_y_quadrotor_1_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(19,figsize=(6,4),dpi=300)
    plt.plot(Time,uq_traj[task_idx][0][:,2],linewidth=1,label='actual dddwz')
    plt.plot(Time,scuq_traj[task_idx][0][:,2],linewidth=1,label='safe dddwz')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('uc_z')
    plt.grid()
    plt.savefig('Planning_plots_multiagent_meta_COM_Dyn/uc_z_quadrotor_1_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(20,figsize=(6,4),dpi=300)
    plt.plot(Time,uq_traj[task_idx][0][:,3],linewidth=1,label='actual ddt')
    plt.plot(Time,scuq_traj[task_idx][0][:,3],linewidth=1,label='safe ddt')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('ddt')
    plt.grid()
    plt.savefig('Planning_plots_multiagent_meta_COM_Dyn/ddt_quadrotor_1_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()


"""---------------------------------Main function-----------------------------"""





if mode =="t":
    train(m0,v0,lr0,lr_nn,Tunable_para0,NN_l,NN_i,wt0,wrp0,max_iter_ADMM,initial_model)
else:
    loss_train = np.load('trained_data_multiagent_meta_COM_Dyn (best_backup_useThis,middle_network_px_pu,k_const=10,LR=0.2)/loss_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    task_index = input("enter 0, 1, ..., 9")
    evaluate(len(loss_train)-1,int(task_index),max_iter_ADMM,initial_model)
    # evaluate(0,int(task_index),max_iter_ADMM,initial_model)
    # evaluate(1)
