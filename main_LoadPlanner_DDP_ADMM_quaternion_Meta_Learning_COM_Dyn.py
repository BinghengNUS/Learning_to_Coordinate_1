"""
Main function of the load planner (Tension Allocation)
------------------------------------------------------
1st version, Dr. Wang Bingheng, 19-Dec-2024
2nd version, Dr. Wang Bingheng, 30-Mar-2025
3rd version, Dr. Wang Bingheng, 10-Nov-2025
"""

from casadi import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import Dynamics_meta_learning_COM_Dyn
import Optimal_Allocation_DDP_quaternion_autotuning_ADMM_COM_Dyn
import math
import time as TM
from scipy.spatial.transform import Rotation as Rot
import os
import Neural_network
import torch


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


if not os.path.exists("trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)"):
    os.makedirs("trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)")

print("=============================================")
print("Main code for training or evaluating Automultilift")
print("Please choose mode")
mode = input("enter 't' or 'e' without the quotation mark, t: training; e: evaluation")
if mode == 't':
    print("Should we generate new initial models randomly?")
    initial_mode = input("enter 'y' or 'n' without the quotation mark, y: yes; n: no")
print("Please choose initial model")
initial_model = int(input("enter '0', '1', '2', '3' or '4' without the quotation mark"))
print("Please choose ADMM truncation number for stage 1")
max_iter_ADMM = int(input("enter '2', '3', '4', or '5' without the quotation mark"))
print("Please choose weight_mode")
weight_mode = input("enter 'n' or 'f' without the quotation mark, n: neural network; f: fixed")
print("=============================================")

m1        = 0.45  # the load's net weight [kg], a circular basket with uniform mass distribution
m2        = 0.25  # the added mass [kg]
mtot      = m1+m2 # the total weight [kg]
nq        = 4     # the number of quadrotors
cl0       = 1     # the cable length [m]
rq        = 0.15  # the radius of quadrotor [m]
rl        = 0.25  # the radius of the load [m]
ro        = 0.65  # the radius of obstacle [m]
"""--------------------------------------Load Environment---------------------------------------"""
r_inertia = 1     # in training, we use a large rotational inertia for high efficiency. In evaluation, we can reduce this value to the normal one.
sysm_para = np.array([m1, m2, 
                      r_inertia*1/4*m1*rl**2, r_inertia*1/4*m1*rl**2, r_inertia*1/2*m1*rl**2, 
                      rl, nq, rq, cl0, ro])
dt        = 0.04 # for small rotational inertia, the time-step should be very small. Otherwise, one will observe DDP divergence!
sysm      = Dynamics_meta_learning_COM_Dyn.multilift_model(sysm_para,dt)
rp0       = np.array([[0.05,0.05,0]]).T # the initial 
sysm.Rotational_Inertia(rp0)
sysm.model()
nxl       = sysm.nxl # dimension of the load's state
nul       = 3*nq # total dimension of the load's control = 6 (wrench) + 3*6-6 (null-space vector)
nWl       = sysm.nWl
k_const   = 10 # 1-20

"""--------------------------------------Define Planner---------------------------------------"""
horizon    = 100
e_abs, e_rel = 1e-4, 1e-3
MPC_load   = Optimal_Allocation_DDP_quaternion_autotuning_ADMM_COM_Dyn.MPC_Planner(sysm_para, dt, horizon, e_abs, e_rel)
# pob1, pob2 = np.array([[1.7,1.3]]).T, np.array([[0.3,3.1]]).T # planar positions of the two obstacle in the world frame
pob1, pob2 = np.array([[1.7,1.15]]).T, np.array([[0.3,3.05]]).T
print('obstacle_distance=',LA.norm(pob1-pob2))
rg0        = m2/mtot*rp0
MPC_load.allocation_martrix(rg0)
MPC_load.SetStateVariable(sysm.xl)
MPC_load.SetCtrlVariable(sysm.Wl)
MPC_load.SetDyn(sysm.model_l)
MPC_load.SetLearnablePara()
MPC_load.SetConstraints_ADMM_Subp2(pob1,pob2)
MPC_load.SetCostDyn_ADMM(max_iter_ADMM)
MPC_load.ADMM_SubP2_Init()
MPC_load.system_derivatives_DDP_ADMM()
MPC_load.system_derivatives_SubP2_ADMM()
MPC_load.system_derivatives_SubP3_ADMM()

# define the network size
D_in, D_h1, D_h2, D_out = 1, 16, 32, MPC_load.n_Pauto  # D_out = 39, 13*2+6+3+4 
# D_in, D_h1, D_h2, D_out = 1, 8, 16, MPC_load.n_Pauto  # D_out = 39, 13*2+6+3+4
def convert_nn(nn_i_outcolumn):
    # convert a column tensor to a row np.array
    nn_i_row = np.zeros((1,D_out))
    for i in range(D_out):
        nn_i_row[0,i] = nn_i_outcolumn[i,0]
    return nn_i_row



# generate a list that saves random load center-of-mass coordinates for tasks
num_task   = 10
max_radius = 0.15  # reference length [m]
# if mode == 't':
#     rp_task   = []
#     for _ in range(num_task):
#         rp        = np.random.uniform(0,max_radius) # in training, we do not make it very large for the concern of stable training
#         alpha     = np.random.uniform(0,2*np.pi)
#         random_rp = np.array([[rp*np.cos(alpha),rp*np.sin(alpha),0]]).T # unit: [m]
#         rp_task  += [random_rp] # in this stage, we CANNOT normalize it as it is needed in the load dynamics!
#     print('rp_task=',rp_task)
#     np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/rp_task',rp_task)
#use the saved rg_task
rp_task = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/rp_task.npy')
# parameters of RMSProp
lr0       = 0.25 # 0.25
lr_nn     = lr0
epsilon   = 1e-8
v0        = np.zeros(MPC_load.n_Pauto)

# parameters of ADAM
m0        = np.zeros(MPC_load.n_Pauto)
beta1     = 0.9 # 
beta2     = 0.999 # default settings

"""--------------------------------------Redefine Gradient Solver---------------------------------------"""
Grad_Solver = Optimal_Allocation_DDP_quaternion_autotuning_ADMM_COM_Dyn.Gradient_Solver(horizon,sysm.xl,sysm.Wl,MPC_load.sc_xl,MPC_load.sc_Wl,MPC_load.P1_l,MPC_load.P_auto)

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
# initial palyload's state
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
# np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/xl_init_'+str(max_iter_ADMM),xl_init)
xl_init    = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/xl_init_'+str(max_iter_ADMM)+'.npy')


# it has no physical meaning and only scales the numerical representation of the input data 
# it helps the neural network to learn the mapping between the input data and the ADMM-DDP hyperparameters.
# In evaluation, we can use the same value of k.
# initial weights of the meta-loss
wt0, wrp0 = 1, 1


# if mode == 't':
#     if initial_mode == 'y':
#         Tunable_para0 = []
#         NN_waypoint   = []
#         for _ in range(5): # generate 5 candidates
#             # initialization, the std is tuned to generate an initial loss compariable to that of the trajectories optimized using the neural adaptive hyperpara.
#             Tunable_para0 += [np.random.normal(0,0.1,MPC_load.n_Pauto)]
#             NN_waypoint += [Neural_network.Net(D_in,D_h1,D_h2,D_out)]
#         np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Tunable_para0',Tunable_para0) 
#         PATHl_init    = "trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/initial_NN_waypoint.pt"   
#         torch.save(NN_waypoint,PATHl_init)

PATHl_init    = "trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/initial_NN_waypoint.pt"   
Tunable_para0 = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Tunable_para0.npy')
NN_waypoint   = compatible_torch_load(PATHl_init) #

# Solve the load's MPC planner
def train(m0,v0,lr0,Tunable_para0,NN_waypoint,max_iter_ADMM,wt0,wrp0,initial_model):
    tunable_para  = Tunable_para0[initial_model]
    nn_waypoint   = NN_waypoint[initial_model]
    wt            = wt0
    wrp           = wrp0
    i             = 1
    i_max         = 50
    delta_loss    = 1e2
    loss0         = 1e2
    epi           = 1e-1
    xl_train      = []
    Wl_train      = []
    scxl_train    = []
    scWl_train    = []
    Tl_train      = []
    loss_train    = []
    wloss_train   = []
    losst_train   = []
    lossrp_train  = []
    Wt            = []
    Wrp           = []
    K_auto        = []
    iter_train    = []
    gradtimeOur   = []
    gradtimePDP   = []
    gradtimeCao   = []
    gradtimeCaos  = []
    meanerrorPDP  = []
    meanerrorCao  = []
    gMeanerror_load = []
    start_time1   = TM.time()
    v             = v0
    m             = m0
    
    optimizer   = torch.optim.Adam(nn_waypoint.parameters(),lr=lr_nn,betas=(beta1, beta2),eps=1e-08,weight_decay=0)
    while delta_loss>epi and i<i_max:
    # for i in range(10): # for comparing gradient computation time
        task_loss    = 0
        taskt_loss   = 0
        taskrp_loss  = 0
        taskt_gloss  = 0
        taskrp_gloss = 0
        task_grad    = 0
        task_loss_nn = 0
        xl_task    = []
        Wl_task    = []
        scxl_task  = []
        scWl_task  = []
        Tl_task    = []
        Wt        += [wt]
        Wrp       += [wrp]
        Loss_task  = []
        wLoss_task = []
        for task_idx in range(num_task):
            sysm.Rotational_Inertia(rp_task[task_idx])
            sysm.model()
            rg_task = m2/mtot*rp_task[task_idx] # keep its unit [m]
            MPC_load.allocation_martrix(rg_task)
            MPC_load.SetDyn(sysm.model_l)
            MPC_load.SetConstraints_ADMM_Subp2(pob1,pob2)
            MPC_load.SetCostDyn_ADMM(max_iter_ADMM)
            MPC_load.ADMM_SubP2_Init()
            MPC_load.system_derivatives_DDP_ADMM()
            MPC_load.system_derivatives_SubP2_ADMM()
            Ref_xl  = np.zeros(nxl*(horizon+1))
            Ref_Wl  = np.zeros(nWl*horizon)
            Time    = []
            time    = 0
            for k in range(horizon):
                Time  += [time]
                ref_xl, ref_Wl = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
                Ref_xl[k*nxl:(k+1)*nxl] = ref_xl
                Ref_Wl[k*nWl:(k+1)*nWl] = ref_Wl
                time += dt
            # Time  += [time]
            ref_xl, ref_Wl = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
            Ref_xl[horizon*nxl:(horizon+1)*nxl] = ref_xl
            # generate the corresponding hyperparameters, given the task rg
            radius     = np.sqrt(rg_task[0]**2+rg_task[1]**2)# m
            if weight_mode == 'n':
                nn_input   = np.reshape(radius/max_radius*k_const,(1,1)) #dimensionless
                nn_output_task = convert_nn(nn_waypoint(nn_input))
                weight     = Grad_Solver.Set_Parameters_nn(nn_output_task)
            else:
                weight     = Grad_Solver.Set_Parameters(tunable_para)
            p_weight1  = weight[0:MPC_load.n_P1]
            p_weight2  = weight[MPC_load.n_P1:MPC_load.n_P1 + MPC_load.n_P2]
            px         = p_weight1[-4]
            pu         = p_weight1[-3]
            gammax     = p_weight1[-2]
            gammau     = p_weight1[-1]
            
            print('iter_train=',i,'task_idx=',task_idx,'Q=',p_weight1[0:MPC_load.n_xl],'QN=',p_weight1[MPC_load.n_xl:2*MPC_load.n_xl],'R=',p_weight1[2*MPC_load.n_xl:2*MPC_load.n_xl+MPC_load.n_Wl])
            print('iter_train=',i,'task_idx=',task_idx,'nv_w=',p_weight2[0:MPC_load.n_P2],'px=',px,'gammax=',gammax,'pu=',pu,'gammau=',gammau,'rg_radius (m) =',radius)
            print('iter_train=',i,'task_idx=',task_idx,'nv_w/px=',p_weight2[0:MPC_load.n_P2]/px,'nv_w/pu=',p_weight2[0:MPC_load.n_P2]/pu,'weightmode:',weight_mode,'initial_model=',initial_model)
            start_time = TM.time()
            xl_init_task    = np.zeros(nxl)
            xl_init_task[0] = xl_init[0] + rg_task[0]
            xl_init_task[1] = xl_init[1] + rg_task[1]
            xl_init_task[2:nxl] = xl_init[2:nxl]
            Opt_Sol1, Opt_Sol2, Opt_Y, Opt_Eta  = MPC_load.ADMM_forward_MPC_DDP(xl_init_task,Ref_xl,Ref_Wl,p_weight1,p_weight2,max_iter_ADMM)
            mpctime    = (TM.time() - start_time)*1000
            print("a:--- %s ms ---" % format(mpctime,'.2f'))
            # start_time = TM.time()
            Grad_Out1, Grad_Out2, Grad_Out3, GradTime, GradTimePDP, GradTimeCao, GradTimeCaos, MeanerrorCao, MeanerrorPDP, gMeanerror_l = MPC_load.ADMM_Gradient_Solver(Opt_Sol1,Opt_Sol2,Opt_Y,Opt_Eta,Ref_xl,Ref_Wl,p_weight1,p_weight2)
            gradtime    = (TM.time() - start_time)*1000
            print("g:--- %s ms ---" % format(gradtime,'.2f'))
            gradtimeOur   += [GradTime[-1]]
            gradtimePDP   += [GradTimePDP[-1]]
            gradtimeCao   += [GradTimeCao[-1]]
            gradtimeCaos  += [GradTimeCaos[-1]]
            meanerrorCao  += [MeanerrorCao[-1]]
            meanerrorPDP  += [MeanerrorPDP[-1]]
            gMeanerror_load += [gMeanerror_l]
            dldw, loss, loss_track, loss_rp, gloss_t, gloss_rp  = Grad_Solver.ChainRule(Opt_Sol1, Opt_Sol2, Ref_xl, Grad_Out1, Grad_Out2, wt, wrp)
            task_loss   += loss[0]
            taskt_loss  += loss_track[0]
            taskrp_loss += loss_rp[0]
            taskt_gloss += gloss_t
            taskrp_gloss+= gloss_rp
            Loss_task      += [loss_track[0]+loss_rp[0]]

            if weight_mode == 'n':
                dwdp        = Grad_Solver.ChainRule_Gradient_nn(nn_output_task)
                dldp        = np.reshape(dldw@dwdp,(1,MPC_load.n_Pauto))
                loss_nn     = nn_waypoint.myloss(nn_waypoint(nn_input),dldp)
                task_loss_nn += loss_nn
                task_grad  += np.reshape(dldp,MPC_load.n_Pauto)
            else:
                dwdp        = Grad_Solver.ChainRule_Gradient(tunable_para)
                dldp        = np.reshape(dldw@dwdp,MPC_load.n_Pauto)
                task_grad  += dldp
            xl_task    += [Opt_Sol1[-1]['xl_opt']]
            Wl_task    += [Opt_Sol1[-1]['Wl_opt']]
            scxl_task  += [Opt_Sol2[-1]['scxl_opt']]
            scWl_task  += [Opt_Sol2[-1]['scWl_opt']]
            Tl_task    += [Opt_Sol2[-1]['Tl_opt']]
            
            print('iter_train=',i,'task_idx=',task_idx,'loss_task=',loss_track+loss_rp)          
        
        if weight_mode == 'n':
            optimizer.zero_grad()
            avg_loss_nn = task_loss_nn/num_task
            avg_loss_nn.backward()
            optimizer.step()
            avg_grad    = task_grad/num_task
        else:
            avg_grad    = task_grad/num_task
            for k in range(int(MPC_load.n_Pauto)):
                m[k]    = beta1*m[k] + (1-beta1)*avg_grad[k]
                m_hat   = m[k]/(1-beta1**i)
                v[k]    = beta2*v[k] + (1-beta2)*avg_grad[k]**2
                v_hat   = v[k]/(1-beta2**i)
                lr      = lr0/(np.sqrt(v_hat+epsilon))
                tunable_para[k] = tunable_para[k] - lr*m_hat

        avg_loss    = task_loss/num_task
        avg_losst   = taskt_loss/num_task
        avg_lossrp  = taskrp_loss/num_task
        avg_glosst  = taskt_gloss/num_task
        avg_glossrp = taskrp_gloss/num_task
        # update thw weights of the meta-loss
        wt, wrp, k_auto = Grad_Solver.adaptive_meta_loss_weights(avg_losst,avg_lossrp,avg_glosst,avg_glossrp,wt)
        loss_train   += [avg_losst+avg_lossrp]
        wloss_train  += [avg_loss]
        losst_train  += [avg_losst]
        lossrp_train += [avg_lossrp]
        xl_train  += [xl_task]
        Wl_train  += [Wl_task]
        scxl_train+= [scxl_task]
        scWl_train+= [scWl_task]
        Tl_train  += [Tl_task]
        iter_train += [i]
        K_auto     += [k_auto]
        if i==1:
            epi = 1e-3*(avg_losst+avg_lossrp)
        if i>10: # enough training
            delta_loss = abs(avg_losst+avg_lossrp-loss0)
        loss0      = avg_losst+avg_lossrp
        print('iter_train=',i,'loss=',avg_losst+avg_lossrp,'loss_t=',avg_losst,'loss_rp=',avg_lossrp,'loss_train=',loss_train,'wt=',wt,'wrp=',wrp,'kauto=',k_auto,'medain of Loss_task=',np.median(Loss_task),'std of Loss_task=',np.std(Loss_task))
        print('iter_train=',i,'dldpQ=',avg_grad[0:MPC_load.n_xl],'dldpR=',avg_grad[2*MPC_load.n_xl:2*MPC_load.n_xl+MPC_load.n_Wl],'dldpNv=',avg_grad[MPC_load.n_P1:MPC_load.n_P1+MPC_load.n_P2],'dldppx=',avg_grad[MPC_load.n_P1-4],'dldpgammax=',avg_grad[MPC_load.n_P1-2],'dldppu=',avg_grad[MPC_load.n_P1-3],'dldpgammau=',avg_grad[MPC_load.n_P1-1],'weightmode:',weight_mode,'initial_model:',initial_model)
        i += 1


        # below is the code for saving the trajectory optimization results using the last updated neural network
        if delta_loss <=epi or i>i_max:
            task_loss    = 0
            taskt_loss   = 0
            taskrp_loss  = 0
            task_grad    = 0
            task_loss_nn = 0
            xl_task    = []
            Wl_task    = []
            scxl_task  = []
            scWl_task  = []
            Tl_task    = []
            Loss_task  = []
            wLoss_task = []
            Losst_task = []
            Lossrp_task = []
            for task_idx in range(num_task):
                sysm.Rotational_Inertia(rp_task[task_idx])
                sysm.model()
                rg_task = m2/mtot*rp_task[task_idx] # keep its unit [m]
                MPC_load.allocation_martrix(rg_task)
                MPC_load.SetDyn(sysm.model_l)
                MPC_load.SetConstraints_ADMM_Subp2(pob1,pob2)
                MPC_load.SetCostDyn_ADMM(max_iter_ADMM)
                MPC_load.ADMM_SubP2_Init()
                MPC_load.system_derivatives_DDP_ADMM()
                MPC_load.system_derivatives_SubP2_ADMM()
                Ref_xl  = np.zeros(nxl*(horizon+1))
                Ref_Wl  = np.zeros(nWl*horizon)
                Time    = []
                time    = 0
                for k in range(horizon):
                    Time  += [time]
                    ref_xl, ref_Wl = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
                    Ref_xl[k*nxl:(k+1)*nxl] = ref_xl
                    Ref_Wl[k*nWl:(k+1)*nWl] = ref_Wl
                    time += dt
                # Time  += [time]
                ref_xl, ref_Wl = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
                Ref_xl[horizon*nxl:(horizon+1)*nxl] = ref_xl
                # generate the corresponding hyperparameters, given the task rg
                radius     = np.sqrt(rg_task[0]**2+rg_task[1]**2)# m
                if weight_mode == 'n':
                    nn_input   = np.reshape(radius/max_radius*k_const,(1,1))
                    nn_output_task = convert_nn(nn_waypoint(nn_input))
                    weight     = Grad_Solver.Set_Parameters_nn(nn_output_task)
                else:
                    weight     = Grad_Solver.Set_Parameters(tunable_para)
                p_weight1  = weight[0:MPC_load.n_P1]
                p_weight2  = weight[MPC_load.n_P1:MPC_load.n_P1 + MPC_load.n_P2]
                px         = p_weight1[-4]
                pu         = p_weight1[-3]
                gammax     = p_weight1[-2]
                gammau     = p_weight1[-1]
                print('iter_train=',i,'task_idx=',task_idx,'Q=',p_weight1[0:MPC_load.n_xl],'QN=',p_weight1[MPC_load.n_xl:2*MPC_load.n_xl],'R=',p_weight1[2*MPC_load.n_xl:2*MPC_load.n_xl+MPC_load.n_Wl])
                print('iter_train=',i,'task_idx=',task_idx,'nv_w=',p_weight2[0:MPC_load.n_P2],'px=',px,'gammax=',gammax,'pu=',pu,'gammau=',gammau,'rp_task(m)=',rp_task[task_idx],'rg_radius (m) =',radius)
                print('iter_train=',i,'task_idx=',task_idx,'nv_w/px=',p_weight2[0:MPC_load.n_P2]/px,'nv_w/pu=',p_weight2[0:MPC_load.n_P2]/pu,'weightmode:',weight_mode,'initial_model=',initial_model)
                start_time = TM.time()
                xl_init_task    = np.zeros(nxl)
                xl_init_task[0] = xl_init[0] + rg_task[0]
                xl_init_task[1] = xl_init[1] + rg_task[1]
                xl_init_task[2:nxl] = xl_init[2:nxl]
                Opt_Sol1, Opt_Sol2, Opt_Y, Opt_Eta  = MPC_load.ADMM_forward_MPC_DDP(xl_init_task,Ref_xl,Ref_Wl,p_weight1,p_weight2,max_iter_ADMM)
                mpctime    = (TM.time() - start_time)*1000
                print("a:--- %s ms ---" % format(mpctime,'.2f'))
                # start_time = TM.time()
                Grad_Out1, Grad_Out2, Grad_Out3, GradTime, GradTimePDP, GradTimeCao, GradTimeCaos, MeanerrorCao, MeanerrorPDP, gMeanerror_l = MPC_load.ADMM_Gradient_Solver(Opt_Sol1,Opt_Sol2,Opt_Y,Opt_Eta,Ref_xl,Ref_Wl,p_weight1,p_weight2)
                gradtime    = (TM.time() - start_time)*1000
                print("g:--- %s ms ---" % format(gradtime,'.2f'))
                gradtimeOur   += [GradTime[-1]]
                gradtimePDP   += [GradTimePDP[-1]]
                gradtimeCao   += [GradTimeCao[-1]]
                gradtimeCaos  += [GradTimeCaos[-1]]
                meanerrorCao  += [MeanerrorCao[-1]]
                meanerrorPDP  += [MeanerrorPDP[-1]]
                gMeanerror_load += [gMeanerror_l]
                dldw, loss, loss_track, loss_rp, gloss_t, gloss_rp  = Grad_Solver.ChainRule(Opt_Sol1, Opt_Sol2, Ref_xl, Grad_Out1, Grad_Out2, wt, wrp)
                task_loss   += loss[0]
                taskt_loss  += loss_track[0]
                taskrp_loss += loss_rp[0]
                Loss_task      += [loss_track[0]+loss_rp[0]]
                wLoss_task     += [loss[0]]
                Losst_task     += [loss_track[0]]
                Lossrp_task    += [loss_rp[0]]
                if weight_mode == 'n':
                    dwdp        = Grad_Solver.ChainRule_Gradient_nn(nn_output_task)
                    dldp        = np.reshape(dldw@dwdp,(1,MPC_load.n_Pauto))
                    loss_nn     = nn_waypoint.myloss(nn_waypoint(nn_input),dldp)
                    task_loss_nn += loss_nn
                    task_grad  += np.reshape(dldp,MPC_load.n_Pauto)
                else:
                    dwdp        = Grad_Solver.ChainRule_Gradient(tunable_para)
                    dldp        = np.reshape(dldw@dwdp,MPC_load.n_Pauto)
                    task_grad  += dldp
                xl_task    += [Opt_Sol1[-1]['xl_opt']]
                Wl_task    += [Opt_Sol1[-1]['Wl_opt']]
                scxl_task  += [Opt_Sol2[-1]['scxl_opt']]
                scWl_task  += [Opt_Sol2[-1]['scWl_opt']]
                Tl_task    += [Opt_Sol2[-1]['Tl_opt']]
            
                print('iter_train=',i,'task_idx=',task_idx,'loss_task=',loss_track+loss_rp)          
        
            
            avg_loss    = task_loss/num_task
            avg_losst   = taskt_loss/num_task
            avg_lossrp  = taskrp_loss/num_task
     
            loss_train   += [avg_losst+avg_lossrp]
            wloss_train  += [avg_loss]
            losst_train  += [avg_losst]
            lossrp_train += [avg_lossrp]
            xl_train  += [xl_task]
            Wl_train  += [Wl_task]
            scxl_train+= [scxl_task]
            scWl_train+= [scWl_task]
            Tl_train  += [Tl_task]
            iter_train += [i]
            delta_loss = abs(avg_losst+avg_lossrp - loss0)
            if delta_loss > epi:
                if weight_mode == 'n':
                    optimizer.zero_grad()
                    avg_loss_nn = task_loss_nn/num_task
                    avg_loss_nn.backward()
                    optimizer.step()
                    avg_grad    = task_grad/num_task
                else:
                    avg_grad    = task_grad/num_task
                    for k in range(int(MPC_load.n_Pauto)):
                        m[k]    = beta1*m[k] + (1-beta1)*avg_grad[k]
                        m_hat   = m[k]/(1-beta1**i)
                        v[k]    = beta2*v[k] + (1-beta2)*avg_grad[k]**2
                        v_hat   = v[k]/(1-beta2**i)
                        lr      = lr0/(np.sqrt(v_hat+epsilon))
                        tunable_para[k] = tunable_para[k] - lr*m_hat
            
            print('iter_train=',i,'loss=',avg_losst+avg_lossrp,'loss_t=',avg_losst,'loss_rp=',avg_lossrp,'loss_train=',loss_train,'wt=',wt,'wrp=',wrp,'weightmode:',weight_mode,'initial_model=',initial_model,'medain of Loss_task=',np.median(Loss_task),'std of Loss_task=',np.std(Loss_task),'wloss_train=',wloss_train,'median of wloss_task=',np.median(wLoss_task),'std of wLoss_task=',np.std(wLoss_task))
            i += 1
    traintime    = (TM.time() - start_time1)
    print("train:--- %s s ---" % format(traintime,'.2f'))
    # save the trained network models
    if weight_mode == 'n':
        PATH2   = "trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/trained_nn_waypoint_"+str(initial_model)+'_'+str(max_iter_ADMM)+"_"+str(weight_mode)+".pt"
        torch.save(nn_waypoint,PATH2)
    else:
        np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/tunable_para_trained_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),tunable_para)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/loss_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),loss_train)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/wloss_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),wloss_train)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/losst_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),losst_train)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/lossrp_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),lossrp_train)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Losst_task_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Losst_task)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Lossrp_task_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Lossrp_task)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Wt_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Wt)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Wrp_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Wrp)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Kauto_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),K_auto)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/xl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),xl_train)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/scxl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),scxl_train)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Wl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Wl_train)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/scWl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),scWl_train)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Tl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Tl_train)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/gradtimeOur_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimeOur)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/gradtimePDP_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimePDP)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/gradtimeCao_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimeCao)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/gradtimeCaos_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gradtimeCaos)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/meanerrorCao_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),meanerrorCao)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/meanerrorPDP_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),meanerrorPDP)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/gMeanerrorload_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),gMeanerror_load)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Final_loss_task_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Loss_task)
    np.save('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Grad_Out1_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Grad_Out1)

    plt.figure(1,figsize=(6,4),dpi=400)
    plt.plot(wloss_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('wLoss')
    plt.grid()
    plt.savefig('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/wloss_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(2,figsize=(6,4),dpi=400)
    plt.plot(loss_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/loss_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(3,figsize=(6,4),dpi=400)
    plt.plot(losst_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Loss_track')
    plt.grid()
    plt.savefig('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/losst_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(4,figsize=(6,4),dpi=400)
    plt.plot(lossrp_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Loss_residual')
    plt.grid()
    plt.savefig('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/lossrp_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(5,figsize=(6,4),dpi=400)
    plt.plot(Wt, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Wt')
    plt.grid()
    plt.savefig('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Wt_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(6,figsize=(6,4),dpi=400)
    plt.plot(Wrp, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Wrp')
    plt.grid()
    plt.savefig('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Wrp_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()

    plt.figure(7,figsize=(6,4),dpi=400)
    plt.plot(K_auto, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('K_auto')
    plt.grid()
    plt.savefig('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Kauto_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=300)
    plt.show()



def evaluate(i_train,task_idx,initial_model):
    Ref_xl = np.zeros((nxl,horizon))
    rp_task = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/rp_task.npy')
    rg_task  = m2/mtot*rp_task[task_idx]
    MPC_load.allocation_martrix(rg_task)
    time   = 0
    for k in range(horizon):
        ref_xl, ref_Wl = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
        Ref_xl[:,k:(k+1)] = np.reshape(ref_xl,(nxl,1))
        time += dt

    radius     = np.sqrt(rg_task[0]**2+rg_task[1]**2)# m
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([Neural_network.Net])
    if not os.path.exists("Planning_plots_meta_COM_Dyn"):
        os.makedirs("Planning_plots_meta_COM_Dyn")
    if weight_mode == 'n':
        PATH2   = "trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/trained_nn_waypoint_"+str(initial_model)+'_'+str(max_iter_ADMM)+"_"+str(weight_mode)+".pt"
        nn_waypoint = compatible_torch_load(PATH2)
        nn_input   = np.reshape(radius/max_radius*k_const,(1,1))
        nn_output  = convert_nn(nn_waypoint(nn_input))
        weight     = Grad_Solver.Set_Parameters_nn(nn_output)
    else:
        tunable_para = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/tunable_para_trained_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
        weight     = Grad_Solver.Set_Parameters(tunable_para)
    np.save('Planning_plots_meta_COM_Dyn/weight_'+str(task_idx),weight)
    p_weight1  = weight[0:MPC_load.n_P1]
    p_weight2  = weight[MPC_load.n_P1:MPC_load.n_P1 + MPC_load.n_P2]
    px         = p_weight1[-4]
    pu         = p_weight1[-3]
    gammax     = p_weight1[-2]
    gammau     = p_weight1[-1]
    disx       = MPC_load.Discount_rate(gammax,max_iter_ADMM-1,max_iter_ADMM)
    disu       = MPC_load.Discount_rate(gammau,max_iter_ADMM-1,max_iter_ADMM)
    print('task_idx=',task_idx,'Q=',p_weight1[0:nxl],'QN=',p_weight1[nxl:2*nxl],'R=',p_weight1[2*nxl:2*nxl+nWl])
    print('task_idx=',task_idx,'nv_w=',p_weight2[0:MPC_load.n_P2],'px=',px,'gammax=',gammax,'pu=',pu,'gammau=',gammau,'rg_radius [m] =',radius,'rg_task [m] =',rg_task)
    print('task_idx=',task_idx,'nv_w/(disx*px)=',p_weight2[0:MPC_load.n_P2]/(disx*px),'nv_w/(disu*pu)=',p_weight2[0:MPC_load.n_P2]/(disu*pu))
    np.save('Planning_plots_meta_COM_Dyn/p_weight_ratio_x_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(task_idx),p_weight2[0:MPC_load.n_P2]/px)
    np.save('Planning_plots_meta_COM_Dyn/p_weight_ratio_u_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(task_idx),p_weight2[0:MPC_load.n_P2]/pu)
    xl_train    = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/xl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    scxl_train  = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/scxl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    Wl_train    = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Wl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    scWl_train  = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/scWl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    Tl_train    = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/Tl_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    xl_opt      = xl_train[i_train]
    scxl_opt    = scxl_train[i_train]
    Wl_opt      = Wl_train[i_train]
    scWl_opt    = scWl_train[i_train]
    Tl_opt      = Tl_train[i_train]
    # System open-loop predicted trajectories
    P_pinv      = MPC_load.P_pinv # pseudo-inverse of P matrix
    P_ns        = MPC_load.P_ns # null-space of P matrix
    Pl          = np.zeros((3,horizon))
    scPl        = np.zeros((3,horizon))
    Euler_l     = np.zeros((3,horizon))
    norm_2_Ql   = np.zeros(horizon)
    for k in range(horizon):
        Pl[:,k:k+1]   = np.reshape(xl_opt[task_idx][k,0:3],(3,1))
        scPl[:,k:k+1] = np.reshape(scxl_opt[task_idx][k,0:3],(3,1))
        ql_k  = np.reshape(xl_opt[task_idx][k,6:10],(4,1))
        norm_2_Ql[k]  = LA.norm(ql_k)
        Rl_k          = sysm.q_2_rotation(ql_k)
        rk            = Rot.from_matrix(Rl_k)
        euler_k       = np.reshape(rk.as_euler('xyz',degrees=True),(3,1))
        Euler_l[:,k:k+1] = euler_k 
    Xq         = [] # list that stores all quadrotors' predicted trajectories
    DI         = [] # list that stores all cables' direction trajectories
    Aq         = [] # list that stores all cable attachments' trajectories in the world frame
    Tq         = np.zeros((nq,horizon))
    for i in range(nq):
        Pi     = np.zeros((3,horizon))
        di     = np.zeros((3,horizon))
        # ri     = np.array([[rl*math.cos(i*alpha),rl*math.sin(i*alpha),0]]).T- np.reshape(np.vstack((rg_task[0],rg_task[1],0)),(3,1)) 
        ri     = np.reshape(MPC_load.ra[:,i],(3,1))
        ai     = np.zeros((3,horizon))
        for k in range(horizon):
            pl_k  = np.reshape(xl_opt[task_idx][k,0:3],(3,1)) # the position of the load's COM in the world frame
            ql_k  = np.reshape(xl_opt[task_idx][k,6:10],(4,1))
            Rl_k  = sysm.q_2_rotation(ql_k)
            Fl_k  = np.reshape(Wl_opt[task_idx][k,0:3],(3,1)) # world frame
            Ml_k  = np.reshape(Wl_opt[task_idx][k,3:6],(3,1)) # body frame
            wl_k  = np.reshape(np.vstack((Rl_k.T@Fl_k,Ml_k)),(6,1))
            nv_k  = np.reshape(Tl_opt[task_idx][k,:],(3*nq-6,1)) # 3-D null-space vector at the kth step
            t_k   = P_pinv@wl_k + P_ns@nv_k # 9-D tension vector at the kth step in the load's body frame
            ti_k  = np.reshape(t_k[3*i:3*(i+1)],(3,1))
            pi_k  = pl_k + Rl_k@(ri + cl0*ti_k/LA.norm(ti_k))
            di_k  = Rl_k@ti_k/LA.norm(ti_k)
            ai_k  = pl_k + Rl_k@ri
            Pi[:,k:k+1] = pi_k
            di[:,k:k+1] = di_k # inertial frame
            ai[:,k:k+1] = ai_k
            Tq[i,k] = LA.norm(ti_k)
        Xq += [Pi]
        DI += [di]
        Aq += [ai]

    # Save data
    np.save('Planning_plots_meta_COM_Dyn/tension_magnitude_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),Tq)
    np.save('Planning_plots_meta_COM_Dyn/cable_direction_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode),DI)
    

    print('norm of quaternion=',norm_2_Ql)
    
    # Plots

    fig1, ax1 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax1.add_patch(obs1)
    ax1.add_patch(obs2)
    ax1.plot(Xq[0][0,:],Xq[0][1,:],label='1st quadrotor',linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,color='blue',fill=False)
        ax1.add_patch(quad)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True)
    fig1.savefig('Planning_plots_meta_COM_Dyn/quadrotor1_traj_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax2.add_patch(obs1)
    ax2.add_patch(obs2)
    ax2.plot(Xq[1][0,:],Xq[1][1,:],label='2nd quadrotor',linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,color='blue',fill=False)
        ax2.add_patch(quad)
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True)
    fig2.savefig('Planning_plots_meta_COM_Dyn/quadrotor2_traj_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax3.add_patch(obs1)
    ax3.add_patch(obs2)
    ax3.plot(Xq[2][0,:],Xq[2][1,:],label='3rd quadrotor',linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,color='blue',fill=False)
        ax3.add_patch(quad)
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True)
    fig3.savefig('Planning_plots_meta_COM_Dyn/quadrotor3_traj_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax4.add_patch(obs1)
    ax4.add_patch(obs2)
    ax4.plot(Xq[3][0,:],Xq[3][1,:],label='4th quadrotor',linewidth=1)
    for k in range(horizon):
        quad  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,color='blue',fill=False)
        ax4.add_patch(quad)
    ax4.set_xlabel('x [m]')
    ax4.set_ylabel('y [m]')
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.grid(True)
    fig4.savefig('Planning_plots_meta_COM_Dyn/quadrotor4_traj_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    if nq==6:
        fig5, ax5 = plt.subplots(figsize=(5,5),dpi=300)
        obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
        obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
        ax5.add_patch(obs1)
        ax5.add_patch(obs2)
        ax5.plot(Xq[4][0,:],Xq[4][1,:],label='5th quadrotor',linewidth=1)
        for k in range(horizon):
            quad  = Circle((Xq[4][0,k],Xq[4][1,k]),rq,color='blue',fill=False)
            ax5.add_patch(quad)
        ax5.set_xlabel('x [m]')
        ax5.set_ylabel('y [m]')
        ax5.set_aspect('equal')
        ax5.legend()
        ax5.grid(True)
        fig5.savefig('Planning_plots_meta_COM_Dyn/quadrotor5_traj_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
        plt.show()

        fig6, ax6 = plt.subplots(figsize=(5,5),dpi=300)
        obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
        obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
        ax6.add_patch(obs1)
        ax6.add_patch(obs2)
        ax6.plot(Xq[5][0,:],Xq[5][1,:],label='6th quadrotor',linewidth=1)
        for k in range(horizon):
            quad  = Circle((Xq[5][0,k],Xq[5][1,k]),rq,color='blue',fill=False)
            ax6.add_patch(quad)
        ax6.set_xlabel('x [m]')
        ax6.set_ylabel('y [m]')
        ax6.set_aspect('equal')
        ax6.legend()
        ax6.grid(True)
        fig6.savefig('Planning_plots_meta_COM_Dyn/quadrotor6_traj_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
        plt.show()


    fig7, ax7 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax7.add_patch(obs1)
    ax7.add_patch(obs2)
    ax7.plot(Ref_xl[0,:],Ref_xl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax7.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
    ax7.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
    for k in range(horizon):
        
        if k==1 or k==35 or k==50 or k==65 or k==horizon-1:
            if nq ==4:
                # four quadrotors
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

            else:
                # six quadrotors
                quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
                ax7.add_patch(quad1)
                quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
                ax7.add_patch(quad2)
                quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
                ax7.add_patch(quad3)
                quad4  = Circle((Xq[3][0,k],Xq[3][1,k]),rq,fill=False)
                ax7.add_patch(quad4)
                quad5  = Circle((Xq[4][0,k],Xq[4][1,k]),rq,fill=False)
                ax7.add_patch(quad5)
                quad6  = Circle((Xq[5][0,k],Xq[5][1,k]),rq,fill=False)
                ax7.add_patch(quad6)
                ax7.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
                ax7.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
                ax7.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
                ax7.plot([Xq[3][0,k],Aq[3][0,k]],[Xq[3][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
                ax7.plot([Xq[4][0,k],Aq[4][0,k]],[Xq[4][1,k],Aq[4][1,k]],color='blue',linewidth=0.5)
                ax7.plot([Xq[5][0,k],Aq[5][0,k]],[Xq[5][1,k],Aq[5][1,k]],color='blue',linewidth=0.5)
                ax7.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
                ax7.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
                ax7.plot([Aq[2][0,k],Aq[3][0,k]],[Aq[2][1,k],Aq[3][1,k]],color='blue',linewidth=0.5)
                ax7.plot([Aq[3][0,k],Aq[4][0,k]],[Aq[3][1,k],Aq[4][1,k]],color='blue',linewidth=0.5)
                ax7.plot([Aq[4][0,k],Aq[5][0,k]],[Aq[4][1,k],Aq[5][1,k]],color='blue',linewidth=0.5)
                ax7.plot([Aq[5][0,k],Aq[0][0,k]],[Aq[5][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)


    ax7.set_xlabel('x [m]')
    ax7.set_ylabel('y [m]')
    ax7.set_aspect('equal')
    ax7.legend()
    ax7.grid(True)
    fig7.savefig('Planning_plots_meta_COM_Dyn/system_traj_quadrotor_num4_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()


    plt.figure(8,figsize=(6,4),dpi=300)
    plt.plot(Time,Tq[0,:],linewidth=1,label='1st cable')
    plt.plot(Time,Tq[1,:],linewidth=1,label='2nd cable')
    plt.plot(Time,Tq[2,:],linewidth=1,label='3rd cable')
    plt.plot(Time,Tq[3,:],linewidth=1,label='4th cable')
    if nq==6:
        plt.plot(Time,Tq[4,:],linewidth=1,label='5th cable')
        plt.plot(Time,Tq[5,:],linewidth=1,label='6th cable')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('MPC tension force [N]')
    plt.grid()
    plt.savefig('Planning_plots_meta_COM_Dyn/cable_MPC_tensions_4_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()


    plt.figure(9,figsize=(6,4),dpi=300)
    plt.plot(Time,Euler_l[0,:],linewidth=1,label='roll')
    plt.plot(Time,Euler_l[1,:],linewidth=1,label='pitch')
    plt.plot(Time,Euler_l[2,:],linewidth=1,label='yaw')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Euler angle [deg]')
    plt.grid()
    plt.savefig('Planning_plots_meta_COM_Dyn/euler_4_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(10,figsize=(6,4),dpi=300)
    plt.plot(Time,Pl[2,:],linewidth=1,label='actual height')
    plt.plot(Time,Ref_xl[2,:],linewidth=1,label='desired height')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.grid()
    plt.savefig('Planning_plots_meta_COM_Dyn/height_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(11,figsize=(6,4),dpi=300)
    plt.plot(Time,Wl_opt[task_idx][:,0],linewidth=1,label='actual Fx')
    plt.plot(Time,scWl_opt[task_idx][:,0],linewidth=1,label='safe Fx')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Fx [N]')
    plt.grid()
    # plt.savefig('Planning_plots_meta_COM_Dyn/Fx_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(12,figsize=(6,4),dpi=300)
    plt.plot(Time,Wl_opt[task_idx][:,1],linewidth=1,label='actual Fy')
    plt.plot(Time,scWl_opt[task_idx][:,1],linewidth=1,label='safe Fy')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Fy [N]')
    plt.grid()
    # plt.savefig('Planning_plots_meta_COM_Dyn/Fx_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(13,figsize=(6,4),dpi=300)
    plt.plot(Time,Wl_opt[task_idx][:,2],linewidth=1,label='actual Fz')
    plt.plot(Time,scWl_opt[task_idx][:,2],linewidth=1,label='safe Fz')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Fz [N]')
    plt.grid()
    # plt.savefig('Planning_plots_meta_COM_Dyn/Fx_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(14,figsize=(6,4),dpi=300)
    plt.plot(Time,Wl_opt[task_idx][:,3],linewidth=1,label='actual Mx')
    plt.plot(Time,scWl_opt[task_idx][:,3],linewidth=1,label='safe Mx')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Mx [Nm]')
    plt.grid()
    # plt.savefig('Planning_plots_meta_COM_Dyn/Mx_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(15,figsize=(6,4),dpi=300)
    plt.plot(Time,Wl_opt[task_idx][:,4],linewidth=1,label='actual My')
    plt.plot(Time,scWl_opt[task_idx][:,4],linewidth=1,label='safe My')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('My [Nm]')
    plt.grid()
    # plt.savefig('Planning_plots_meta_COM_Dyn/My_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(16,figsize=(6,4),dpi=300)
    plt.plot(Time,Wl_opt[task_idx][:,5],linewidth=1,label='actual Mz')
    plt.plot(Time,scWl_opt[task_idx][:,5],linewidth=1,label='safe Mz')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Mz [Nm]')
    plt.grid()
    # plt.savefig('Planning_plots_meta_COM_Dyn/Mz_ddp_admm_'+str(i_train)+'_'+str(task_idx)+'_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()


"""---------------------------------Main function-----------------------------"""


if mode =="t":
    train(m0,v0,lr0,Tunable_para0,NN_waypoint,max_iter_ADMM,wt0,wrp0,initial_model)
else:
    loss_train = np.load('trained_data_meta_COM_Dyn (good_useThis_latest_new_sigmoid_pxpu,IPOPT_no_adaptive)/loss_train_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'.npy')
    print("Please choose task index")
    task_index = input("enter 0, 1, ..., 9")
    # evaluate(len(loss_train)-1,int(task_index),initial_model)
    evaluate(0,int(task_index),initial_model)
    # evaluate(0)
    # evaluate(4)
