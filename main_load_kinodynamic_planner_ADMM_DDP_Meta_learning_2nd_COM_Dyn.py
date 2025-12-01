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

max_iter_ADMM = 3 # try 2, 3, 4, and 5
max_iter_ADMM_1 = 3
print("=============================================")
print("Main code for training or evaluating Automultilift")
print("Please choose mode")
mode = input("enter 't' or 'e' without the quotation mark, t: training; e: evaluation")

print("Please choose weight_mode")
weight_mode = input("enter 'n' or 'f' without the quotation mark, n: neural network; f: fixed")
print("=============================================")

if not os.path.exists("trained_data_multiagent_meta_COM_Dyn"):
    os.makedirs("trained_data_multiagent_meta_COM_Dyn")

m1        = 0.5   # the load's net weight [kg], a circular basket with uniform mass distribution
m2        = 0.2   # the added mass [kg]
mq        = 0.25  # the quadrotor mass [kg]
fmax      = 10    # the quadrotor maximum thrust [N]
vmax      = 2     # the quadrotor maximum velocity [m/s]
mtot      = m1+m2 # the total weight [kg]
nq        = 4     # the number of quadrotors
cl0       = 1     # the cable length [m]
rq        = 0.15  # the radius of quadrotor [m]
rl        = 0.25  # the radius of the load [m]
ro        = 0.65  # the radius of obstacle [m]

"""--------------------------------------Load Environment---------------------------------------"""
sysm_para = np.array([m1, m2, 
                      1/4*m1*rl**2, 1/4*m1*rl**2, 1/2*m1*rl**2, 
                      rl, nq, rq, mq, fmax, vmax,
                      cl0, ro])
# for a load with a biased CoM, different quadrotors need different weights
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
pob1, pob2 = np.array([[1.7,1.25]]).T, np.array([[0.3,3.15]]).T
MPC_load  = Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn.MPC_Planner(sysm_para,dt,horizon)
MPC_load.Rotational_Inertia(rp0)
rg0       = m2/mtot*rp0
MPC_load.allocation_martrix(rg0)
MPC_load.SetStateVariables(sysm.xl,sysm.xi)
MPC_load.SetCtrlVariables(sysm.ul,sysm.ui)
MPC_load.SetDyns(sysm.model_l,sysm.model_i)
MPC_load.SetWeightPara()
MPC_load.SetPayloadCostDyn()
MPC_load.SetCableCostDyn()
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

D_inl, D_h1l, D_h2l, D_outl = 2, 16, 32, MPC_load.npl 
def convert_nn_l(nn_l_outcolumn):
    # convert a column tensor to a row np.array
    nn_l_row = np.zeros((1,D_outl))
    for i in range(D_outl):
        nn_l_row[0,i] = nn_l_outcolumn[i,0]
    return nn_l_row

D_ini, D_h1i, D_h2i, D_outi = 2, 10, 20, MPC_load.npi
def convert_nn_i(nn_i_outcolumn):
    # convert a column tensor to a row np.array
    nn_i_row = np.zeros((1,D_outi))
    for i in range(D_outi):
        nn_i_row[0,i] = nn_i_outcolumn[i,0]
    return nn_i_row

num_task   = 10
rp_task    = np.load('trained_data_meta_COM_Dyn/rp_task.npy') # unit: [m]
max_radius = 0.15  # reference length [m]
# parameters of ADAM
lr0       = 0.25
lr_nn     = lr0 # even reducing the learning rate to 0.1 cannot stabilize the training process with the pair (0.01, 0.02)
coefficient = 1 # dimensionless
epsilon   = 1e-8
m0        = np.zeros(MPC_load.n_Pauto)
v0        = np.zeros(MPC_load.n_Pauto)
mt0       = 0
vt0       = 0
mrp0      = 0
vrp0      = 0
beta1     = 0.95 
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
xl_init = np.load('trained_data_meta_COM_Dyn/xl_init.npy') # 1-dim array

# MPC parameters
# MPC weights (learnable parameters, now manually tuned)
tunable_para0 = np.random.normal(0,0.1,npauto) # initialization
wt0, wrp0     = 1, 1 

def train(m0,v0,mt0,vt0,mrp0,vrp0,lr0,lr_nn,tunable_para0,wt0,wrp0,max_iter_ADMM,adaptiveADMM):
    tunable_para   = tunable_para0
    wt             = wt0
    wrp            = wrp0
    i_iter         = 1
    loss_max       = 10
    i_max          = 1e2
    delta_loss     = 1e2
    loss0          = 1e2
    epi            = 1e-1
    xl_train       = []
    Kfbl_train     = []
    scxl_train     = []
    xc_train       = []
    uc_train       = []
    scxc_train     = []
    loss_train     = []
    losst_train    = []
    lossrp_train   = []
    iter_train     = []
    Wt             = []
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
    start_time1  = TM.time()
    m            = m0
    v            = v0
    mt           = mt0
    vt           = vt0
    mrp          = mrp0
    vrp          = vrp0
    PATHl_init   = "trained_data_multiagent_meta_COM_Dyn/initial_nn_l.pt"
    PATHi_init   = "trained_data_multiagent_meta_COM_Dyn/initial_nn_i.pt"
    # nn_l = Neural_network.Net(D_inl,D_h1l,D_h2l,D_outl)
    # torch.save(nn_l,PATHl_init)
    # nn_i = Neural_network.Net(D_ini,D_h1i,D_h2i,D_outi)
    # torch.save(nn_i,PATHi_init)
    # use the saved initial model for training
    nn_l       = torch.load(PATHl_init, weights_only=False)
    nn_i       = torch.load(PATHi_init, weights_only=False)
    pl_min       = Grad_Solver.p_min
    pi_min       = Grad_Solver.p_min
    p_min        = Grad_Solver.p_min
    norm_dldw    = 0
    avg_loss     = 500
    optimizer_l  = torch.optim.Adam(nn_l.parameters(),lr=lr_nn,betas=(beta1, beta2),eps=1e-08,weight_decay=0) 
    optimizer_i  = torch.optim.Adam(nn_i.parameters(),lr=lr_nn,betas=(beta1, beta2),eps=1e-08,weight_decay=0)
    # optimizer_l= torch.optim.AdamW(nn_l.parameters(),lr=lr_nn,weight_decay=1e-4,betas=(beta1, beta2)) # can also lead to an increase of the loss
    # optimizer_i= torch.optim.AdamW(nn_i.parameters(),lr=lr_nn,weight_decay=1e-4,betas=(beta1, beta2))
    weight_mode1 = 'n'
    adaptiveADMM1 = 'a' 
    loss_train1  = np.load('trained_data_meta_COM_Dyn/loss_train_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'_'+str(adaptiveADMM1)+'.npy')
    i_train_1    = len(loss_train1)-1
    
    while delta_loss>epi and i_iter<=i_max:
        Wt           += [wt]
        task_loss     = 0 # meta-loss
        task_losst    = 0 # tracking error loss
        task_lossrp   = 0 # ADMM residual loss
        task_grad     = 0 # grad w.r.t. ADMM-DDP hyperparameters
        task_loss_nnl = 0 # meta-loss for the load
        task_loss_nni = 0 # meta-loss for the cables
        xl_task    = []
        Kfbl_task  = []
        scxl_task  = []
        xc_task    = []
        uc_task    = []
        scxc_task  = []
        for task_idx in range(num_task):
            sysm.Rotational_Inertia(rp_task[task_idx]) # [m]
            sysm.model()
            MPC_load.Rotational_Inertia(rp_task[task_idx]) # [m]
            rg_task = m2/mtot*rp_task[task_idx]
            MPC_load.allocation_martrix(rg_task) # [m]
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
            DI     = np.load('Planning_plots_meta_COM_Dyn/cable_direction_'+str(i_train_1)+'_'+str(task_idx)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'_'+str(adaptiveADMM1)+'.npy')
            TI     = np.load('Planning_plots_meta_COM_Dyn/tension_magnitude_'+str(i_train_1)+'_'+str(task_idx)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'_'+str(adaptiveADMM1)+'.npy')
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
                    ref_ti_k = np.reshape(TI[i,k],(1,1))
                    ref_dti_k= np.zeros((1,1))
                    ref_xi_k = np.reshape(np.vstack((ref_di_k,ref_wi_k,ref_ti_k,ref_dti_k)),nxi)
                    ref_xc[i][k*nxi:(k+1)*nxi] = ref_xi_k
                ref_uq = np.zeros(int(nq)*nui)
            # the reference in the terminal stage
            ref_xl, ref_ul = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
            for i in range(nq):
                ref_di_N = np.reshape(DI[i][:,-1],(3,1))
                ref_wi_N = np.zeros((3,1))
                ref_ti_N = np.reshape(TI[i,-1],(1,1))
                ref_dti_N= np.zeros((1,1))
                ref_xi_N = np.reshape(np.vstack((ref_di_N,ref_wi_N,ref_ti_N,ref_dti_N)),nxi)
                ref_xc[i][horizon*nxi:(horizon+1)*nxi]=ref_xi_N
            Ref_xl[horizon*nxl:(horizon+1)*nxl] = ref_xl
            # the initial cable state (world frame)
            xq_init = np.zeros(nq*nxi) 
            for i in range(nq):
                di_init   = np.reshape(DI[i][:,0],(3,1)) 
                wi_init   = np.zeros((3,1))
                ti_init   = np.array([[TI[i,0]]])
                dti_init  = np.zeros((1,1))
                xi_init   = np.reshape(np.vstack((di_init,wi_init,ti_init,dti_init)),nxi)
                xq_init[i*nxi:(i+1)*nxi] = xi_init
            # generate the corresponding hyperparameters, given the task rg
            if weight_mode == 'n':
                nn_input         = np.reshape(rg_task[0:2]/max_radius,(2,1)) # dimensionless
                nn_l_output_task = convert_nn_l(nn_l(nn_input))
                # if norm_dldw>=loss_max: # a heuristic method to enhance learning stability
                #     pl_min = np.clip(pl_min+1e-3,Grad_Solver.p_min,1e-1) # 1e-2, 2e-1 work well!
                #     pi_min = np.clip(pi_min+1e-3,Grad_Solver.p_min,1e-1)
                P_weight1        = Grad_Solver.Set_Parameters_nn_l(nn_l_output_task,pl_min)
                nn_i_output_task = convert_nn_i(nn_i(nn_input))
                P_weight2        = Grad_Solver.Set_Parameters_nn_i(nn_i_output_task,pi_min)
                print('iter_train=',i_iter,'rg_task [m]:',rg_task,'task_idx=',task_idx,'pl_min=',pl_min,'pi_min=',pi_min)
                np.save('trained_data_multiagent_meta_COM_Dyn/pl_min',pl_min)
                np.save('trained_data_multiagent_meta_COM_Dyn/pi_min',pi_min)
            else:
                # if norm_dldw>=loss_max:
                #     p_min  = np.clip(p_min+1e-3,Grad_Solver.p_min,1e-1)
                weight     = Grad_Solver.Set_Parameters(tunable_para,p_min)
                P_weight1  = weight[0:npl]
                P_weight2  = weight[npl:npauto]
                print('iter_train=',i_iter,'task_idx=',task_idx,'p_min=',p_min)
                np.save('trained_data_multiagent_meta_COM_Dyn/p_min',p_min)

            print('iter_train=',i_iter,'task_idx=',task_idx,'Ql=',P_weight1[0:nxl],'QlN=',P_weight1[nxl:2*nxl],'Rl=',P_weight1[2*nxl:2*nxl+nul],'p=',P_weight1[-1],'lr_nn=',lr_nn)
            print('iter_train=',i_iter,'task_idx=',task_idx,'Qi=',P_weight2[0:nxi],'QiN=',P_weight2[nxi:2*nxi],'Ri=',P_weight2[2*nxi:2*nxi+nui],'pi=',P_weight2[-1])
            start_time = TM.time()
            opt_sol, Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3 = MPC_load.ADMM_forward_MPC(Ref_xl,Ref_ul,ref_xc,ref_uq,xl_init,xq_init,P_weight1,P_weight2,max_iter_ADMM,adaptiveADMM)
            mpctime    = (TM.time() - start_time)*1000
            print("forward mpc:--- %s ms ---" % format(mpctime,'.2f'))
            start_time = TM.time()
            Grad_Out1l, Grad_Out1c, Grad_Out2, Grad_Out3, GradTime, GradTimeCaos, GradTimeCao,  GradTimePDP,  GradTime_c, GradTimeCaos_c, GradTimeCao_c, GradTimePDP_c,  MeanerrorCao, MeanerrorPDP, MeanerrorCao_c, MeanerrorPDP_c   = MPC_load.ADMM_Gradient_Solver(Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3, Ref_xl, Ref_ul, ref_xc, ref_uq, P_weight1, P_weight2, adaptiveADMM)
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
            dldw, loss, loss_track, loss_resid  = Grad_Solver.ChainRule(Opt_Sol1_l,Opt_Sol1_c,Opt_Sol2,Ref_xl,ref_xc,Grad_Out1l,Grad_Out1c,Grad_Out2,wt,wrp)
            task_loss      += loss[0]
            task_losst     += loss_track[0]
            task_lossrp    += loss_resid[0]
            
            norm_dldw       = LA.norm(dldw)
            # if norm_dldw>=loss_max: # to enhence trainnig stability
            #     dldw = dldw/norm_dldw*loss_max
            norm_dldw_new = LA.norm(dldw)
            print('iter_train=',i_iter,'task_idx=',task_idx,'norm_dldw=',norm_dldw,'norm_dldw_new=',norm_dldw_new,'loss_max=',loss_max)
            if weight_mode == 'n':
                dwdpl        = Grad_Solver.ChainRule_Gradient_nn_l(nn_l_output_task,pl_min)
                dldwl        = np.reshape(dldw[0,0:npl],(1,npl))
                dldpl        = np.reshape(dldwl@dwdpl,(1,npl))
                loss_nn_l    = nn_l.myloss(nn_l(nn_input),dldpl)
                task_loss_nnl += loss_nn_l
                dldpl        = np.reshape(dldpl,npl)
                dwdpi        = Grad_Solver.ChainRule_Gradient_nn_i(nn_i_output_task,pi_min)
                dldwi        = np.reshape(dldw[0,npl:npauto],(1,npi))
                dldpi        = np.reshape(dldwi@dwdpi,(1,npi))
                loss_nn_i    = nn_i.myloss(nn_i(nn_input),dldpi)
                task_loss_nni += loss_nn_i
                dldpi        = np.reshape(dldpi,npi)
                dldp         = np.append(dldpl,dldpi)
                task_grad  += dldp
            else:
                dwdp        = Grad_Solver.ChainRule_Gradient(tunable_para,p_min)
                dldp        = np.reshape(dldw@dwdp,npauto)
                task_grad  += dldp
            xl_task    += [opt_sol['xl_traj']]
            Kfbl_task  += [opt_sol['Kfbl_traj']]
            scxl_task  += [opt_sol['scxl_traj']]
            xc_task    += [opt_sol['xc_traj']]
            uc_task    += [opt_sol['uc_traj']]
            scxc_task  += [opt_sol['scxc_traj']]

        if weight_mode == 'n':
            optimizer_l.zero_grad()
            avg_loss_nn_l = task_loss_nnl/num_task
            avg_loss_nn_l.backward()
            # for p in nn_l.parameters():
            #     if p.grad is not None:
            #         p.grad += 0.01 * torch.randn_like(p.grad)  # noise scale can be tuned
            # torch.nn.utils.clip_grad_norm_(nn_l.parameters(), 3.0)
            optimizer_l.step()
            optimizer_i.zero_grad()
            avg_loss_nn_i = task_loss_nni/num_task
            avg_loss_nn_i.backward()
            # for p in nn_i.parameters():
            #     if p.grad is not None:
            #         p.grad += 0.02 * torch.randn_like(p.grad)  # noise scale can be tuned
            # torch.nn.utils.clip_grad_norm_(nn_i.parameters(), 3.0)
            optimizer_i.step()
            avg_grad    = task_grad/num_task
            
        else:
            avg_grad    = task_grad/num_task
            # ADAM adaptive learning
            for k in range(int(npauto)):
                m[k]    = beta1*m[k] + (1-beta1)*avg_grad[k]
                m_hat   = m[k]/(1-beta1**i_iter)
                v[k]    = beta2*v[k] + (1-beta2)*avg_grad[k]**2
                v_hat   = v[k]/(1-beta2**i_iter)
                lr      = lr0/(np.sqrt(v_hat)+epsilon)
                tunable_para[k] = tunable_para[k] - lr*m_hat
        # update the weights in the meta-loss
        avg_losst   = task_losst/num_task
        avg_lossrp  = task_lossrp/num_task
        wt   = Grad_Solver.adaptive_meta_loss_weights(avg_losst,avg_lossrp,wt)
        # avg_gradt   = task_gradt/num_task
        # avg_gradrp  = task_gradrp/num_task
        # # ADAM adaptive learning for updating wt
        # mt      = beta1*mt + (1-beta1)*avg_gradt
        # mt_hat  = mt/(1-beta1**i_iter)
        # vt      = beta2*vt + (1-beta2)*avg_gradt**2
        # vt_hat  = vt/(1-beta2**i_iter)
        # lrt     = lr0/(np.sqrt(vt_hat)+epsilon)
        # wt      = wt - lrt*mt_hat
        # # ADAM adaptive learning for updating wrp
        # mrp     = beta1*mrp + (1-beta1)*avg_gradrp
        # mrp_hat = mrp/(1-beta1**i_iter)
        # vrp     = beta2*vrp + (1-beta2)*avg_gradrp**2
        # vrp_hat = vrp/(1-beta2**i_iter)
        # lrrp    = lr0/(np.sqrt(vrp_hat)+epsilon)
        # wrp     = wrp - lrrp*mrp_hat

        avg_loss    = task_loss/num_task
        
        loss_train += [avg_loss]
        losst_train += [avg_losst]
        lossrp_train += [avg_lossrp]
        xl_train   += [xl_task]
        Kfbl_train += [Kfbl_task]
        scxl_train += [scxl_task]
        xc_train   += [xc_task]
        uc_train   += [uc_task]
        scxc_train += [scxc_task]
        iter_train += [i_iter]
        if i_iter==1:
            epi = 1e-3*avg_loss
        if i_iter>2:
            delta_loss = abs(avg_loss-loss0)
        loss0      = avg_loss
        dldp1      = avg_grad[0:npl]
        dldp2      = avg_grad[npl:npauto]
        print('iter_train=',i_iter,'loss=',avg_loss,'loss_t=',avg_losst,'loss_rp=',avg_lossrp,'loss_train=',loss_train,'wt=',wt,'wrp=',wrp)
        print('iter_train=',i_iter,'dldpQl=',dldp1[0:nxl],'dldpRl=',dldp1[2*nxl:2*nxl+nul],'dldpp=',dldp1[-1])
        print('iter_train=',i_iter,'dldpQi=',dldp2[0:nxi],'dldpRi=',dldp2[2*nxi:2*nxi+nui],'dldppi=',dldp2[-1],'weightmode:',weight_mode,'ADMM_adaptive:',adaptiveADMM)
        i_iter += 1 # comment this for comparing gradient computation time
        # below is the code for saving the trajectory optimization results using the last updated neural network (through lines 380 & 384)
        if delta_loss <=epi or i_iter>i_max:
            task_loss  = 0
            task_grad  = 0
            task_loss_nnl = 0
            task_loss_nni = 0
            xl_task    = []
            Kfbl_task  = []
            scxl_task  = []
            xc_task    = []
            uc_task    = []
            scxc_task  = []
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
                DI     = np.load('Planning_plots_meta_COM_Dyn/cable_direction_'+str(i_train_1)+'_'+str(task_idx)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'_'+str(adaptiveADMM1)+'.npy')
                TI     = np.load('Planning_plots_meta_COM_Dyn/tension_magnitude_'+str(i_train_1)+'_'+str(task_idx)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'_'+str(adaptiveADMM1)+'.npy')
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
                        ref_ti_k = np.reshape(TI[i,k],(1,1))
                        ref_dti_k= np.zeros((1,1))
                        ref_xi_k = np.reshape(np.vstack((ref_di_k,ref_wi,ref_ti_k,ref_dti_k)),nxi)
                        ref_xc[i][k*nxi:(k+1)*nxi] = ref_xi_k
                    ref_uq = np.zeros(int(nq)*nui)
                # the reference in the terminal stage
                ref_xl, ref_ul = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time,rg_task)
                for i in range(nq):
                    ref_di_N = np.reshape(DI[i][:,-1],(3,1))
                    ref_wi_N = np.zeros((3,1))
                    ref_ti_N = np.reshape(TI[i,-1],(1,1))
                    ref_dti_N= np.zeros((1,1))
                    ref_xi_N = np.reshape(np.vstack((ref_di_N,ref_wi_N,ref_ti_N,ref_dti_N)),nxi)
                    ref_xc[i][horizon*nxi:(horizon+1)*nxi]=ref_xi_N
                Ref_xl[horizon*nxl:(horizon+1)*nxl] = ref_xl 
                # the initial cable state (world frame)
                xq_init = np.zeros(nq*nxi) 
                for i in range(nq):
                    di_init   = np.reshape(DI[i][:,0],(3,1)) 
                    wi_init   = np.zeros((3,1))
                    ti_init   = np.array([[TI[i,0]]])
                    dti_init  = np.zeros((1,1))
                    xi_init   = np.reshape(np.vstack((di_init,wi_init,ti_init,dti_init)),nxi)
                    xq_init[i*nxi:(i+1)*nxi] = xi_init
            
                if weight_mode == 'n':
                    nn_input         = np.reshape(rg_task[0:2]/max_radius,(2,1)) # dimensionless
                    nn_l_output_task = convert_nn_l(nn_l(nn_input))
                    P_weight1  = Grad_Solver.Set_Parameters_nn_l(nn_l_output_task,pl_min)
                    nn_i_output_task = convert_nn_i(nn_i(nn_input))
                    P_weight2  = Grad_Solver.Set_Parameters_nn_i(nn_i_output_task,pi_min)
                    print('iter_train=',i_iter,'task_idx=',task_idx,'pl_min=',pl_min,'pi_min=',pi_min)
                    np.save('trained_data_multiagent_meta_COM_Dyn/pl_min',pl_min)
                    np.save('trained_data_multiagent_meta_COM_Dyn/pi_min',pi_min)
                else:
                    weight     = Grad_Solver.Set_Parameters(tunable_para,p_min)
                    P_weight1  = weight[0:npl]
                    P_weight2  = weight[npl:npauto]
                    print('iter_train=',i_iter,'task_idx=',task_idx,'p_min=',p_min)
                    np.save('trained_data_multiagent_meta_COM_Dyn/p_min',p_min)

                print('iter_train=',i_iter,'task_idx=',task_idx,'Ql=',P_weight1[0:nxl],'QlN=',P_weight1[nxl:2*nxl],'Rl=',P_weight1[2*nxl:2*nxl+nul],'p=',P_weight1[-1])
                print('iter_train=',i_iter,'task_idx=',task_idx,'Qi=',P_weight2[0:nxi],'QiN=',P_weight2[nxi:2*nxi],'Ri=',P_weight2[2*nxi:2*nxi+nui],'pi=',P_weight2[-1])
                start_time = TM.time()
                opt_sol, Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3 = MPC_load.ADMM_forward_MPC(Ref_xl,Ref_ul,ref_xc,ref_uq,xl_init,xq_init,P_weight1,P_weight2,max_iter_ADMM,adaptiveADMM)
                mpctime    = (TM.time() - start_time)*1000
                print("forward mpc:--- %s ms ---" % format(mpctime,'.2f'))
                start_time = TM.time()
                Grad_Out1l, Grad_Out1c, Grad_Out2, Grad_Out3, GradTime, GradTimeCaos, GradTimeCao,  GradTimePDP,  GradTime_c, GradTimeCaos_c, GradTimeCao_c, GradTimePDP_c,  MeanerrorCao, MeanerrorPDP, MeanerrorCao_c, MeanerrorPDP_c   = MPC_load.ADMM_Gradient_Solver(Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3, Ref_xl, Ref_ul, ref_xc, ref_uq, P_weight1, P_weight2, adaptiveADMM)
                gradtime    = (TM.time() - start_time)*1000
                print("backward:--- %s ms ---" % format(gradtime,'.2f'))
                dldw, loss, loss_track, loss_resid  = Grad_Solver.ChainRule(Opt_Sol1_l,Opt_Sol1_c,Opt_Sol2,Ref_xl,ref_xc,Grad_Out1l,Grad_Out1c,Grad_Out2,wt,wrp)
                task_loss  += loss[0]
                xl_task    += [opt_sol['xl_traj']]
                Kfbl_task  += [opt_sol['Kfbl_traj']]
                scxl_task  += [opt_sol['scxl_traj']]
                xc_task    += [opt_sol['xc_traj']]
                uc_task    += [opt_sol['uc_traj']]
                scxc_task  += [opt_sol['scxc_traj']]
            avg_loss    = task_loss/num_task
            loss_train += [avg_loss]
            losst_train += [avg_losst]
            lossrp_train += [avg_lossrp]
            xl_train   += [xl_task]
            Kfbl_train += [Kfbl_task]
            scxl_train += [scxl_task]
            xc_train   += [xc_task]
            uc_train   += [uc_task]
            scxc_train += [scxc_task]
            iter_train += [i_iter]
            print('iter_train=',i_iter,'loss=',avg_loss)

        # save the trained network models
        if weight_mode == 'n':
            PATH2   = "trained_data_multiagent_meta_COM_Dyn/trained_nn_l_"+str(max_iter_ADMM)+"_"+str(weight_mode)+"_"+str(adaptiveADMM)+".pt"
            torch.save(nn_l,PATH2)
            PATH3   = "trained_data_multiagent_meta_COM_Dyn/trained_nn_i_"+str(max_iter_ADMM)+"_"+str(weight_mode)+"_"+str(adaptiveADMM)+".pt"
            torch.save(nn_i,PATH3)
        

    traintime    = (TM.time() - start_time1)
    print("train:--- %s s ---" % format(traintime,'.2f'))
    np.save('trained_data_multiagent_meta_COM_Dyn/tunable_para_trained_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),tunable_para)
    np.save('trained_data_multiagent_meta_COM_Dyn/loss_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),loss_train)
    np.save('trained_data_multiagent_meta_COM_Dyn/losst_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),losst_train)
    np.save('trained_data_multiagent_meta_COM_Dyn/lossrp_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),lossrp_train)
    np.save('trained_data_multiagent_meta_COM_Dyn/Wt_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),Wt)
    np.save('trained_data_multiagent_meta_COM_Dyn/xl_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),xl_train)
    np.save('trained_data_multiagent_meta_COM_Dyn/Kfbl_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),Kfbl_train)
    np.save('trained_data_multiagent_meta_COM_Dyn/scxl_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),scxl_train)
    np.save('trained_data_multiagent_meta_COM_Dyn/xc_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),xc_train)
    np.save('trained_data_multiagent_meta_COM_Dyn/uc_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),uc_train)
    np.save('trained_data_multiagent_meta_COM_Dyn/scxc_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),scxc_train)
    np.save('trained_data_multiagent_meta_COM_Dyn/train_num_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),iter_train)
    np.save('trained_data_multiagent_meta_COM_Dyn/gradtimeOur_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),gradtimeOur)
    np.save('trained_data_multiagent_meta_COM_Dyn/gradtimeOur_c_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),gradtimeOur_c)
    np.save('trained_data_multiagent_meta_COM_Dyn/gradtimeCaos_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),gradtimeCaos)
    np.save('trained_data_multiagent_meta_COM_Dyn/gradtimeCaos_c_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),gradtimeCaos_c)
    np.save('trained_data_multiagent_meta_COM_Dyn/gradtimeCao_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),gradtimeCao)
    np.save('trained_data_multiagent_meta_COM_Dyn/gradtimeCao_c_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),gradtimeCao_c)
    np.save('trained_data_multiagent_meta_COM_Dyn/gradtimePDP_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),gradtimePDP)
    np.save('trained_data_multiagent_meta_COM_Dyn/gradtimePDP_c_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),gradtimePDP_c)
    np.save('trained_data_multiagent_meta_COM_Dyn/meanerrorCao_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),meanerrorCao)
    np.save('trained_data_multiagent_meta_COM_Dyn/meanerrorPDP_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),meanerrorPDP)
    np.save('trained_data_multiagent_meta_COM_Dyn/meanerrorCao_c_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),meanerrorCao_c)
    np.save('trained_data_multiagent_meta_COM_Dyn/meanerrorPDP_c_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM),meanerrorPDP_c)


    plt.figure(1,figsize=(6,4),dpi=400)
    plt.plot(loss_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('trained_data_multiagent_meta_COM_Dyn/loss_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM)+'.png',dpi=300)
    plt.show()

    plt.figure(2,figsize=(6,4),dpi=400)
    plt.plot(losst_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Loss_track')
    plt.grid()
    plt.savefig('trained_data_multiagent_meta_COM_Dyn/losst_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM)+'.png',dpi=300)
    plt.show()

    plt.figure(3,figsize=(6,4),dpi=400)
    plt.plot(lossrp_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Loss_residuals')
    plt.grid()
    plt.savefig('trained_data_multiagent_meta_COM_Dyn/lossrp_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM)+'.png',dpi=300)
    plt.show()

    plt.figure(4,figsize=(6,4),dpi=400)
    plt.plot(Wt, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Wt')
    plt.grid()
    plt.savefig('trained_data_multiagent_meta_COM_Dyn/Wt_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM)+'.png',dpi=300)
    plt.show()

    return avg_loss


def evaluate(i_train,task_idx,max_iter_ADMM,adaptiveADMM):
    if not os.path.exists("Planning_plots_multiagent_meta_COM_Dyn"):
        os.makedirs("Planning_plots_multiagent_meta_COM_Dyn")
    rp_task = np.load('trained_data_meta_COM_Dyn/rp_task.npy')
    rg_task = m2/mtot*rp_task[task_idx] # [m]
    pl_min  = np.load('trained_data_multiagent_meta_COM_Dyn/pl_min.npy')
    pi_min  = np.load('trained_data_multiagent_meta_COM_Dyn/pi_min.npy')
    weight_mode1 = 'n'
    adaptiveADMM1 = 'a'
    loss_train1  = np.load('trained_data_meta_COM_Dyn/loss_train_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'_'+str(adaptiveADMM1)+'.npy')
    i_train_1    = len(loss_train1)-1
    
    if weight_mode == 'n':
        PATH2            = "trained_data_multiagent_meta_COM_Dyn/trained_nn_l_"+str(max_iter_ADMM)+"_"+str(weight_mode)+"_"+str(adaptiveADMM)+".pt"
        PATH3            = "trained_data_multiagent_meta_COM_Dyn/trained_nn_i_"+str(max_iter_ADMM)+"_"+str(weight_mode)+"_"+str(adaptiveADMM)+".pt"
        nn_l             = torch.load(PATH2, weights_only=False)
        nn_i             = torch.load(PATH3, weights_only=False)
        nn_input         = np.reshape(rg_task[0:2]/max_radius,(2,1)) # dimensionless
        nn_l_output_task = convert_nn_l(nn_l(nn_input))
        P_weight1        = Grad_Solver.Set_Parameters_nn_l(nn_l_output_task,pl_min)
        nn_i_output_task = convert_nn_i(nn_i(nn_input))
        P_weight2        = Grad_Solver.Set_Parameters_nn_i(nn_i_output_task,pi_min)
    else:
        tunable_para_trained = np.load('trained_data_multiagent_meta_COM_Dyn/tunable_para_trained_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM)+'.npy')
        weight           = Grad_Solver.Set_Parameters(tunable_para_trained,pl_min)
        P_weight1        = weight[0:npl]
        P_weight2        = weight[npl:npauto]
    print('task_idx=',task_idx,'rg_task[m]=',rg_task,'Ql=',P_weight1[0:nxl],'QlN=',P_weight1[nxl:2*nxl],'Rl=',P_weight1[2*nxl:2*nxl+nul],'p=',P_weight1[-1])
    print('task_idx=',task_idx,'Qi=',P_weight2[0:nxi],'QiN=',P_weight2[nxi:2*nxi],'Ri=',P_weight2[2*nxi:2*nxi+nui],'pi=',P_weight2[-1])
    xl_train    = np.load('trained_data_multiagent_meta_COM_Dyn/xl_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM)+'.npy')
    scxl_train  = np.load('trained_data_multiagent_meta_COM_Dyn/scxl_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM)+'.npy')
    xc_train    = np.load('trained_data_multiagent_meta_COM_Dyn/xc_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM)+'.npy')
    scxc_train  = np.load('trained_data_multiagent_meta_COM_Dyn/scxc_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM)+'.npy')
    xl_traj     = xl_train[i_train]
    scxl_traj   = scxl_train[i_train]
    xq_traj     = xc_train[i_train]
    scxq_traj   = scxc_train[i_train]
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
    alpha      = 2*np.pi/nq
    Tq         = np.zeros((nq,horizon))
    scTq       = np.zeros((nq,horizon))

    DI     = np.load('Planning_plots_meta_COM_Dyn/cable_direction_'+str(i_train_1)+'_'+str(task_idx)+'_'+str(max_iter_ADMM_1)+'_'+str(weight_mode1)+'_'+str(adaptiveADMM1)+'.npy')
 
    for i in range(nq):
        Pi     = np.zeros((3,horizon))
        scPi   = np.zeros((3,horizon))
        refPi  = np.zeros((3,horizon))
        ri     = np.array([[rl*math.cos(i*alpha),rl*math.sin(i*alpha),0]]).T- np.reshape(np.vstack((rg_task[0],rg_task[1],0)),(3,1))
        ai     = np.zeros((3,horizon))
        for k in range(horizon):
            pl_k   = np.reshape(xl_traj[task_idx][k,0:3],(3,1))
            ql_k   = np.reshape(xl_traj[task_idx][k,6:10],(4,1))
            Rl_k   = sysm.q_2_rotation(ql_k)
            di_k   = np.reshape(xq_traj[task_idx][i][k,0:3],(3,1))
            ti_k   = xq_traj[task_idx][i][k,6]
            scdi_k = np.reshape(scxq_traj[task_idx][i][k,0:3],(3,1))
            scti_k = scxq_traj[task_idx][i][k,6]
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
    fig1.savefig('Planning_plots_multiagent_meta_COM_Dyn/quadrotor1_traj_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
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
    fig2.savefig('Planning_plots_multiagent_meta_COM_Dyn/quadrotor2_traj_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
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
    fig3.savefig('Planning_plots_multiagent_meta_COM_Dyn/quadrotor3_traj_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
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
    fig4.savefig('Planning_plots_multiagent_meta_COM_Dyn/quadrotor4_traj_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    
    fig5, ax5 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax5.add_patch(obs1)
    ax5.add_patch(obs2)
    ax5.plot(Ref_xl[0,:],Ref_xl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax5.plot(Pl[0,:],Pl[1,:],label='Planned',linewidth=1)
    ax5.plot(scPl[0,:],scPl[1,:],label='Planned_safe_copy',color='black',marker='.',markersize=1,linewidth=1)
    kt= 50
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
    fig5.savefig('Planning_plots_multiagent_meta_COM_Dyn/system_traj_quadrotor_num6_'+str(i_train)+'_'+str(task_idx)+'_'+str(kt)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()


    plt.figure(6,figsize=(6,4),dpi=300)
    plt.plot(Time,Tq[0,:],linewidth=1,label='1st cable')
    plt.plot(Time,Tq[1,:],linewidth=1,label='2nd cable')
    plt.plot(Time,Tq[2,:],linewidth=1,label='3rd cable')
    plt.plot(Time,Tq[3,:],linewidth=1,label='4th cable')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('MPC tension force [N]')
    plt.grid()
    plt.savefig('Planning_plots_multiagent_meta_COM_Dyn/cable_MPC_tensions_6_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(7,figsize=(6,4),dpi=300)
    plt.plot(Time,Eulerl[0,:],linewidth=1,label='roll')
    plt.plot(Time,Eulerl[1,:],linewidth=1,label='pitch')
    plt.plot(Time,Eulerl[2,:],linewidth=1,label='yaw')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Euler angle [deg]')
    plt.grid()
    plt.savefig('Planning_plots_multiagent_meta_COM_Dyn/euler_6_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()

    plt.figure(8,figsize=(6,4),dpi=300)
    plt.plot(Time,Pl[2,:],linewidth=1,label='actual')
    plt.plot(Time,Ref_xl[2,:],linewidth=0.5,linestyle='--',label='desired')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.grid()
    plt.savefig('Planning_plots_multiagent_meta_COM_Dyn/height_'+str(i_train)+'_'+str(task_idx)+'_'+str(weight_mode)+'.png',dpi=400)
    plt.show()


"""---------------------------------Main function-----------------------------"""



print("Please choose ADMM penalty mode")
adaptiveADMM = input("enter 'a' or 'f' without the quotation mark, a: iteration-adaptive; f: iteration-fixed")

if mode =="t":
    avg_loss = train(m0,v0,mt0,vt0,mrp0,vrp0,lr0,lr_nn,tunable_para0,wt0,wrp0,max_iter_ADMM,adaptiveADMM)
else:
    loss_train = np.load('trained_data_multiagent_meta_COM_Dyn/loss_train_'+str(max_iter_ADMM)+'_'+str(weight_mode)+'_'+str(adaptiveADMM)+'.npy')
    task_index = input("enter 0, 1, ..., 9")
    evaluate(len(loss_train)-1,int(task_index),max_iter_ADMM,adaptiveADMM)
    # evaluate(0,int(task_index),max_iter_ADMM,adaptiveADMM)
    # evaluate(1)