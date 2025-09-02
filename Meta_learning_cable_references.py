"""
Main function of the load planner (Tension Allocation)
------------------------------------------------------
1st version, Dr. Wang Bingheng, 19-Dec-2024
"""

from casadi import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import Dynamics
import Optimal_Allocation_DDP_quaternion_autotuning_ADMM
import math
import time as TM
from scipy.spatial.transform import Rotation as Rot
import os
import Neural_network
import torch

print("=============================================")
print("Main code for training or evaluating Automultilift")
print("Please choose mode")
mode = input("enter 't' or 'e' without the quotation mark:")
if mode =='t':
    print("Please choose weight_mode")
    weight_mode = input("enter 'n' or 'f' without the quotation mark:")
print("=============================================")




"""--------------------------------------Load Environment---------------------------------------"""
sysm_para = np.array([3, 0.25, 0.25,0.25,0.25, 0.02,0.02,0, 6, 1.25, 0.125, 0.5])
dt        = 0.04 # step size 0.1s
rl        = sysm_para[1]
rq        = sysm_para[10]
ro        = sysm_para[11]
nq        = int(sysm_para[8])
cl0       = sysm_para[9] # cable length
sysm      = Dynamics.multilift_model(sysm_para,dt)
sysm.model()
nxl       = sysm.nxl # dimension of the load's state
nul       = 3*nq # total dimension of the load's control = 6 (wrench) + 3*6-6 (null-space vector)
nWl       = sysm.nWl


"""--------------------------------------Define Planner---------------------------------------"""
horizon   = 100
e_abs, e_rel = 1e-4, 1e-3
MPC_load  = Optimal_Allocation_DDP_quaternion_autotuning_ADMM.MPC_Planner(sysm_para,dt,horizon,e_abs,e_rel)
pob1, pob2 = np.array([[1.5,1.3]]).T, np.array([[0.4,2.9]]).T # planar positions of the two obstacle in the world frame
print('obstacle_distance=',LA.norm(pob1-pob2))
MPC_load.SetStateVariable(sysm.xl)
MPC_load.SetCtrlVariable(sysm.Wl)
# MPC_load.SetDyn(sysm.model_l)
MPC_load.SetLearnablePara()
# MPC_load.SetConstraints_ADMM_Subp2(pob1,pob2)
# MPC_load.SetCostDyn_ADMM()
# MPC_load.ADMM_SubP2_Init()
# MPC_load.system_derivatives_DDP_ADMM()
# MPC_load.system_derivatives_SubP2_ADMM()
# MPC_load.system_derivatives_SubP3_ADMM()

# define the network size
D_in, D_h1, D_h2, D_out = 2, 16, 32, MPC_load.n_Pauto 
def convert_nn(nn_i_outcolumn):
    # convert a column tensor to a row np.array
    nn_i_row = np.zeros((1,D_out))
    for i in range(D_out):
        nn_i_row[0,i] = nn_i_outcolumn[i,0]
    return nn_i_row

# generate a list that saves random load center-of-mass coordinates for tasks
rg_task   = []
num_task  = 10
for _ in range(num_task):
    # random_rg = np.random.uniform([-0.05,-0.05],[0.05,0.05])
    rg        = np.random.uniform(0,0.04)
    alpha     = np.random.uniform(0,2*np.pi)
    random_rg = np.array([rg*np.cos(alpha),rg*np.sin(alpha)])
    rg_task  += [random_rg]
print('rg_task=',rg_task)
np.save('trained_data_meta/rg_task',rg_task)
# parameters of RMSProp
lr0       = 1# 0.1 for better ADMM initalization
lr_nn     = 1
epsilon   = 1e-8
v0        = np.zeros(MPC_load.n_Pauto)

# parameters of ADAM
m0        = np.zeros(MPC_load.n_Pauto)
beta1     = 0.9 # 0.8 for better ADMM initialization
beta2     = 0.999 # 0.5 for better ADMM initialization

"""--------------------------------------Define Load Reference---------------------------------------"""
Coeffx        = np.zeros((2,8))
Coeffy        = np.zeros((2,8))
Coeffz        = np.zeros((2,8))
for k in range(2):
    Coeffx[k,:] = np.load('Reference_traj_4/coeffx'+str(k+1)+'.npy')
    Coeffy[k,:] = np.load('Reference_traj_4/coeffy'+str(k+1)+'.npy')
    Coeffz[k,:] = np.load('Reference_traj_4/coeffz'+str(k+1)+'.npy')
Ref_xl = np.zeros(nxl*(horizon+1))
Ref_ul = np.zeros(nul*horizon)
Ref_pl = np.zeros((3,horizon+1))
Ref_Wl = np.zeros(nWl*horizon)
Ref_ac = np.zeros((2,horizon+1))
Time   = []
time   = 0
for k in range(horizon):
    Time  += [time]
    ref_xl, ref_ul, ref_p, ref_Wl, ac = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time)
    Ref_xl[k*nxl:(k+1)*nxl] = ref_xl
    Ref_ul[k*nul:(k+1)*nul] = ref_ul
    Ref_Wl[k*nWl:(k+1)*nWl] = ref_Wl
    Ref_pl[:,k:k+1] = ref_p
    Ref_ac[:,k:k+1] = ac
    time += dt
# Time  += [time]
ref_xl, ref_ul, ref_p, ref_Wl, ac = sysm.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time)
Ref_xl[horizon*nxl:(horizon+1)*nxl] = ref_xl
Ref_pl[:,horizon:horizon+1] = ref_p
Ref_ac[:,horizon:horizon+1] = ac
print('max_ac_1=',Ref_ac[:,int(50)])
# print('max_ac_2=',Ref_ac[:,int(63)])
# print('max_ac=',np.max(abs(Ref_ac[1,:])))
# initial palyload's state
x0         = np.random.normal(0,0.01)
y0         = np.random.normal(0,0.01)
z0         = np.random.normal(0.5,0.01)
pl         = np.array([[x0,y0,z0]]).T
vl         = np.reshape(np.random.normal(0,0.01,3),(3,1)) # initial velocity of CO in {I}
Eulerl     = np.clip(np.reshape(np.random.normal(0,0.01,3),(3,1)),-2/57.3,2/57.3)
Rl0        = sysm.dir_cosine(Eulerl)
r          = Rot.from_matrix(Rl0)  
# quaternion in the format of x, y, z, w 
# (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_quat.html)
ql0        = r.as_quat() 
ql         = np.array([[ql0[3], ql0[0], ql0[1], ql0[2]]]).T
wl         = np.reshape(np.random.normal(0,0.01,3),(3,1))
xl_init    = np.reshape(np.vstack((pl,vl,ql,wl)),nxl)

# MPC weights (learnable parameters, now manually tuned)
tunable_para0 = np.random.normal(0,0.1,MPC_load.n_Pauto) # initialization

# Solve the load's MPC planner
def train(m0,v0,lr0,Ref_xl,Ref_Wl,tunable_para0):
    if not os.path.exists("trained_data_meta"):
        os.makedirs("trained_data_meta")
    tunable_para = tunable_para0
    i = 1
    i_max      = 25
    delta_loss = 1e2
    loss0      = 1e2
    epi        = 1e-1
    xl_train   = []
    Wl_train   = []
    scxl_train = []
    Tl_train   = []
    loss_train = []
    iter_train = []
    gradtimeRe1 = []
    gradtimeRe2 = []
    gradtimeNO1 = []
    gradtimeNO2 = []
    gradtimeCao1 = []
    gradtimeCao2 = []
    gradtimeCao1_s = []
    gradtimeCao2_s = []
    meanerror1  = []
    meanerror2  = []
    Auxtime1_1  = []
    Auxtime1_2  = []
    Auxtime2NO_1= []
    Auxtime2NO_2= []  
    start_time1 = TM.time()
    v          = v0
    m          = m0
    PATHl_init = "trained_data_meta/initial_nn_waypoint.pt"
    nn_waypoint = Neural_network.Net(D_in,D_h1,D_h2,D_out)
    torch.save(nn_waypoint,PATHl_init)
    optimizer   = torch.optim.Adam(nn_waypoint.parameters(),lr=lr_nn)
    while delta_loss>epi and i<=i_max:
        task_loss  = 0
        task_grad  = 0
        task_loss_nn = 0
        xl_task    = []
        Wl_task    = []
        scxl_task  = []
        Tl_task    = []
        for task_idx in range(num_task):
            sysm_para_task = sysm_para.copy()
            sysm_para_task[5:7] = rg_task[task_idx]
            sysm_task      = Dynamics.multilift_model(sysm_para_task,dt)
            sysm_task.model()
            """--------------------------------------Redefine Planner---------------------------------------"""
            MPC_load_task  = Optimal_Allocation_DDP_quaternion_autotuning_ADMM.MPC_Planner(sysm_para_task,dt,horizon,e_abs,e_rel)
            MPC_load_task.SetStateVariable(sysm_task.xl)
            MPC_load_task.SetCtrlVariable(sysm_task.Wl)
            MPC_load_task.SetDyn(sysm_task.model_l)
            MPC_load_task.SetLearnablePara()
            MPC_load_task.SetConstraints_ADMM_Subp2(pob1,pob2)
            MPC_load_task.SetCostDyn_ADMM()
            MPC_load_task.ADMM_SubP2_Init()
            MPC_load_task.system_derivatives_DDP_ADMM()
            MPC_load_task.system_derivatives_SubP2_ADMM()
            MPC_load_task.system_derivatives_SubP3_ADMM()
            """--------------------------------------Redefine Gradient Solver---------------------------------------"""
            Grad_Solver_task = Optimal_Allocation_DDP_quaternion_autotuning_ADMM.Gradient_Solver(sysm_para_task, horizon,sysm_task.xl,sysm_task.Wl,MPC_load_task.sc_xl,MPC_load_task.sc_Wl,MPC_load_task.nv,MPC_load_task.P_auto,
                                                                       MPC_load_task.P_pinv,MPC_load_task.P_ns,e_abs,e_rel)
            

            Ref_xl = np.zeros(nxl*(horizon+1))
            Ref_ul = np.zeros(nul*horizon)
            Ref_pl = np.zeros((3,horizon+1))
            Ref_Wl = np.zeros(nWl*horizon)
            Ref_ac = np.zeros((2,horizon+1))
            Time   = []
            time   = 0
            for k in range(horizon):
                Time  += [time]
                ref_xl, ref_ul, ref_p, ref_Wl, ac = sysm_task.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time)
                Ref_xl[k*nxl:(k+1)*nxl] = ref_xl
                Ref_ul[k*nul:(k+1)*nul] = ref_ul
                Ref_Wl[k*nWl:(k+1)*nWl] = ref_Wl
                Ref_pl[:,k:k+1] = ref_p
                Ref_ac[:,k:k+1] = ac
                time += dt
            # Time  += [time]
            ref_xl, ref_ul, ref_p, ref_Wl, ac = sysm_task.minisnap_load_circle(Coeffx,Coeffy,Coeffz,time)
            Ref_xl[horizon*nxl:(horizon+1)*nxl] = ref_xl
            Ref_pl[:,horizon:horizon+1] = ref_p
            Ref_ac[:,horizon:horizon+1] = ac
            # generate the corresponding hyperparameters, given the task rg
            if weight_mode == 'n':
                nn_input   = np.reshape(1e2*rg_task[task_idx],(2,1)) # unit cm
                nn_output_task = convert_nn(nn_waypoint(nn_input))
                weight     = Grad_Solver_task.Set_Parameters_nn(nn_output_task)
            else:
                weight     = Grad_Solver_task.Set_Parameters(tunable_para)
            p_weight1  = weight[0:MPC_load_task.n_P1]
            p_weight2  = weight[MPC_load_task.n_P1:MPC_load_task.n_P1 + MPC_load_task.n_P2]
            p1         = weight[-1]
            
        
            start_time = TM.time()
            Opt_Sol1, Opt_Sol2, Opt_Y, Opt_Eta  = MPC_load_task.ADMM_forward_MPC_DDP(xl_init,Ref_xl,Ref_Wl,p_weight1,p_weight2,p1)
            mpctime    = (TM.time() - start_time)*1000
            print("a:--- %s ms ---" % format(mpctime,'.2f'))
            # start_time = TM.time()
            Grad_Out1, Grad_Out2, Grad_Out3, GradTime, GradTimeNO, GradTimeCao, GradTimeCao_s, Meanerror, AuxTime1, AuxTime2NO = MPC_load_task.ADMM_Gradient_Solver(Opt_Sol1,Opt_Sol2,Opt_Y,Opt_Eta,Ref_xl,Ref_Wl,p_weight1,p_weight2,p1)
            # gradtime    = (TM.time() - start_time)*1000
            # print("g:--- %s ms ---" % format(gradtime,'.2f'))
            # gradtimeRe1 += [GradTime[0]]
            # gradtimeRe2 += [GradTime[1]]
            # gradtimeNO1 += [GradTimeNO[0]]
            # gradtimeNO2 += [GradTimeNO[1]]
            # gradtimeCao1 += [GradTimeCao[0]]
            # gradtimeCao2 += [GradTimeCao[1]]
            # gradtimeCao1_s += [GradTimeCao_s[0]]
            # gradtimeCao2_s += [GradTimeCao_s[1]]
            # meanerror1  += [Meanerror[0]]
            # meanerror2  += [Meanerror[1]]
            # Auxtime1_1  += [AuxTime1[0]]
            # Auxtime1_2  += [AuxTime1[1]]
            # Auxtime2NO_1+= [AuxTime2NO[0]]
            # Auxtime2NO_2+= [AuxTime2NO[1]]
            dldw, loss  = Grad_Solver_task.ChainRule(Opt_Sol1,Opt_Sol2,Ref_xl,Grad_Out1,Grad_Out2,p1)
            task_loss  += loss[0]
            if weight_mode == 'n':
                dwdp        = Grad_Solver_task.ChainRule_Gradient_nn(nn_output_task)
                dldp        = np.reshape(dldw@dwdp,(1,MPC_load_task.n_Pauto))
                loss_nn     = nn_waypoint.myloss(nn_waypoint(nn_input),dldp)
                task_loss_nn += loss_nn
                task_grad  += np.reshape(dldp,MPC_load_task.n_Pauto)
            else:
                dwdp        = Grad_Solver_task.ChainRule_Gradient(tunable_para)
                dldp        = np.reshape(dldw@dwdp,MPC_load_task.n_Pauto)
                task_grad  += dldp
            xl_task    += [Opt_Sol1[1]['xl_opt']]
            Wl_task    += [Opt_Sol1[1]['Wl_opt']]
            scxl_task  += [Opt_Sol2[1]['scxl_opt']]
            Tl_task    += [Opt_Sol2[1]['Tl_opt']]
            print('iter_train=',i,'task_idx=',task_idx,'Q=',p_weight1[0:12],'QN=',p_weight1[12:24],'R=',p_weight1[24:30])
            print('iter_train=',i,'task_idx=',task_idx,'nv_w=',p_weight2[0:MPC_load_task.n_P2],'p1=',p1,'loss_task=',loss)
            
        
        if weight_mode == 'n':
            optimizer.zero_grad()
            avg_loss_nn = task_loss_nn/num_task
            avg_loss_nn.backward()
            optimizer.step()
            avg_grad    = task_grad/num_task
        else:
            avg_grad    = task_grad/num_task
            for k in range(int(MPC_load_task.n_Pauto)):
                m[k]    = beta1*m[k] + (1-beta1)*avg_grad[k]
                m_hat   = m[k]/(1-beta1**i)
                v[k]    = beta2*v[k] + (1-beta2)*avg_grad[k]**2
                v_hat   = v[k]/(1-beta2**i)
                lr      = lr0/(np.sqrt(v_hat+epsilon))
                tunable_para[k] = tunable_para[k] - lr*m_hat

        avg_loss    = task_loss/num_task
        loss_train += [avg_loss]
        xl_train  += [xl_task]
        Wl_train  += [Wl_task]
        scxl_train+= [scxl_task]
        Tl_train  += [Tl_task]
        iter_train += [i]
        if i==1:
            epi = 2e-4*avg_loss
        if i>2:
            delta_loss = abs(avg_loss-loss0)
        loss0      = avg_loss
        print('iter_train=',i,'loss=',avg_loss,'dldpQ=',avg_grad[0:12],'dldpR=',avg_grad[24:30],'dldpNv=',avg_grad[30:33],'dldpp1=',avg_grad[-1])
        i += 1
    traintime    = (TM.time() - start_time1)
    print("train:--- %s s ---" % format(traintime,'.2f'))
    np.save('trained_data_meta/tunable_para_trained',tunable_para)
    np.save('trained_data_meta/loss_train',loss_train)
    np.save('trained_data_meta/xl_train',xl_train)
    np.save('trained_data_meta/scxl_train',scxl_train)
    np.save('trained_data_meta/Wl_train',Wl_train)
    np.save('trained_data_meta/Tl_train',Tl_train)
    # np.save('trained_data/training_time',traintime)
    # np.save('trained_data/gradtimeRe1',gradtimeRe1)
    # np.save('trained_data/gradtimeRe2',gradtimeRe2)
    # np.save('trained_data/gradtimeNO1',gradtimeNO1)
    # np.save('trained_data/gradtimeNO2',gradtimeNO2)
    # np.save('trained_data/gradtimeCao1',gradtimeCao1)
    # np.save('trained_data/gradtimeCao2',gradtimeCao2)
    # np.save('trained_data/gradtimeCao1_s',gradtimeCao1_s)
    # np.save('trained_data/gradtimeCao2_s',gradtimeCao2_s)
    # np.save('trained_data/meanerror1',meanerror1)
    # np.save('trained_data/meanerror2',meanerror2)
    # np.save('trained_data/Auxtime1_1',Auxtime1_1)
    # np.save('trained_data/Auxtime1_2',Auxtime1_2)
    # np.save('trained_data/Auxtime2NO_1',Auxtime2NO_1)
    # np.save('trained_data/Auxtime2NO_2',Auxtime2NO_2)

    # save the trained network models
    PATH2   = "trained_data_meta/trained_nn_waypoint.pt"
    torch.save(nn_waypoint,PATH2)
    plt.figure(1,figsize=(6,4),dpi=400)
    plt.plot(loss_train, linewidth=1.5)
    plt.xlabel('Training episodes')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('trained_data_meta/loss_train.png',dpi=300)
    plt.show()



def evaluate(i_train,task_idx):
    if not os.path.exists("Planning_plots_meta"):
        os.makedirs("Planning_plots_meta")
    print('rg_task=',rg_task[task_idx])
    xl_train = np.load('trained_data_meta/xl_train.npy')
    scxl_train = np.load('trained_data_meta/scxl_train.npy')
    Wl_train = np.load('trained_data_meta/Wl_train.npy')
    Tl_train = np.load('trained_data_meta/Tl_train.npy')
    xl_opt   = xl_train[i_train]
    scxl_opt = scxl_train[i_train]
    Wl_opt   = Wl_train[i_train]
    Tl_opt   = Tl_train[i_train]
    # System open-loop predicted trajectories
    P_pinv     = MPC_load.P_pinv # pseudo-inverse of P matrix
    P_ns       = MPC_load.P_ns # null-space of P matrix
    Pl         = np.zeros((3,horizon))
    scPl       = np.zeros((3,horizon))
    Euler_l    = np.zeros((3,horizon))
    norm_2_Ql  = np.zeros(horizon)
    for k in range(horizon):
        Pl[:,k:k+1] = np.reshape(xl_opt[task_idx][k,0:3],(3,1))
        scPl[:,k:k+1] = np.reshape(scxl_opt[task_idx][k,0:3],(3,1))
        ql_k  = np.reshape(xl_opt[task_idx][k,6:10],(4,1))
        norm_2_Ql[k] = LA.norm(ql_k)
        Rl_k  = sysm.q_2_rotation(ql_k)
        rk    = Rot.from_matrix(Rl_k)
        euler_k = np.reshape(rk.as_euler('xyz',degrees=True),(3,1))
        Euler_l[:,k:k+1] = euler_k 
    Xq         = [] # list that stores all quadrotors' predicted trajectories
    DI         = [] # list that stores all cables' direction trajectories
    Aq         = [] # list that stores all cable attachments' trajectories in the world frame
    alpha      = 2*np.pi/nq
    Tq         = np.zeros((nq,horizon))
    for i in range(nq):
        Pi     = np.zeros((3,horizon))
        di     = np.zeros((3,horizon))
        ri     = np.array([[rl*math.cos(i*alpha),rl*math.sin(i*alpha),0]]).T
        ai     = np.zeros((3,horizon))
        for k in range(horizon):
            wl_k  = np.reshape(Wl_opt[task_idx][k,:],(6,1)) # 6-D wrench at the kth step
            nv_k  = np.reshape(Tl_opt[task_idx][k,:],(3*nq-6,1)) # 3-D null-space vector at the kth step
            t_k   = P_pinv@wl_k + P_ns@nv_k # 9-D tension vector at the kth step in the load's body frame
            ti_k  = np.reshape(t_k[3*i:3*(i+1)],(3,1))
            pl_k  = np.reshape(xl_opt[task_idx][k,0:3],(3,1))
            ql_k  = np.reshape(xl_opt[task_idx][k,6:10],(4,1))
            Rl_k  = sysm.q_2_rotation(ql_k)
            pi_k  = pl_k + Rl_k@(ri + cl0*ti_k/LA.norm(ti_k))
            di_k  = Rl_k@ti_k/LA.norm(ti_k)
            ai_k  = pl_k + Rl_k@ri
            Pi[:,k:k+1] = pi_k
            di[:,k:k+1] = di_k
            ai[:,k:k+1] = ai_k
            Tq[i,k] = LA.norm(ti_k)
        Xq += [Pi]
        DI += [di]
        Aq += [ai]

    # Save data
    np.save('Planning_plots_meta/tension_magnitude_'+str(task_idx),Tq)
    np.save('Planning_plots_meta/cable_direction_'+str(task_idx),DI)
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
    fig1.savefig('Planning_plots_meta/quadrotor1_traj_ddp_admm'+str(i_train)+'_'+str(task_idx)+'.png',dpi=400)
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
    fig2.savefig('Planning_plots_meta/quadrotor2_traj_ddp_admm'+str(i_train)+'_'+str(task_idx)+'.png',dpi=400)
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
    fig3.savefig('Planning_plots_meta/quadrotor3_traj_ddp_admm'+str(i_train)+'_'+str(task_idx)+'.png',dpi=400)
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
    fig4.savefig('Planning_plots_meta/quadrotor4_traj_admm'+str(i_train)+'_'+str(task_idx)+'.png',dpi=400)
    plt.show()

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
    fig5.savefig('Planning_plots_meta/quadrotor5_traj_admm'+str(i_train)+'_'+str(task_idx)+'.png',dpi=400)
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
    fig6.savefig('Planning_plots_meta/quadrotor6_traj_admm'+str(i_train)+'_'+str(task_idx)+'.png',dpi=400)
    plt.show()

    fig7, ax7 = plt.subplots(figsize=(5,5),dpi=300)
    obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
    obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
    ax7.add_patch(obs1)
    ax7.add_patch(obs2)
    ax7.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
    ax7.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
    ax7.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
    for k in range(horizon):
        if k==2 or k==50 or k==98:
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
    fig7.savefig('Planning_plots_meta/system_traj_quadrotor_num6_ddp_admm'+str(i_train)+'_'+str(task_idx)+'.png',dpi=400)
    plt.show()


    plt.figure(8,figsize=(6,4),dpi=300)
    plt.plot(Time,Tq[0,:],linewidth=1,label='1st cable')
    plt.plot(Time,Tq[1,:],linewidth=1,label='2nd cable')
    plt.plot(Time,Tq[2,:],linewidth=1,label='3rd cable')
    plt.plot(Time,Tq[3,:],linewidth=1,label='4th cable')
    plt.plot(Time,Tq[4,:],linewidth=1,label='5th cable')
    plt.plot(Time,Tq[5,:],linewidth=1,label='6th cable')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('MPC tension force [N]')
    plt.grid()
    plt.savefig('Planning_plots_meta/cable_MPC_tensions_6_ddp_admm'+str(i_train)+'_'+str(task_idx)+'.png',dpi=400)
    plt.show()


    plt.figure(9,figsize=(6,4),dpi=300)
    plt.plot(Time,Euler_l[0,:],linewidth=1,label='roll')
    plt.plot(Time,Euler_l[1,:],linewidth=1,label='pitch')
    plt.plot(Time,Euler_l[2,:],linewidth=1,label='yaw')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Euler angle [deg]')
    plt.grid()
    plt.savefig('Planning_plots_meta/euler_6_ddp_admm'+str(i_train)+'_'+str(task_idx)+'.png',dpi=400)
    plt.show()

"""---------------------------------Main function-----------------------------"""
if mode =="t":
    train(m0,v0,lr0,Ref_xl,Ref_Wl,tunable_para0)
else:
    loss_train = np.load('trained_data_meta/loss_train.npy')
    evaluate(len(loss_train)-1,9)
    # evaluate(0)
    # evaluate(4)
