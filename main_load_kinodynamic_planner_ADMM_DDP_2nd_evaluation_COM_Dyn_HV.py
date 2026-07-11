"""
Main function of the load planner (Tension Dynamic Allocation)
horizontal obstacles
------------------------------------------------------
1st version, Dr. Wang Bingheng, 07-Mar-2025
2nd version, Dr. Wang Bingheng, 17-June-2025
3rd version, Dr. Wang Bingheng, 02-Dec-2025
"""

from casadi import *
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import Dynamics_load_cable_autotuning_2nd_COM_Dyn
import Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn_HV
import math
import time as TM
from scipy.spatial.transform import Rotation as Rot
import os
import Neural_network
import torch
import random


if not os.path.exists("Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV"):
    os.makedirs("Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV")
print("Please choose weifght_mode")
weight_mode      = input("enter 'n' or 'f' without the quotation mark, n: task-adaptive; f: task-fixed")
print("Please choose ADMM truncation number for stage 1 in testing")
max_iter_ADMM1_test  = int(input("enter '2', '3', '4', or '5' without the quotation mark"))
print("Please choose ADMM truncation number for stage 2 in training")
max_iter_ADMM_train    = int(input("enter '2', '3', '4', or '5' without the quotation mark"))
print("Please choose ADMM truncation number for stage 2 in testing")
max_iter_ADMM    = int(input("enter '2', '3', '4', or '5' without the quotation mark"))
print("Please choose initial model for stage 2")
initial_model    = int(input("enter '0', '1', '2', '3' or '4' without the quotation mark"))



m1        = float(np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/m1.npy'))  # the load's net weight [kg], a circular basket with uniform mass distribution
m2        = float(np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/m2.npy')) # the added mass [kg]
mq        = 0.32  # the quadrotor mass [kg] including the battery
fmax      = 0.75*9.8    # the quadrotor maximum thrust [N]
mtot      = m1+m2 # the total weight [kg]
nq        = int(np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/nq.npy'))    # the number of quadrotors
cl0       = float(np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/cl0.npy')) # the cable length [m]
rq        = float(np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/rq.npy')) # the radius of quadrotor [m]
rl        = float(np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/rl.npy'))  # the radius of the load [m]
ro        = float(np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/ro.npy'))  # the radius of obstacle [m]
rov       = float(np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/rov.npy'))  # the radius of horiontal obstacle [m]
"""--------------------------------------Load Environment---------------------------------------"""
sysm_para = np.array([m1, m2, 
                      1/4*m1*rl**2, 1/4*m1*rl**2, 1/2*m1*rl**2, 
                      rl, nq, rq, mq, fmax, 
                      cl0, ro, rov])
# for a load with a biased CoM, different quadrotors need different weights
dt        = float(np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/dt.npy'))
sysm      = Dynamics_load_cable_autotuning_2nd_COM_Dyn.multilift_model(sysm_para,dt)
# set the coordinate of the added mass in the load body frame
rp        = np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/rp_task.npy')
sysm.Rotational_Inertia(rp)
sysm.model()
nxl       = sysm.nxl # dimension of the load's state
nul       = sysm.nul # dimension of the load's control
nxi       = sysm.nxi # dimension of the cable's state
nui       = sysm.nui # dimension of the cable's control

"""--------------------------------------Define Planner---------------------------------------"""
horizon   = 120 #int(np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/horizon.npy'))
width     = np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/width.npy')
deepth    = np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/deepth.npy')
pob1, pob2 = np.array([[-(ro+width/2),-deepth/2]]).T, np.array([[(ro+width/2),deepth/2]]).T # vertical colum
height    = np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/height.npy')
height2   = np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/height2.npy')
pob3, pob4 = np.array([[0,height +height2+2*rq+rov]]).T, np.array([[0,height-rq-rov]]).T # [y,z] 4.18 is the limit, yielding the same height margin (0.25cm) as in Dr. Sun Sihao's paper.
MPC_load  = Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn_HV.MPC_Planner(sysm_para,dt,horizon)
MPC_load.Rotational_Inertia(rp)
rg_task   = m2/mtot*rp
print('rp=',rp,'rg_task=',rg_task)
MPC_load.allocation_martrix(rg_task)
MPC_load.SetStateVariables(sysm.xl,sysm.xi)
MPC_load.SetCtrlVariables(sysm.ul,sysm.ui)
MPC_load.SetDyns(sysm.model_l,sysm.model_i)
MPC_load.SetWeightPara()
MPC_load.SetPayloadCostDyn(max_iter_ADMM)
MPC_load.SetCableCostDyn(max_iter_ADMM)
MPC_load.SetConstriants(pob1,pob2,pob3,pob4)
MPC_load.SetADMMSubP2_SoftCost_k()
MPC_load.SetADMMSubP2_SoftCost_N()
MPC_load.ADMM_SubP2_Init()
MPC_load.ADMM_SubP2_N_Init()
MPC_load.Load_derivatives_DDP_ADMM()
MPC_load.Cable_derivatives_DDP_ADMM()

npl       = MPC_load.npl
npi       = MPC_load.npi
npauto    = MPC_load.n_Pauto

max_radius = np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/max_radius.npy')

D_inl, D_h1l, D_h2l, D_outl = 2, 16, 32, MPC_load.npl 
def convert_nn_l(nn_l_outcolumn):
    # convert a column tensor to a row np.array
    nn_l_row = np.zeros((1,D_outl))
    for i in range(D_outl):
        nn_l_row[0,i] = nn_l_outcolumn[i,0]
    return nn_l_row

D_ini, D_h1i, D_h2i, D_outi = 2, 16, 32, MPC_load.npi
def convert_nn_i(nn_i_outcolumn):
    # convert a column tensor to a row np.array
    nn_i_row = np.zeros((1,D_outi))
    for i in range(D_outi):
        nn_i_row[0,i] = nn_i_outcolumn[i,0]
    return nn_i_row

"""--------------------------------------Define Load Reference---------------------------------------"""
Coeffx        = np.zeros((4,8))
Coeffy        = np.zeros((4,8))
Coeffz        = np.zeros((4,8))
for k in range(4):
    Coeffx[k,:] = np.load('Reference_traj_6_S_shape_evaluation/coeffx'+str(k+1)+'.npy')
    Coeffy[k,:] = np.load('Reference_traj_6_S_shape_evaluation/coeffy'+str(k+1)+'.npy')
    Coeffz[k,:] = np.load('Reference_traj_6_S_shape_evaluation/coeffz'+str(k+1)+'.npy')

# initial palyload's state
xl_init    = np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/xl_init.npy')
k_const    = 10 # 
"""--------------------------------------Define Gradient Solver--------------------------------------"""
Grad_Solver   = Kinodynamic_Planning_ADMM_quaternion_DDP_autotuning_2nd_COM_Dyn_HV.Gradient_Solver(sysm_para, horizon, MPC_load.xl, MPC_load.ul, MPC_load.scxl, MPC_load.scul, MPC_load.xi, MPC_load.ui, MPC_load.scxi, MPC_load.scui, MPC_load.P_auto, MPC_load.para_l, MPC_load.para_i)

# MPC weights

if weight_mode == 'n':
    PATH2            = "trained_data_multiagent_meta_COM_Dyn/trained_nn_l_"+str(initial_model)+'_'+str(max_iter_ADMM_train)+"_"+str(weight_mode)+".pt"
    PATH3            = "trained_data_multiagent_meta_COM_Dyn/trained_nn_i_"+str(initial_model)+'_'+str(max_iter_ADMM_train)+"_"+str(weight_mode)+".pt"
    nn_l             = torch.load(PATH2, weights_only=False)
    nn_i             = torch.load(PATH3, weights_only=False)
    # radius           = np.sqrt(rg_task[0]**2+rg_task[1]**2)# m
    nn_input         = np.reshape(rg_task[0:2]/max_radius*k_const,(2,1)) #dimensionless
    nn_l_output_task = convert_nn_l(nn_l(nn_input))
    P_weight1        = Grad_Solver.Set_Parameters_nn_l(nn_l_output_task)
    nn_i_output_task = convert_nn_i(nn_i(nn_input))
    P_weight2        = Grad_Solver.Set_Parameters_nn_i(nn_i_output_task)
else:
    tunable_para_trained = np.load('trained_data_multiagent_meta_COM_Dyn/tunable_para_trained_'+str(initial_model)+'_'+str(max_iter_ADMM_train)+'_'+str(weight_mode)+'.npy')
    weight           = Grad_Solver.Set_Parameters(tunable_para_trained)
    P_weight1        = weight[0:npl]
    P_weight2        = weight[npl:npauto]
# P_weight2[-1]    = P_weight2[-1]*10

print('Ql=',P_weight1[0:nxl],'QlN=',P_weight1[nxl:2*nxl],'Rl=',P_weight1[2*nxl:2*nxl+nul],'px=',P_weight1[-4],'gammax=',P_weight1[-2],'pu=',P_weight1[-3],'gammau=',P_weight1[-1])
print('Qi=',P_weight2[0:nxi],'QiN=',P_weight2[nxi:2*nxi],'Ri=',P_weight2[2*nxi:2*nxi+nui],'pix=',P_weight2[-4],'gammaix=',P_weight2[-2],'piu=',P_weight2[-3],'gammaiu=',P_weight2[-1])


"""--------------------------------------Define System References---------------------------------------"""
Ref_xl = np.zeros(nxl*(horizon+1))
Ref_ul = np.zeros(nul*horizon)
Ref_pl = np.zeros((3,horizon))
Time   = []
time   = 0
DI     = np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/cable_direction_'+str(max_iter_ADMM1_test)+'_'+str(nq)+'.npy')
TI     = np.load('Planning_plots_meta_evaluation_COM_Dyn_HV/tension_magnitude_'+str(max_iter_ADMM1_test)+'_'+str(nq)+'.npy')
ref_xc = [np.zeros((horizon+1)*nxi) for _ in range(nq)]
# the reference in the running stage
for k in range(horizon):
    Time  += [time]
    ref_xl, ref_ul = sysm.minisnap_load_S_shape(Coeffx,Coeffy,Coeffz,time,rg_task)
    Ref_xl[k*nxl:(k+1)*nxl] = ref_xl
    Ref_ul[k*nul:(k+1)*nul] = ref_ul
    Ref_pl[:,k:(k+1)]       = np.reshape(ref_xl[0:3],(3,1))
    time += dt
    for i in range(nq):
        ref_di_k = np.reshape(DI[i][:,k],(3,1))
        ref_wi = np.zeros((3,1))
        ref_ai = np.zeros((3,1))
        ref_ji = np.zeros((3,1))
        ref_ti_k = np.reshape(TI[i,k],(1,1))
        ref_dti_k= np.zeros((1,1))
        ref_xi_k = np.reshape(np.vstack((ref_di_k,ref_wi,ref_ai,ref_ji, ref_ti_k,ref_dti_k)),nxi)
        ref_xc[i][k*nxi:(k+1)*nxi] = ref_xi_k
ref_uq = np.zeros(int(nq)*nui)
# the reference in the terminal stage
ref_xl, ref_ul = sysm.minisnap_load_S_shape(Coeffx,Coeffy,Coeffz,time,rg_task)
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

# solve ADMM-DDP in the forward pass
MPC_load.PrewarmSubP2(Ref_xl,Ref_ul,ref_xc,ref_uq,P_weight1,P_weight2)
MPC_load.PrewarmJaxSubP1(Ref_xl,Ref_ul,ref_xc,ref_uq,xl_init,xq_init,P_weight1,P_weight2,max_iter_ADMM)
opt_sol, Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3 = MPC_load.ADMM_forward_MPC(Ref_xl,Ref_ul,ref_xc,ref_uq,xl_init,xq_init,P_weight1,P_weight2,max_iter_ADMM)
subp1time  = MPC_load.last_subp1_time_ms
subp2time  = MPC_load.last_subp2_time_ms
subp3time  = MPC_load.last_subp3_time_ms
mpctime    = MPC_load.last_subproblem_time_ms
print("forward mpc:--- %s ms ---" % format(mpctime,'.2f'))
xl_traj    = opt_sol['xl_traj']   
ul_traj    = opt_sol['ul_traj'] 
scxl_traj  = opt_sol['scxl_traj']
scul_traj  = opt_sol['scul_traj']
xq_traj    = opt_sol['xc_traj']
uq_traj    = opt_sol['uc_traj']
scxq_traj  = opt_sol['scxc_traj']
scuq_traj  = opt_sol['scuc_traj']
K_fb_traj  = opt_sol['Kfbl_traj']
np.save('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/xl_traj_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(nq),xl_traj)
np.save('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/xq_traj_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(nq),xq_traj)
np.save('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/scxl_traj_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(nq),scxl_traj)
np.save('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/scxq_traj_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(nq),scxq_traj)
np.save('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/Kfb_traj_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(nq),K_fb_traj)
np.save('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/mpctime_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(nq)+'_'+str(horizon),mpctime)
np.save('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/subp1time_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(nq)+'_'+str(horizon),subp1time)
np.save('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/subp2time_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(nq)+'_'+str(horizon),subp2time)
np.save('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/subp3time_'+str(initial_model)+'_'+str(max_iter_ADMM)+'_'+str(nq)+'_'+str(horizon),subp3time)
# System open-loop predicted trajectories
Pl         = np.zeros((3,horizon))
scPl       = np.zeros((3,horizon))
Eulerl     = np.zeros((3,horizon))
scEulerl    = np.zeros((3,horizon))
norm_2_Ql  = np.zeros(horizon)
for k in range(horizon):
    Pl[:,k:k+1] = np.reshape(xl_traj[k,0:3],(3,1))
    scPl[:,k:k+1]=np.reshape(scxl_traj[k,0:3],(3,1))
    ql_k        = np.reshape(xl_traj[k,6:10],(4,1))
    scql_k      = np.reshape(scxl_traj[k,6:10],(4,1))
    norm_2_Ql[k] = LA.norm(ql_k)
    Rl_k        = np.asarray(sysm.q_2_rotation(ql_k), dtype=np.float64)
    scRl_k      = np.asarray(sysm.q_2_rotation(scql_k), dtype=np.float64)
    rl_k        = Rot.from_matrix(Rl_k)
    scrl_k      = Rot.from_matrix(scRl_k)
    eulerl_k    = np.reshape(rl_k.as_euler('zyx',degrees=True),(3,1))
    sceuler_k   = np.reshape(scrl_k.as_euler('zyx',degrees=True),(3,1))
    Eulerl[:,k:k+1] = eulerl_k
    scEulerl[:,k:k+1] = sceuler_k

Xq         = [] # list that stores all quadrotors' predicted trajectories
Aq         = [] # list that stores all cable attachments' trajectories in the world frame
scXq       = [] # list that stores all quadrotors' safe copy predicted trajectories
refXq      = [] # list that stores all quadrotors' reference trajectories
Tq         = np.zeros((nq,horizon))
scTq       = np.zeros((nq,horizon))
for i in range(nq):
    Pi     = np.zeros((3,horizon))
    scPi   = np.zeros((3,horizon))
    refPi  = np.zeros((3,horizon))
    ri     = np.reshape(MPC_load.ra[:,i],(3,1))
    ai     = np.zeros((3,horizon))
    for k in range(horizon):
        pl_k   = np.reshape(xl_traj[k,0:3],(3,1))
        ql_k   = np.reshape(xl_traj[k,6:10],(4,1))
        Rl_k   = sysm.q_2_rotation(ql_k)
        di_k   = np.reshape(xq_traj[i][k,0:3],(3,1))
        ti_k   = xq_traj[i][k,12]
        scdi_k = np.reshape(scxq_traj[i][k,0:3],(3,1))
        scti_k = scxq_traj[i][k,12]
        ai_k   = pl_k + Rl_k@ri
        pi_k   = ai_k + cl0*di_k
        scpi_k = ai_k + cl0*scdi_k
        ref_plk= np.reshape(Ref_pl[0:3,k],(3,1)) + ri
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
ax1.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
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
fig1.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/quadrotor1_traj_'+str(nq)+'.png',dpi=400)
plt.show()

fig2, ax2 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax2.add_patch(obs1)
ax2.add_patch(obs2)
ax2.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
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
fig2.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/quadrotor2_traj_'+str(nq)+'.png',dpi=400)
plt.show()

fig3, ax3 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax3.add_patch(obs1)
ax3.add_patch(obs2)
ax3.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
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
fig3.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/quadrotor3_traj_'+str(nq)+'.png',dpi=400)
plt.show()


fig7, ax7 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob3[0,0],pob3[1,0]),rov,color='red',alpha=0.5)
obs2      = Circle((pob4[0,0],pob4[1,0]),rov,color='red',alpha=0.5)
ax7.plot([pob1[1,0],pob1[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax7.plot([pob2[1,0],pob2[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax7.add_patch(obs1)
ax7.add_patch(obs2)
ax7.plot(Xq[0][1,:],Xq[0][2,:],label='1st quadrotor',marker='o',markersize=2,linewidth=1)
ax7.plot(scXq[0][1,:],scXq[0][2,:],label='1st quadrotor_safe copy',color='black',marker='.',markersize=1,linewidth=1)
ax7.plot(refXq[0][1,:],refXq[0][2,:],label='1st quadrotor_ref',color='orange',marker='.',markersize=1,linewidth=1)
ax7.set_xlabel('y [m]')
ax7.set_ylabel('z [m]')
ax7.legend()
ax7.set_aspect('equal')
ax7.grid(True)
ax7.set_ylim(0, 3)
fig7.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/quadrotor1_traj_'+str(nq)+'_yz.png',dpi=400)
plt.show()

fig8, ax8 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob3[0,0],pob3[1,0]),rov,color='red',alpha=0.5)
obs2      = Circle((pob4[0,0],pob4[1,0]),rov,color='red',alpha=0.5)
ax8.plot([pob1[1,0],pob1[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax8.plot([pob2[1,0],pob2[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax8.add_patch(obs1)
ax8.add_patch(obs2)
ax8.plot(Xq[1][1,:],Xq[1][2,:],label='2nd quadrotor',marker='o',markersize=2,linewidth=1)
ax8.plot(scXq[1][1,:],scXq[1][2,:],label='2nd quadrotor_safe copy',color='black',marker='.',markersize=1,linewidth=1)
ax8.plot(refXq[1][1,:],refXq[1][2,:],label='2nd quadrotor_ref',color='orange',marker='.',markersize=1,linewidth=1)
ax8.set_xlabel('y [m]')
ax8.set_ylabel('z [m]')
ax8.legend()
ax8.set_aspect('equal')
ax8.grid(True)
ax8.set_ylim(0, 3)
fig8.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/quadrotor2_traj_'+str(nq)+'_yz.png',dpi=400)
plt.show()

fig9, ax9 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob3[0,0],pob3[1,0]),rov,color='red',alpha=0.5)
obs2      = Circle((pob4[0,0],pob4[1,0]),rov,color='red',alpha=0.5)
ax9.plot([pob1[1,0],pob1[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax9.plot([pob2[1,0],pob2[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax9.add_patch(obs1)
ax9.add_patch(obs2)
ax9.plot(Xq[2][1,:],Xq[2][2,:],label='3rd quadrotor',marker='o',markersize=2,linewidth=1)
ax9.plot(scXq[2][1,:],scXq[2][2,:],label='3rd quadrotor_safe copy',color='black',marker='.',markersize=1,linewidth=1)
ax9.plot(refXq[2][1,:],refXq[2][2,:],label='3rd quadrotor_ref',color='orange',marker='.',markersize=1,linewidth=1)
ax9.set_xlabel('y [m]')
ax9.set_ylabel('z [m]')
ax9.legend()
ax9.set_aspect('equal')
ax9.grid(True)
ax9.set_ylim(0, 3)
fig9.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/quadrotor3_traj_'+str(nq)+'_yz.png',dpi=400)
plt.show()

fig10, ax10 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax10.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
ax10.add_patch(obs1)
ax10.add_patch(obs2)
ax10.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
ax10.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
ax10.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
kt = 30
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
        ax10.add_patch(quad1)
        quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
        ax10.add_patch(quad2)
        quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
        ax10.add_patch(quad3)
        ax10.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
        ax10.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax10.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax10.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax10.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax10.plot([Aq[2][0,k],Aq[0][0,k]],[Aq[2][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)

    
ax10.set_xlabel('x [m]')
ax10.set_ylabel('y [m]')
ax10.set_aspect('equal')
ax10.legend()
ax10.grid(True)
# plt.axis('equal')
fig10.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'.png',dpi=400)
plt.show()


fig11, ax11 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax11.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
ax11.add_patch(obs1)
ax11.add_patch(obs2)
ax11.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
ax11.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
ax11.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
kt = 34
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
        ax11.add_patch(quad1)
        quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
        ax11.add_patch(quad2)
        quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
        ax11.add_patch(quad3)
        ax11.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
        ax11.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax11.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax11.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax11.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax11.plot([Aq[2][0,k],Aq[0][0,k]],[Aq[2][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)

    
ax11.set_xlabel('x [m]')
ax11.set_ylabel('y [m]')
ax11.set_aspect('equal')
ax11.legend()
ax11.grid(True)
# plt.axis('equal')
fig11.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'.png',dpi=400)
plt.show()

fig12, ax12 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax12.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
ax12.add_patch(obs1)
ax12.add_patch(obs2)
ax12.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
ax12.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
ax12.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
kt = 38
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
        ax12.add_patch(quad1)
        quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
        ax12.add_patch(quad2)
        quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
        ax12.add_patch(quad3)
        ax12.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
        ax12.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax12.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax12.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax12.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax12.plot([Aq[2][0,k],Aq[0][0,k]],[Aq[2][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)

    
ax12.set_xlabel('x [m]')
ax12.set_ylabel('y [m]')
ax12.set_aspect('equal')
ax12.legend()
ax12.grid(True)
# plt.axis('equal')
fig12.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'.png',dpi=400)
plt.show()

fig13, ax13 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax13.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
ax13.add_patch(obs1)
ax13.add_patch(obs2)
ax13.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
ax13.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
ax13.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
kt = 42
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
        ax13.add_patch(quad1)
        quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
        ax13.add_patch(quad2)
        quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
        ax13.add_patch(quad3)
        ax13.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
        ax13.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax13.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax13.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax13.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax13.plot([Aq[2][0,k],Aq[0][0,k]],[Aq[2][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)

    
ax13.set_xlabel('x [m]')
ax13.set_ylabel('y [m]')
ax13.set_aspect('equal')
ax13.legend()
ax13.grid(True)
# plt.axis('equal')
fig13.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'.png',dpi=400)
plt.show()


fig14, ax14 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax14.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
ax14.add_patch(obs1)
ax14.add_patch(obs2)
ax14.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
ax14.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
ax14.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
kt = 46
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
        ax14.add_patch(quad1)
        quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
        ax14.add_patch(quad2)
        quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
        ax14.add_patch(quad3)
        ax14.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
        ax14.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax14.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax14.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax14.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax14.plot([Aq[2][0,k],Aq[0][0,k]],[Aq[2][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)

    
ax14.set_xlabel('x [m]')
ax14.set_ylabel('y [m]')
ax14.set_aspect('equal')
ax14.legend()
ax14.grid(True)
# plt.axis('equal')
fig14.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'.png',dpi=400)
plt.show()

fig15, ax15 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax15.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
ax15.add_patch(obs1)
ax15.add_patch(obs2)
ax15.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
ax15.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
ax15.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
kt = 50
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
        ax15.add_patch(quad1)
        quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
        ax15.add_patch(quad2)
        quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
        ax15.add_patch(quad3)
        ax15.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
        ax15.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax15.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax15.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax15.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax15.plot([Aq[2][0,k],Aq[0][0,k]],[Aq[2][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)

    
ax15.set_xlabel('x [m]')
ax15.set_ylabel('y [m]')
ax15.set_aspect('equal')
ax15.legend()
ax15.grid(True)
# plt.axis('equal')
fig15.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'.png',dpi=400)
plt.show()

fig16, ax16 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax16.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
ax16.add_patch(obs1)
ax16.add_patch(obs2)
ax16.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
ax16.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
ax16.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
kt = 54
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
        ax16.add_patch(quad1)
        quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
        ax16.add_patch(quad2)
        quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
        ax16.add_patch(quad3)
        ax16.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
        ax16.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax16.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax16.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax16.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax16.plot([Aq[2][0,k],Aq[0][0,k]],[Aq[2][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)

    
ax16.set_xlabel('x [m]')
ax16.set_ylabel('y [m]')
ax16.set_aspect('equal')
ax16.legend()
ax16.grid(True)
# plt.axis('equal')
fig16.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'.png',dpi=400)
plt.show()


fig17, ax17 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax17.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
ax17.add_patch(obs1)
ax17.add_patch(obs2)
ax17.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
ax17.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
ax17.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
kt = 58
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
        ax17.add_patch(quad1)
        quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
        ax17.add_patch(quad2)
        quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
        ax17.add_patch(quad3)
        ax17.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
        ax17.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax17.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax17.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax17.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax17.plot([Aq[2][0,k],Aq[0][0,k]],[Aq[2][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)

    
ax17.set_xlabel('x [m]')
ax17.set_ylabel('y [m]')
ax17.set_aspect('equal')
ax17.legend()
ax17.grid(True)
# plt.axis('equal')
fig17.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'.png',dpi=400)
plt.show()


fig18, ax18 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax18.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
ax18.add_patch(obs1)
ax18.add_patch(obs2)
ax18.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
ax18.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
ax18.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
kt = 62
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
        ax18.add_patch(quad1)
        quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
        ax18.add_patch(quad2)
        quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
        ax18.add_patch(quad3)
        ax18.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
        ax18.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax18.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax18.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax18.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax18.plot([Aq[2][0,k],Aq[0][0,k]],[Aq[2][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)

    
ax18.set_xlabel('x [m]')
ax18.set_ylabel('y [m]')
ax18.set_aspect('equal')
ax18.legend()
ax18.grid(True)
# plt.axis('equal')
fig18.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'.png',dpi=400)
plt.show()


fig19, ax19 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax19.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
ax19.add_patch(obs1)
ax19.add_patch(obs2)
ax19.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
ax19.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
ax19.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
kt = 66
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
        ax19.add_patch(quad1)
        quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
        ax19.add_patch(quad2)
        quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
        ax19.add_patch(quad3)
        ax19.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
        ax19.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax19.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax19.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax19.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax19.plot([Aq[2][0,k],Aq[0][0,k]],[Aq[2][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)

    
ax19.set_xlabel('x [m]')
ax19.set_ylabel('y [m]')
ax19.set_aspect('equal')
ax19.legend()
ax19.grid(True)
# plt.axis('equal')
fig19.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'.png',dpi=400)
plt.show()


fig20, ax20 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob1[0,0],pob1[1,0]),ro,color='red',alpha=0.5)
obs2      = Circle((pob2[0,0],pob2[1,0]),ro,color='red',alpha=0.5)
ax20.plot([-2.5,2.5],[pob3[0,0],pob4[0,0]], marker='o',color='red',alpha=0.5,label='horizontal gap')
ax20.add_patch(obs1)
ax20.add_patch(obs2)
ax20.plot(Ref_pl[0,:],Ref_pl[1,:],label='Ref',linewidth=1,linestyle='--')
ax20.plot(Pl[0,:],Pl[1,:],label='Planned_SubP1',linewidth=1)
ax20.plot(scPl[0,:],scPl[1,:],label='Planned_SubP2',linewidth=1)
kt = 70
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][0,k],Xq[0][1,k]),rq,fill=False)
        ax20.add_patch(quad1)
        quad2  = Circle((Xq[1][0,k],Xq[1][1,k]),rq,fill=False)
        ax20.add_patch(quad2)
        quad3  = Circle((Xq[2][0,k],Xq[2][1,k]),rq,fill=False)
        ax20.add_patch(quad3)
        ax20.plot((Xq[0][0,k],Aq[0][0,k]),[Xq[0][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)
        ax20.plot([Xq[1][0,k],Aq[1][0,k]],[Xq[1][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax20.plot([Xq[2][0,k],Aq[2][0,k]],[Xq[2][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax20.plot([Aq[0][0,k],Aq[1][0,k]],[Aq[0][1,k],Aq[1][1,k]],color='blue',linewidth=0.5)
        ax20.plot([Aq[1][0,k],Aq[2][0,k]],[Aq[1][1,k],Aq[2][1,k]],color='blue',linewidth=0.5)
        ax20.plot([Aq[2][0,k],Aq[0][0,k]],[Aq[2][1,k],Aq[0][1,k]],color='blue',linewidth=0.5)

    
ax20.set_xlabel('x [m]')
ax20.set_ylabel('y [m]')
ax20.set_aspect('equal')
ax20.legend()
ax20.grid(True)
# plt.axis('equal')
fig20.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'.png',dpi=400)
plt.show()


fig22, ax22 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob3[0,0],pob3[1,0]),rov,color='red',alpha=0.5)
obs2      = Circle((pob4[0,0],pob4[1,0]),rov,color='red',alpha=0.5)
ax22.plot([pob1[1,0],pob1[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax22.plot([pob2[1,0],pob2[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax22.add_patch(obs1)
ax22.add_patch(obs2)
ax22.plot(Ref_pl[1,:],Ref_pl[2,:],label='Ref',linewidth=1,linestyle='--')
ax22.plot(Pl[1,:],Pl[2,:],label='Planned_SubP1',linewidth=1)
ax22.plot(scPl[1,:],scPl[2,:],label='Planned_SubP2',linewidth=1)
kt = 46
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][1,k],Xq[0][2,k]),0.5*rq,fill=False)
        ax22.add_patch(quad1)
        quad2  = Circle((Xq[1][1,k],Xq[1][2,k]),0.5*rq,fill=False)
        ax22.add_patch(quad2)
        quad3  = Circle((Xq[2][1,k],Xq[2][2,k]),0.5*rq,fill=False)
        ax22.add_patch(quad3)
        ax22.plot((Xq[0][1,k],Aq[0][1,k]),[Xq[0][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
        ax22.plot([Xq[1][1,k],Aq[1][1,k]],[Xq[1][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax22.plot([Xq[2][1,k],Aq[2][1,k]],[Xq[2][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax22.plot([Aq[0][1,k],Aq[1][1,k]],[Aq[0][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax22.plot([Aq[1][1,k],Aq[2][1,k]],[Aq[1][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax22.plot([Aq[2][1,k],Aq[0][1,k]],[Aq[2][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
ax22.set_xlabel('y [m]')
ax22.set_ylabel('z [m]')
ax22.set_aspect('equal')
ax22.set_ylim(0, 3)
ax22.legend()
ax22.grid(True)
fig22.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'_yz.png',dpi=400)
plt.show()


fig23, ax23 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob3[0,0],pob3[1,0]),rov,color='red',alpha=0.5)
obs2      = Circle((pob4[0,0],pob4[1,0]),rov,color='red',alpha=0.5)
ax23.plot([pob1[1,0],pob1[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax23.plot([pob2[1,0],pob2[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax23.add_patch(obs1)
ax23.add_patch(obs2)
ax23.plot(Ref_pl[1,:],Ref_pl[2,:],label='Ref',linewidth=1,linestyle='--')
ax23.plot(Pl[1,:],Pl[2,:],label='Planned_SubP1',linewidth=1)
ax23.plot(scPl[1,:],scPl[2,:],label='Planned_SubP2',linewidth=1)
kt = 50
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][1,k],Xq[0][2,k]),0.5*rq,fill=False)
        ax23.add_patch(quad1)
        quad2  = Circle((Xq[1][1,k],Xq[1][2,k]),0.5*rq,fill=False)
        ax23.add_patch(quad2)
        quad3  = Circle((Xq[2][1,k],Xq[2][2,k]),0.5*rq,fill=False)
        ax23.add_patch(quad3)
        ax23.plot((Xq[0][1,k],Aq[0][1,k]),[Xq[0][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
        ax23.plot([Xq[1][1,k],Aq[1][1,k]],[Xq[1][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax23.plot([Xq[2][1,k],Aq[2][1,k]],[Xq[2][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax23.plot([Aq[0][1,k],Aq[1][1,k]],[Aq[0][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax23.plot([Aq[1][1,k],Aq[2][1,k]],[Aq[1][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax23.plot([Aq[2][1,k],Aq[0][1,k]],[Aq[2][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
ax23.set_xlabel('y [m]')
ax23.set_ylabel('z [m]')
ax23.set_aspect('equal')
ax23.set_ylim(0, 3)
ax23.legend()
ax23.grid(True)
fig23.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'_yz.png',dpi=400)
plt.show()


fig24, ax24 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob3[0,0],pob3[1,0]),rov,color='red',alpha=0.5)
obs2      = Circle((pob4[0,0],pob4[1,0]),rov,color='red',alpha=0.5)
ax24.plot([pob1[1,0],pob1[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax24.plot([pob2[1,0],pob2[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax24.add_patch(obs1)
ax24.add_patch(obs2)
ax24.plot(Ref_pl[1,:],Ref_pl[2,:],label='Ref',linewidth=1,linestyle='--')
ax24.plot(Pl[1,:],Pl[2,:],label='Planned_SubP1',linewidth=1)
ax24.plot(scPl[1,:],scPl[2,:],label='Planned_SubP2',linewidth=1)
kt = 54
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][1,k],Xq[0][2,k]),0.5*rq,fill=False)
        ax24.add_patch(quad1)
        quad2  = Circle((Xq[1][1,k],Xq[1][2,k]),0.5*rq,fill=False)
        ax24.add_patch(quad2)
        quad3  = Circle((Xq[2][1,k],Xq[2][2,k]),0.5*rq,fill=False)
        ax24.add_patch(quad3)
        ax24.plot((Xq[0][1,k],Aq[0][1,k]),[Xq[0][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
        ax24.plot([Xq[1][1,k],Aq[1][1,k]],[Xq[1][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax24.plot([Xq[2][1,k],Aq[2][1,k]],[Xq[2][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax24.plot([Aq[0][1,k],Aq[1][1,k]],[Aq[0][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax24.plot([Aq[1][1,k],Aq[2][1,k]],[Aq[1][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax24.plot([Aq[2][1,k],Aq[0][1,k]],[Aq[2][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
ax24.set_xlabel('y [m]')
ax24.set_ylabel('z [m]')
ax24.set_aspect('equal')
ax24.set_ylim(0, 3)
ax24.legend()
ax24.grid(True)
fig24.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'_yz.png',dpi=400)
plt.show()

fig25, ax25 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob3[0,0],pob3[1,0]),rov,color='red',alpha=0.5)
obs2      = Circle((pob4[0,0],pob4[1,0]),rov,color='red',alpha=0.5)
ax25.plot([pob1[1,0],pob1[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax25.plot([pob2[1,0],pob2[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax25.add_patch(obs1)
ax25.add_patch(obs2)
ax25.plot(Ref_pl[1,:],Ref_pl[2,:],label='Ref',linewidth=1,linestyle='--')
ax25.plot(Pl[1,:],Pl[2,:],label='Planned_SubP1',linewidth=1)
ax25.plot(scPl[1,:],scPl[2,:],label='Planned_SubP2',linewidth=1)
kt = 58
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][1,k],Xq[0][2,k]),0.5*rq,fill=False)
        ax25.add_patch(quad1)
        quad2  = Circle((Xq[1][1,k],Xq[1][2,k]),0.5*rq,fill=False)
        ax25.add_patch(quad2)
        quad3  = Circle((Xq[2][1,k],Xq[2][2,k]),0.5*rq,fill=False)
        ax25.add_patch(quad3)
        ax25.plot((Xq[0][1,k],Aq[0][1,k]),[Xq[0][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
        ax25.plot([Xq[1][1,k],Aq[1][1,k]],[Xq[1][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax25.plot([Xq[2][1,k],Aq[2][1,k]],[Xq[2][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax25.plot([Aq[0][1,k],Aq[1][1,k]],[Aq[0][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax25.plot([Aq[1][1,k],Aq[2][1,k]],[Aq[1][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax25.plot([Aq[2][1,k],Aq[0][1,k]],[Aq[2][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
ax25.set_xlabel('y [m]')
ax25.set_ylabel('z [m]')
ax25.set_aspect('equal')
ax25.set_ylim(0, 3)
ax25.legend()
ax25.grid(True)
fig25.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'_yz.png',dpi=400)
plt.show()

fig26, ax26 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob3[0,0],pob3[1,0]),rov,color='red',alpha=0.5)
obs2      = Circle((pob4[0,0],pob4[1,0]),rov,color='red',alpha=0.5)
ax26.plot([pob1[1,0],pob1[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax26.plot([pob2[1,0],pob2[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax26.add_patch(obs1)
ax26.add_patch(obs2)
ax26.plot(Ref_pl[1,:],Ref_pl[2,:],label='Ref',linewidth=1,linestyle='--')
ax26.plot(Pl[1,:],Pl[2,:],label='Planned_SubP1',linewidth=1)
ax26.plot(scPl[1,:],scPl[2,:],label='Planned_SubP2',linewidth=1)
kt = 62
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][1,k],Xq[0][2,k]),0.5*rq,fill=False)
        ax26.add_patch(quad1)
        quad2  = Circle((Xq[1][1,k],Xq[1][2,k]),0.5*rq,fill=False)
        ax26.add_patch(quad2)
        quad3  = Circle((Xq[2][1,k],Xq[2][2,k]),0.5*rq,fill=False)
        ax26.add_patch(quad3)
        ax26.plot((Xq[0][1,k],Aq[0][1,k]),[Xq[0][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
        ax26.plot([Xq[1][1,k],Aq[1][1,k]],[Xq[1][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax26.plot([Xq[2][1,k],Aq[2][1,k]],[Xq[2][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax26.plot([Aq[0][1,k],Aq[1][1,k]],[Aq[0][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax26.plot([Aq[1][1,k],Aq[2][1,k]],[Aq[1][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax26.plot([Aq[2][1,k],Aq[0][1,k]],[Aq[2][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
ax26.set_xlabel('y [m]')
ax26.set_ylabel('z [m]')
ax26.set_aspect('equal')
ax26.set_ylim(0, 3)
ax26.legend()
ax26.grid(True)
fig26.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'_yz.png',dpi=400)
plt.show()

fig27, ax27 = plt.subplots(figsize=(5,5),dpi=300)
obs1      = Circle((pob3[0,0],pob3[1,0]),rov,color='red',alpha=0.5)
obs2      = Circle((pob4[0,0],pob4[1,0]),rov,color='red',alpha=0.5)
ax27.plot([pob1[1,0],pob1[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax27.plot([pob2[1,0],pob2[1,0]],[0,3], marker='o',color='red',alpha=0.5,label='vertical gap')
ax27.add_patch(obs1)
ax27.add_patch(obs2)
ax27.plot(Ref_pl[1,:],Ref_pl[2,:],label='Ref',linewidth=1,linestyle='--')
ax27.plot(Pl[1,:],Pl[2,:],label='Planned_SubP1',linewidth=1)
ax27.plot(scPl[1,:],scPl[2,:],label='Planned_SubP2',linewidth=1)
kt = 66
ratio = (horizon/100)
for k in range(horizon):
    if k==2 or k==int(kt*ratio) or k==int(99*ratio):
        quad1  = Circle((Xq[0][1,k],Xq[0][2,k]),0.5*rq,fill=False)
        ax27.add_patch(quad1)
        quad2  = Circle((Xq[1][1,k],Xq[1][2,k]),0.5*rq,fill=False)
        ax27.add_patch(quad2)
        quad3  = Circle((Xq[2][1,k],Xq[2][2,k]),0.5*rq,fill=False)
        ax27.add_patch(quad3)
        ax27.plot((Xq[0][1,k],Aq[0][1,k]),[Xq[0][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
        ax27.plot([Xq[1][1,k],Aq[1][1,k]],[Xq[1][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax27.plot([Xq[2][1,k],Aq[2][1,k]],[Xq[2][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax27.plot([Aq[0][1,k],Aq[1][1,k]],[Aq[0][2,k],Aq[1][2,k]],color='blue',linewidth=0.5)
        ax27.plot([Aq[1][1,k],Aq[2][1,k]],[Aq[1][2,k],Aq[2][2,k]],color='blue',linewidth=0.5)
        ax27.plot([Aq[2][1,k],Aq[0][1,k]],[Aq[2][2,k],Aq[0][2,k]],color='blue',linewidth=0.5)
ax27.set_xlabel('y [m]')
ax27.set_ylabel('z [m]')
ax27.set_aspect('equal')
ax27.set_ylim(0, 3)
ax27.legend()
ax27.grid(True)
fig27.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/system_traj_quadrotor_num_'+str(nq)+'_'+str(kt)+'_yz.png',dpi=400)
plt.show()

plt.figure(55,figsize=(6,4),dpi=300)
plt.plot(Time,Tq[0,:],linewidth=1,label='1st cable')
plt.plot(Time,Tq[1,:],linewidth=1,label='2nd cable')
plt.plot(Time,Tq[2,:],linewidth=1,label='3rd cable')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('MPC tension force [N]')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/cable_MPC_tensions_'+str(nq)+'.png',dpi=400)
plt.show()

plt.figure(56,figsize=(6,4),dpi=300)
plt.plot(Time,Eulerl[0,:],linewidth=1,label='roll')
plt.plot(Time,scEulerl[0,:],linewidth=1,label='safe-roll')
plt.plot(Time,Eulerl[1,:],linewidth=1,label='pitch')
plt.plot(Time,scEulerl[1,:],linewidth=1,label='safe-pitch')
plt.plot(Time,Eulerl[2,:],linewidth=1,label='yaw')
plt.plot(Time,scEulerl[2,:],linewidth=1,label='safe-yaw')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Euler angle [deg]')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/euler_'+str(nq)+'.png',dpi=400)
plt.show()

plt.figure(57,figsize=(6,4),dpi=300)
plt.plot(Time,ul_traj[:,0],linewidth=1,label='fx')
plt.plot(Time,scul_traj[:,0],linewidth=1,label='safe fx')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Fx [N]')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/Fx_'+str(nq)+'.png',dpi=400)
plt.show()


plt.figure(58,figsize=(6,4),dpi=300)
plt.plot(Time,ul_traj[:,1],linewidth=1,label='fy')
plt.plot(Time,scul_traj[:,1],linewidth=1,label='safe fy')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Fy [N]')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/Fy_'+str(nq)+'.png',dpi=400)
plt.show()

plt.figure(59,figsize=(6,4),dpi=300)
plt.plot(Time,ul_traj[:,2],linewidth=1,label='fz')
plt.plot(Time,scul_traj[:,2],linewidth=1,label='safe fz')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Fz [N]')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/Fz_'+str(nq)+'.png',dpi=400)
plt.show()

plt.figure(60,figsize=(6,4),dpi=300)
plt.plot(Time,ul_traj[:,3],linewidth=1,label='Mx')
plt.plot(Time,scul_traj[:,3],linewidth=1,label='safe Mx')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Mx [Nm]')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/Mx_'+str(nq)+'.png',dpi=400)
plt.show()


plt.figure(61,figsize=(6,4),dpi=300)
plt.plot(Time,ul_traj[:,4],linewidth=1,label='My')
plt.plot(Time,scul_traj[:,4],linewidth=1,label='safe My')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('My [Nm]')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/My_'+str(nq)+'.png',dpi=400)
plt.show()

plt.figure(62,figsize=(6,4),dpi=300)
plt.plot(Time,ul_traj[:,5],linewidth=1,label='Mz')
plt.plot(Time,scul_traj[:,5],linewidth=1,label='safe Mz')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Mz [Nm]')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/Mz_'+str(nq)+'.png',dpi=400)
plt.show()


plt.figure(63,figsize=(6,4),dpi=300)
plt.plot(Time,uq_traj[0][:,0],linewidth=1,label='dddwx')
plt.plot(Time,scuq_traj[0][:,0],linewidth=1,label='safe dddwx')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('dddwx')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/uc_x_quadrotor_1_'+str(nq)+'.png',dpi=400)
plt.show()

plt.figure(64,figsize=(6,4),dpi=300)
plt.plot(Time,uq_traj[0][:,1],linewidth=1,label='dddwy')
plt.plot(Time,scuq_traj[0][:,1],linewidth=1,label='safe dddwy')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('dddwy')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/uc_y_quadrotor_1_'+str(nq)+'.png',dpi=400)
plt.show()

plt.figure(65,figsize=(6,4),dpi=300)
plt.plot(Time,uq_traj[0][:,2],linewidth=1,label='dddwz')
plt.plot(Time,scuq_traj[0][:,2],linewidth=1,label='safe dddwz')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('dddwz')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/uc_z_quadrotor_1_'+str(nq)+'.png',dpi=400)
plt.show()

plt.figure(66,figsize=(6,4),dpi=300)
plt.plot(Time,uq_traj[0][:,3],linewidth=1,label='ddt')
plt.plot(Time,scuq_traj[0][:,3],linewidth=1,label='safe ddt')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('ddt')
plt.grid()
plt.savefig('Planning_plots_multiagent_meta_evaluation_COM_Dyn_HV/ddt_quadrotor_1_'+str(nq)+'.png',dpi=400)
plt.show()
