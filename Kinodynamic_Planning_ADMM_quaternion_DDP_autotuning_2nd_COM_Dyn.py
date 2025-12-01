from casadi import *
import numpy as np
from numpy import linalg as LA
import math
from scipy.spatial.transform import Rotation as Rot
from scipy import linalg as sLA
from scipy.linalg import null_space
import time as TM
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import ArpackNoConvergence

"""
Reference
[1] Cao, K., Xu, X., Jin, W., Johansson, K.H. and Xie, L., 2025. 
    A differential dynamic programming framework for inverse reinforcement learning. IEEE Transactions on Robotics.
[2] Jin, W., Wang, Z., Yang, Z. and Mou, S., 2020. 
    Pontryagin differentiable programming: An end-to-end learning and control framework. 
    Advances in Neural Information Processing Systems, 33, pp.7979-7992.

"""

class MPC_Planner:
    def __init__(self, sysm_para, dt_ctrl, horizon):
        # Payload's parameters
        self.m1     = sysm_para[0] # the payload's mass [kg]
        self.m2     = sysm_para[1] # the added mass [kg]
        self.Jlcom  = np.diag(sysm_para[2:5]) # rotational inertia of m1 about its Geometric Center (GC)
        self.rl     = sysm_para[5] # the radius of load [m]
        self.ml     = self.m1 + self.m2 # the total mass [kg]
        # Quadrotor's parameters
        self.nq     = sysm_para[6] # the number of quadrotors
        self.rq     = sysm_para[7] # the radius of quadrotor [m]
        self.mq     = sysm_para[8] # the quadrotor's mass [kg]
        self.fmax   = sysm_para[9] # the maximum quadrotor's thrust [N]
        self.vmax   = sysm_para[10]# the maximum quadrotor's velocity [m/s]
        # Cable and obstacle's parameters
        self.cl0    = sysm_para[11] # the cable length [m]
        self.ro     = sysm_para[12] # the radius of obstacle [m]
        # Unit direction vector free of coordinate
        self.ex     = np.array([[1, 0, 0]]).T
        self.ey     = np.array([[0, 1, 0]]).T
        self.ez     = np.array([[0, 0, 1]]).T
        # Gravitational acceleration
        self.g      = 9.81      
        self.dt     = dt_ctrl
        # MPC's horizon
        self.N      = horizon
        # barrier parameter
        self.p_bar  = 1e-6
    
    def Rotational_Inertia(self,rp):
        # rp=(x,y,0), a column vector, is the coordinate of the point-mass added on the uniform circular plate in its body frame 
        ratio_m    = self.m1*self.m2/self.ml
        self.Jl    = self.Jlcom + ratio_m*(rp.T@rp*np.identity(3)-rp@rp.T)

    def allocation_martrix(self,rg):
        self.alpha  = 2*np.pi/self.nq
        r0          = np.array([[self.rl,0,0]]).T - np.reshape(np.vstack((rg[0],rg[1],0)),(3,1))  # 1st cable attachment point in {Bl}
        self.ra     = r0
        S_r0        = self.skew_sym_numpy(r0)
        I3          = np.identity(3) # 3-by-3 identity matrix
        self.Pt      = np.vstack((I3,S_r0))
        for i in range(int(self.nq)-1):
            ri      = np.array([[self.rl*(math.cos((i+1)*self.alpha)),self.rl*(math.sin((i+1)*self.alpha)),0]]).T - np.reshape(np.vstack((rg[0],rg[1],0)),(3,1))
            S_ri    = self.skew_sym_numpy(ri)
            Pi      = np.vstack((I3,S_ri))
            self.Pt = np.append(self.Pt,Pi,axis=1) # the tension mapping matrix: 6-by-3nq with a rank of 6
            self.ra = np.append(self.ra,ri,axis=1) # a matrix that stores the attachment points
    
    def skew_sym_numpy(self, v):
        v_cross = np.array([
            [0, -v[2, 0], v[1, 0]],
            [v[2, 0], 0, -v[0, 0]],
            [-v[1, 0], v[0, 0], 0]]
        )
        return v_cross
    
    def skew_sym(self, v): # skew-symmetric operator
        v_cross = vertcat(
            horzcat(0, -v[2,0], v[1,0]),
            horzcat(v[2,0], 0, -v[0,0]),
            horzcat(-v[1,0], v[0,0], 0)
        )
        return v_cross


    def SetStateVariables(self, xl, xi):
        self.xl    = xl
        self.xi    = xi
        self.nxl   = xl.numel()
        self.nxi   = xi.numel()
        self.scxc  = SX.sym('scxc',self.nxi*int(self.nq))
        self.xc    = SX.sym('xc',self.nxi*int(self.nq))
        self.scxC  = SX.sym('scxC',self.nxi*int(self.nq))
        self.scxl  = SX.sym('scxl',self.nxl)
        self.scxi  = SX.sym('scxi',self.nxi)
        self.scxL  = SX.sym('scxL',self.nxl) # Lagrangian multiplier of xl
        self.scxI  = SX.sym('scxI',self.nxi) # Lagrangian multiplier of xi
        self.xl_lb = self.nxl*[-1e19]
        self.xl_ub = self.nxl*[1e19]
        self.xi_lb = self.nxi*[-1e19]
        self.xi_ub = self.nxi*[1e19]
        self.t_min = 0.01
        self.t_max = 10 # the maximum tension force
        self.scxi_lb = [-1e19,-1e19,-1e19, -1e19,-1e19,-1e19, self.t_min,-1e19]
        self.scxi_ub = [1e19,1e19,1e19, 1e19,1e19,1e19, self.t_max, 1e19]


    def SetCtrlVariables(self, ul, ui):
        self.ul    = ul
        self.ui    = ui
        self.nul   = ul.numel()
        self.nui   = ui.numel()
        self.scuc  = SX.sym('scuc',self.nui*int(self.nq))
        self.uc    = SX.sym('uc',self.nui*int(self.nq))
        self.scuC  = SX.sym('scuC',self.nui*int(self.nq))
        self.scul  = SX.sym('scul',self.nul)
        self.scui  = SX.sym('scui',self.nui)
        self.scuL  = SX.sym('scuL',self.nul) # Lagrangian multiplier of ul
        self.scuI  = SX.sym('scuI',self.nui) # Lagrangian multiplier of ui
        self.ul_lb = self.nul*[-1e19]
        self.ul_ub = self.nul*[1e19]
        self.ui_lb = self.nui*[-1e19]
        self.ui_ub = self.nui*[1e19]

    def SetDyns(self, model_l, model_i):
        self.model_l = self.xl + self.dt*model_l # 4th-order Runge-Kutta discrete-time load dynamics model
        self.model_i = self.xi + self.dt*model_i # 4th-order Runge-Kutta discrete-time cable dynamics model
        self.model_l_fn = Function('mdynl',[self.xl, self.ul],[self.model_l],['xl0','ul0'],['mdynlf'])
        self.model_i_fn = Function('mdyni',[self.xi, self.ui],[self.model_i],['xi0','ui0'],['mdynif'])

    def SetWeightPara(self):
        # self.nwsl    = self.nxl
        self.para_l  = SX.sym('paral',1,2*self.nxl+self.nul+1) # including the ADMM penalty parameter
        self.npl     = self.para_l.numel()
        self.para_i  = SX.sym('parai',1,2*self.nxi+self.nui+1) # including the ADMM penalty parameter
        self.npi     = self.para_i.numel()
        self.P_auto  = horzcat(self.para_l,self.para_i)
        self.n_Pauto = self.P_auto.numel()

    def Discount_rate(self,a):
        gamma_a = 1/(1+exp(-a)) # Sigmoid function-based discount rate during ADMM iterations
        return gamma_a

    def q_2_rotation(self, q): # from body frame to inertial frame
        # no normalization to avoid singularity in optimization
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3] # q0 denotes a scalar while q1, q2, and q3 represent rotational axes x, y, and z, respectively
        R = vertcat(
        horzcat( 2 * (q0 ** 2 + q1 ** 2) - 1, 2 * q1 * q2 - 2 * q0 * q3, 2 * q0 * q2 + 2 * q1 * q3),
        horzcat(2 * q0 * q3 + 2 * q1 * q2, 2 * (q0 ** 2 + q2 ** 2) - 1, 2 * q2 * q3 - 2 * q0 * q1),
        horzcat(2 * q1 * q3 - 2 * q0 * q2, 2 * q0 * q1 + 2 * q2 * q3, 2 * (q0 ** 2 + q3 ** 2) - 1)
        )
        return R
    
    
    def vee_map(self, v):
        vect = vertcat(v[2, 1], v[0, 2], v[1, 0])
        return vect

    def SetPayloadCostDyn(self):
        self.ref_xl   = SX.sym('refxl',self.nxl,1)
        self.ref_ul   = SX.sym('reful',self.nul,1) 
        track_error_l = self.xl - self.ref_xl
        ctrl_error_l  = self.ul - self.ref_ul
        self.a        = SX.sym('a',1) # the ADMM iteration index
        self.dis_r    = self.Discount_rate(self.a) # iteration-specific discount rate, a global variable within the class
        self.Ql_k     = diag(self.para_l[0,0:self.nxl])
        self.Ql_N     = diag(self.para_l[0,self.nxl:2*self.nxl])
        self.Rl_k     = diag(self.para_l[0,2*self.nxl:2*self.nxl+self.nul])
        self.p        = self.dis_r*self.para_l[-1]
        # path cost
        self.resid_xl = self.xl - self.scxl + self.scxL/self.p
        self.resid_ul = self.ul - self.scul + self.scuL/self.p
        self.Jl_k     = 1/2 * (track_error_l.T@self.Ql_k@track_error_l + ctrl_error_l.T@self.Rl_k@ctrl_error_l) + self.p/2*self.resid_xl.T@self.resid_xl + self.p/2*self.resid_ul.T@self.resid_ul
        self.Jl_kfn   = Function('Jl_k',[self.xl, self.ul, self.scxl, self.scxL, self.scul, self.scuL, self.ref_xl, self.ref_ul, self.para_l, self.a],[self.Jl_k],['xl0', 'ul0', 'scxl0', 'scxL0', 'scul0', 'scuL0', 'refxl0', 'reful0', 'paral0', 'a0'],['Jl_kf'])
        # terminal cost
        self.Jl_N     = 1/2 * track_error_l.T@self.Ql_N@track_error_l + self.p/2*self.resid_xl.T@self.resid_xl
        self.Jl_Nfn   = Function('Jl_N',[self.xl, self.ref_xl, self.scxl, self.scxL, self.para_l, self.a],[self.Jl_N],['xl0', 'refxl0', 'scxl0', 'scxL0', 'paral0', 'a0'],['Jl_Nf'])
        # path cost of ADMM subproblem2
        self.Jl_P2_k  = self.p/2*self.resid_xl.T@self.resid_xl + self.p/2*self.resid_ul.T@self.resid_ul 
        self.Jl_P2_k_fn = Function('Jl_P2_k',[self.xl, self.ul, self.scxl, self.scxL, self.scul, self.scuL, self.para_l, self.a],[self.Jl_P2_k],['xl0', 'ul0', 'scxl0', 'scxL0', 'scul0', 'scuL0', 'paral0', 'a0'],['Jl_P2_kf'])
        # terminal cost of ADMM subproblem2
        self.Jl_P2_N  = self.p/2*self.resid_xl.T@self.resid_xl 
        self.Jl_P2_N_fn = Function('Jl_P2_N',[self.xl, self.scxl, self.scxL, self.para_l, self.a],[self.Jl_P2_N],['xl0', 'scxl0', 'scxL0', 'paral0', 'a0'],['Jl_P2_Nf'])


    def SetCableCostDyn(self):
        self.ref_xi   = SX.sym('refxi',self.nxi,1)
        self.ref_ui   = SX.sym('refui',self.nui,1)
        track_error_i = self.xi - self.ref_xi
        ctrl_error_i  = self.ui - self.ref_ui
        self.Qi_k     = diag(self.para_i[0,0:self.nxi])
        self.Qi_N     = diag(self.para_i[0,self.nxi:2*self.nxi])
        self.Ri_k     = diag(self.para_i[0,2*self.nxi:2*self.nxi+self.nui])
        self.pi       = self.dis_r*self.para_i[-1]
        # path cost
        self.resid_xi = self.xi - self.scxi + self.scxI/self.pi
        self.resid_ui = self.ui - self.scui + self.scuI/self.pi   
        self.Ji_k     = 1/2 * (track_error_i.T@self.Qi_k@track_error_i + ctrl_error_i.T@self.Ri_k@ctrl_error_i) + self.pi/2*self.resid_xi.T@self.resid_xi + self.pi/2*self.resid_ui.T@self.resid_ui 
        self.Ji_k_fn  = Function('Ji_k',[self.xi, self.ui, self.scxi, self.scxI, self.scui, self.scuI, self.ref_xi, self.ref_ui, self.para_i, self.a],[self.Ji_k],['xi0', 'ui0', 'scxi0', 'scxI0', 'scui0', 'scuI0', 'refxi0', 'refui0', 'parai0', 'a0'],['Ji_kf'])
        # terminal cost
        self.Ji_N     = 1/2 * track_error_i.T@self.Qi_N@track_error_i + self.pi/2*self.resid_xi.T@self.resid_xi 
        self.Ji_N_fn  = Function('Ji_N',[self.xi, self.ref_xi, self.scxi, self.scxI, self.para_i, self.a],[self.Ji_N],['xi0', 'refxi0', 'scxi0', 'scxI0', 'parai0', 'a0'],['Ji_Nf'])
        # path cost of ADMM subproblem2
        self.Ji_P2_k  = self.pi/2*self.resid_xi.T@self.resid_xi + self.pi/2*self.resid_ui.T@self.resid_ui 
        self.Ji_P2_k_fn = Function('Ji_P2_k',[self.xi, self.scxi, self.scxI, self.ui, self.scui, self.scuI, self.para_i, self.a],[self.Ji_P2_k],['xi0', 'scxi0', 'scxI0', 'ui0', 'scui0', 'scuI0', 'parai0', 'a0'],['Ji_P2_kf'])
        # terminal cost of ADMM subproblem2
        self.Ji_P2_N  = self.pi/2*self.resid_xi.T@self.resid_xi 
        self.Ji_P2_N_fn = Function('Ji_P2_N',[self.xi, self.scxi, self.scxI, self.para_i, self.a],[self.Ji_P2_N],['xi0', 'scxi0', 'scxI0', 'parai0', 'a0'],['Jl_P2_Nf'])

    def Load_derivatives_DDP_ADMM(self):
        # alpha = 1
        self.Vxl      = SX.sym('Vxl',self.nxl)
        self.Vxlxl    = SX.sym('Vxlxl',self.nxl,self.nxl)
        # gradients of the system dynamics, the cost function, and the Q value function
        self.Fxl      = jacobian(self.model_l,self.xl)
        self.Fxl_fn   = Function('Fxl',[self.xl,self.ul],[self.Fxl],['xl0','ul0'],['Fxl_f'])
        self.Ful      = jacobian(self.model_l,self.ul)
        self.Ful_fn   = Function('Ful',[self.xl,self.ul],[self.Ful],['xl0','ul0'],['Ful_f'])
        self.lxl      = jacobian(self.Jl_k,self.xl)
        self.lxlN     = jacobian(self.Jl_N,self.xl)
        self.lxlN_fn  = Function('lxlN',[self.xl,self.ref_xl,self.scxl,self.scxL,self.para_l,self.a],[self.lxlN],['xl0', 'refxl0', 'scxl0', 'scxL0', 'paral0', 'a0'],['lxlN_f'])
        self.lul      = jacobian(self.Jl_k,self.ul)
        self.Qxl      = self.lxl.T + self.Fxl.T@self.Vxl
        self.Qxl_fn   = Function('Qxl',[self.xl,self.ul,self.Vxl,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.Qxl],['xl0','ul0','Vxl0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['Qxl_f'])
        self.Qul      = self.lul.T + self.Ful.T@self.Vxl
        self.Qul_fn   = Function('Qul',[self.xl,self.ul,self.Vxl,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.Qul],['xl0','ul0','Vxl0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['Qul_f'])
        # hessians of the system dynamics, the cost function, and the Q value function
        self.FxlVxl   = self.Fxl.T@self.Vxl
        self.dFxlVxldxl= jacobian(self.FxlVxl,self.xl) # the hessian of the system dynamics may cause heavy computational burden
        self.dFxlVxldul= jacobian(self.FxlVxl,self.ul)
        self.FulVxl   = self.Ful.T@self.Vxl
        self.dFulVxldul= jacobian(self.FulVxl,self.ul)
        self.lxlxl    = jacobian(self.lxl,self.xl)
        self.lxlxlN   = jacobian(self.lxlN,self.xl)
        self.lxlxlN_fn= Function('lxlxlN',[self.para_l,self.a],[self.lxlxlN],['paral0','a0'],['lxlxlN_f'])
        self.lxlul    = jacobian(self.lxl,self.ul)
        self.lulul    = jacobian(self.lul,self.ul)
        self.Qxlxl_bar    = self.lxlxl #+ alpha*self.dFxlVxldxl  # removing this model hessian can enhance the DDP stability for a larger time step!!!! The removal can also accelerate the DDP computation significantly!
        self.Qxlxl_bar_fn = Function('Qxlxl_bar',[self.xl,self.ul,self.Vxl,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.Qxlxl_bar],['xl0','ul0','Vxl0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['Qxlxl_bar_f'])
        self.Qxlxl_hat    = self.Fxl.T@self.Vxlxl@self.Fxl
        self.Qxlxl_hat_fn = Function('Qxlxl_hat',[self.xl,self.ul,self.Vxlxl],[self.Qxlxl_hat],['xl0','ul0','Vxlxl0'],['Qxlxl_hat_f'])
        self.Qxlul_bar    = self.lxlul #+ alpha*self.dFxlVxldul  # including the model hessian entails a very small time step size (e.g., 0.01s)
        self.Qxlul_bar_fn = Function('Qxlul_bar',[self.xl,self.ul,self.Vxl,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.Qxlul_bar],['xl0','ul0','Vxl0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['Qxlul_bar_f'])
        self.Qxlul_hat    = self.Fxl.T@self.Vxlxl@self.Ful
        self.Qxlul_hat_fn = Function('Qxlul_hat',[self.xl,self.ul,self.Vxlxl],[self.Qxlul_hat],['xl0','ul0','Vxlxl0'],['Qxlul_hat_f'])
        self.Qulul_bar    = self.lulul #+ alpha*self.dFulVxldul
        self.Qulul_bar_fn = Function('Qulul_bar',[self.xl,self.ul,self.Vxl,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.Qulul_bar],['xl0','ul0','Vxl0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['Qulul_bar_f'])
        self.Qulul_hat    = self.Ful.T@self.Vxlxl@self.Ful
        self.Qulul_hat_fn = Function('Qulul_hat',[self.xl,self.ul,self.Vxlxl],[self.Qulul_hat],['xl0','ul0','Vxlxl0'],['Qulul_hat_f'])
        # hessians w.r.t. the hyperparameters
        self.lxlp     = jacobian(self.lxl,self.P_auto)
        self.lxlp_fn  = Function('lxlp',[self.xl,self.ul,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.lxlp],['xl0','ul0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['lxlp_f'])
        self.lulp     = jacobian(self.lul,self.P_auto)
        self.lulp_fn  = Function('lulp',[self.xl,self.ul,self.ref_xl,self.ref_ul,self.scxl,self.scxL,self.scul,self.scuL,self.para_l,self.a],[self.lulp],['xl0','ul0','refxl0','reful0','scxl0','scxL0','scul0','scuL0','paral0','a0'],['lulp_f'])
        self.lxlNp    = jacobian(self.lxlN,self.P_auto)
        self.lxlNp_fn = Function('lxlNp',[self.xl,self.ref_xl,self.scxl,self.scxL,self.para_l,self.a],[self.lxlNp],['xl0', 'refxl0', 'scxl0', 'scxL0', 'paral0','a0'],['lxlNp_f'])


    
    def Cable_derivatives_DDP_ADMM(self):
        self.Vxi      = SX.sym('Vxi',self.nxi)
        self.Vxixi    = SX.sym('Vxixi',self.nxi,self.nxi)
        # gradients of the system dynamics, the cost function, and the Q value function
        self.Fxi      = jacobian(self.model_i,self.xi)
        self.Fxi_fn   = Function('Fxi',[self.xi,self.ui],[self.Fxi],['xi0','ui0'],['Fxi_f'])
        self.Fui      = jacobian(self.model_i,self.ui)
        self.Fui_fn   = Function('Fui',[self.xi,self.ui],[self.Fui],['xi0','ui0'],['Fui_f'])
        self.lxi      = jacobian(self.Ji_k,self.xi)
        self.lxiN     = jacobian(self.Ji_N,self.xi)
        self.lxiN_fn  = Function('lxiN',[self.xi,self.ref_xi,self.scxi,self.scxI,self.para_i,self.a],[self.lxiN],['xi0', 'refxi0', 'scxi0', 'scxI0', 'parai0','a0'],['lxiN_f'])
        self.lui      = jacobian(self.Ji_k,self.ui)
        self.Qxi      = self.lxi.T + self.Fxi.T@self.Vxi
        self.Qxi_fn   = Function('Qxi',[self.xi,self.ui,self.Vxi,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.Qxi],['xi0','ui0','Vxi0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['Qxi_f'])
        self.Qui      = self.lui.T + self.Fui.T@self.Vxi
        self.Qui_fn   = Function('Qui',[self.xi,self.ui,self.Vxi,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.Qui],['xi0','ui0','Vxi0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['Qui_f'])
        # hessians of the system dynamics, the cost function, and the Q value function
        self.FxiVxi   = self.Fxi.T@self.Vxi
        self.dFxiVxidxi= jacobian(self.FxiVxi,self.xi) # the hessian of the system dynamics may cause heavy computational burden
        self.dFxiVxidui= jacobian(self.FxiVxi,self.ui)
        self.FuiVxi   = self.Fui.T@self.Vxi
        self.dFuiVxidui= jacobian(self.FuiVxi,self.ui)
        self.lxixi    = jacobian(self.lxi,self.xi) # already includes pi
        self.lxixiN   = jacobian(self.lxiN,self.xi)
        self.lxixiN_fn= Function('lxixiN',[self.para_i,self.a],[self.lxixiN],['parai0','a0'],['lxixiN_f'])
        self.lxiui    = jacobian(self.lxi,self.ui)
        self.luiui    = jacobian(self.lui,self.ui)
        self.Qxixi_bar    = self.lxixi #+ alpha*self.dFxiVxidxi  # removing this model hessian can enhance the DDP stability for a larger time step!!!! The removal can also accelerate the DDP computation significantly!
        self.Qxixi_bar_fn = Function('Qxixi_bar',[self.xi,self.ui,self.Vxi,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.Qxixi_bar],['xi0','ui0','Vxi0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['Qxixi_bar_f'])
        self.Qxixi_hat    = self.Fxi.T@self.Vxixi@self.Fxi
        self.Qxixi_hat_fn = Function('Qxixi_hat',[self.xi,self.ui,self.Vxixi],[self.Qxixi_hat],['xi0','ui0','Vxixi0'],['Qxixi_hat_f'])
        self.Qxiui_bar    = self.lxiui #+ alpha*self.dFxiVxidui  # including the model hessian entails a very small time step size (e.g., 0.01s)
        self.Qxiui_bar_fn = Function('Qxiui_bar',[self.xi,self.ui,self.Vxi,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.Qxiui_bar],['xi0','ui0','Vxi0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['Qxiui_bar_f'])
        self.Qxiui_hat    = self.Fxi.T@self.Vxixi@self.Fui
        self.Qxiui_hat_fn = Function('Qxiui_hat',[self.xi,self.ui,self.Vxixi],[self.Qxiui_hat],['xi0','ui0','Vxixi0'],['Qxiui_hat_f'])
        self.Quiui_bar    = self.luiui #+ alpha*self.dFuiVxidui
        self.Quiui_bar_fn = Function('Quiui_bar',[self.xi,self.ui,self.Vxi,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.Quiui_bar],['xi0','ui0','Vxi0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['Quiui_bar_f'])
        self.Quiui_hat    = self.Fui.T@self.Vxixi@self.Fui
        self.Quiui_hat_fn = Function('Quiui_hat',[self.xi,self.ui,self.Vxixi],[self.Quiui_hat],['xi0','ui0','Vxixi0'],['Quiui_hat_f'])
        # hessians w.r.t. the hyperparameters
        self.lxip     = jacobian(self.lxi,self.P_auto)
        self.lxip_fn  = Function('lxip',[self.xi,self.ui,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.lxip],['xi0','ui0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['lxip_f'])
        self.luip     = jacobian(self.lui,self.P_auto)
        self.luip_fn  = Function('luip',[self.xi,self.ui,self.ref_xi,self.ref_ui,self.scxi,self.scxI,self.scui,self.scuI,self.para_i,self.a],[self.luip],['xi0','ui0','refxi0','refui0','scxi0','scxI0','scui0','scuI0','parai0','a0'],['luip_f'])
        self.lxiNp    = jacobian(self.lxiN,self.P_auto)
        self.lxiNp_fn = Function('lxiNp',[self.xi,self.ref_xi,self.scxi,self.scxI,self.para_i,self.a],[self.lxiNp],['xi0', 'refxi0', 'scxi0', 'scxI0', 'parai0','a0'],['lxiNp_f'])


    
    
    def Get_AuxSys_DDP_Load(self,opt_sol,Ref_xl,Ref_ul,scxl,scul,scxL,scuL,weight1,i_admm):
        xl_opt   = opt_sol['xl_traj']
        ul_opt   = opt_sol['ul_traj']
        LxlNp    = self.lxlNp_fn(xl0=xl_opt[-1,:],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl],paral0=weight1,a0=i_admm)['lxlNp_f'].full()
        LxlxlN   = self.lxlxlN_fn(paral0=weight1,a0=i_admm)['lxlxlN_f'].full()
        Lxlp     = self.N*[np.zeros((self.nxl,self.n_Pauto))]
        Lulp     = self.N*[np.zeros((self.nul,self.n_Pauto))]
        for k in range(self.N):
            Lxlp[k] = self.lxlp_fn(xl0=xl_opt[k,:],ul0=ul_opt[k,:],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                scxl0=scxl[k,:],scxL0=scxL[k,:],scul0=scul[k,:],scuL0=scuL[k,:],paral0=weight1,a0=i_admm)['lxlp_f'].full()
            Lulp[k] = self.lulp_fn(xl0=xl_opt[k,:],ul0=ul_opt[k,:],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                scxl0=scxl[k,:],scxL0=scxL[k,:],scul0=scul[k,:],scuL0=scuL[k,:],paral0=weight1,a0=i_admm)['lulp_f'].full()
        
        auxSysl = { "HxxN":LxlxlN,
                    "HxNp":LxlNp,
                    "Hxp":Lxlp,
                    "Hup":Lulp
                    }
        
        return auxSysl
    

    def Get_AuxSys_DDP_Cable(self,opt_sol,Ref_xi,Ref_ui,scxi,scui,scxI,scuI,weight2,i_admm):
        xi_opt   = opt_sol['xi_traj']
        ui_opt   = opt_sol['ui_traj']
        LxiNp    = self.lxiNp_fn(xi0=xi_opt[-1,:],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi],parai0=weight2,a0=i_admm)['lxiNp_f'].full()
        LxixiN   = self.lxixiN_fn(parai0=weight2,a0=i_admm)['lxixiN_f'].full()
        Lxip     = self.N*[np.zeros((self.nxi,self.n_Pauto))]
        Luip     = self.N*[np.zeros((self.nui,self.n_Pauto))]
        for k in range(self.N):
            Lxip[k] = self.lxip_fn(xi0=xi_opt[k,:],ui0=ui_opt[k,:],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui[k*self.nui:(k+1)*self.nui],
                                scxi0=scxi[k,:],scxI0=scxI[k,:],scui0=scui[k,:],scuI0=scuI[k,:],parai0=weight2,a0=i_admm)['lxip_f'].full()
            Luip[k] = self.luip_fn(xi0=xi_opt[k,:],ui0=ui_opt[k,:],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui[k*self.nui:(k+1)*self.nui],
                                scxi0=scxi[k,:],scxI0=scxI[k,:],scui0=scui[k,:],scuI0=scuI[k,:],parai0=weight2,a0=i_admm)['luip_f'].full()
        
        auxSysi = { "HxxN":LxixiN,
                    "HxNp":LxiNp,
                    "Hxp":Lxip,
                    "Hup":Luip
                    }
        
        return auxSysi
    

    def symmetry(self,A):
        return 0.5*(A + A.T)

    def chol_solve(self,L, B):
        # Solve (L L^T) X = B
        Y = LA.solve(L, B)
        return LA.solve(L.T, Y)

    def try_cholesky(self,A, jitter0=0.0, max_tries=5):
        """Try Cholesky with growing jitter on the diagonal."""
        jitter = jitter0
        for _ in range(max_tries):
            try:
                return LA.cholesky(A + jitter*np.eye(A.shape[0])), jitter
            except LA.LinAlgError:
                jitter = max(1e-12, 10*(jitter if jitter>0 else 1e-12))
        raise LA.LinAlgError("Cholesky failed even with jitter")

   
    def DDP_Load_ADMM_Subp1(self,xl_0,Ref_xl,Ref_ul,weight1,scxl,scul,scxL,scuL,max_iter,e_tol,i,i_admm):
        reg        = 1e-6 # Regularization term
        reg_max    = 1    # cap to avoid runaway
        reg_up     = 10.0 # how much to bump when ill-conditioned
        alpha_init = 1 # Initial alpha for line search
        alpha_min  = 1e-2  # Minimum allowable alpha
        alpha_factor = 0.5 # 
        max_line_search_steps = 5
        iteration = 1
        ratio = 10
        X_nominal = np.zeros((self.nxl,self.N+1))
        U_nominal = np.zeros((self.nul,self.N))
        X_nominal[:,0:1] = np.reshape(xl_0,(self.nxl,1))
        
        # Initial trajectory and initial cost 
        cost_prev = 0
        if i ==0:
            for k in range(self.N):
                u_k    = np.reshape(Ref_ul[k*self.nul:(k+1)*self.nul],(self.nul,1))
                X_nominal[:,k+1:k+2] = self.model_l_fn(xl0=X_nominal[:,k],ul0=u_k)['mdynlf'].full()
                U_nominal[:,k:k+1]   = u_k
                cost_prev     += self.Jl_kfn(xl0=X_nominal[:,k],ul0=u_k,scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],
                                        scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],
                                        refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Jl_kf'].full()
            cost_prev += self.Jl_Nfn(xl0=X_nominal[:,-1],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],
                                 scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl],paral0=weight1,a0=i_admm)['Jl_Nf'].full()
        else:
            for k in range(self.N):
                X_nominal[:,k+1:k+2] = np.reshape(scxl[k*self.nxl:(k+1)*self.nxl],(self.nxl,1))
                U_nominal[:,k:k+1]   = np.reshape(scul[k*self.nul:(k+1)*self.nul],(self.nul,1))
                cost_prev     += self.Jl_kfn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],
                                        scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],
                                        refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Jl_kf'].full()
            cost_prev += self.Jl_Nfn(xl0=X_nominal[:,-1],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],
                                 scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl],paral0=weight1,a0=i_admm)['Jl_Nf'].full()    

        Qxx_bar     = self.N*[np.zeros((self.nxl,self.nxl))]
        Qxu_bar     = self.N*[np.zeros((self.nxl,self.nul))]
        Quu_bar     = self.N*[np.zeros((self.nul,self.nul))]
        Qxu         = self.N*[np.zeros((self.nxl,self.nul))]
        Quuinv      = self.N*[np.zeros((self.nul,self.nul))]
        Fx          = self.N*[np.zeros((self.nxl,self.nxl))]
        Fu          = self.N*[np.zeros((self.nxl,self.nul))]
        Vx          = (self.N+1)*[np.zeros((self.nxl,1))]
        Vxx         = (self.N+1)*[np.zeros((self.nxl,self.nxl))]
        K_fb        = self.N*[np.zeros((self.nul,self.nxl))] # feedback
        k_ff        = self.N*[np.zeros((self.nul,1))] # feedforward
        Qu_2        = 1000
        I_u         = np.identity(self.nul)
        while Qu_2>e_tol and iteration<=max_iter:
            Vx[self.N] = self.lxlN_fn(xl0=X_nominal[:,self.N],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],
                                      scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],paral0=weight1,a0=i_admm)['lxlN_f'].full()
            Vxx[self.N]= self.lxlxlN_fn(paral0=weight1,a0=i_admm)['lxlxlN_f'].full()
            # backward pass
            Qu_2    = 0
            chol_failed = False
            for k in reversed(range(self.N)): # N-1, N-2,...,0
                Qx_k  = self.Qxl_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxl0=Vx[k+1],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                    scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Qxl_f'].full()
                Qu_k  = self.Qul_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxl0=Vx[k+1],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                    scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Qul_f'].full()
                Qxx_bar_k = self.Qxlxl_bar_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxl0=Vx[k+1],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                    scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Qxlxl_bar_f'].full()
                Qxx_hat_k = self.Qxlxl_hat_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxlxl0=Vxx[k+1])['Qxlxl_hat_f'].full()
                Qxx_k     = Qxx_bar_k + Qxx_hat_k
                Qxu_bar_k = self.Qxlul_bar_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxl0=Vx[k+1],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                    scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Qxlul_bar_f'].full()
                Qxu_hat_k = self.Qxlul_hat_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxlxl0=Vxx[k+1])['Qxlul_hat_f'].full()
                Qxu_k     = Qxu_bar_k + Qxu_hat_k
                Quu_bar_k = self.Qulul_bar_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxl0=Vx[k+1],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],
                                    scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Qulul_bar_f'].full()
                Quu_hat_k = self.Qulul_hat_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],Vxlxl0=Vxx[k+1])['Qulul_hat_f'].full()
                Quu_k     = Quu_bar_k + Quu_hat_k 
                Quu_reg_k = Quu_k + reg*I_u
                L, _jitter   = self.try_cholesky(Quu_reg_k, jitter0=0.0)
                if L is None:
                    chol_failed = True
                    break
                Quu_inv      = self.chol_solve(L, I_u) # only for computing the gradients
                K_fb[k]      = self.chol_solve(L, -Qxu_k.T)
                k_ff[k]      = self.chol_solve(L, -Qu_k)
                Vx[k]        = Qx_k + Qxu_k @ k_ff[k]
                Vxx[k]       = self.symmetry(Qxx_k + Qxu_k @ K_fb[k])
                Fx[k]    = self.Fxl_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k])['Fxl_f'].full()
                Fu[k]    = self.Ful_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k])['Ful_f'].full()
                Qxx_bar[k]   = Qxx_bar_k
                Qxu_bar[k]   = Qxu_bar_k
                Quu_bar[k]   = Quu_bar_k
                Quuinv[k]    = Quu_inv
                Qxu[k]       = Qxu_k
                Qu_2         = max(Qu_2, (LA.norm(Qu_k)))
            # if backward failed, bump reg and retry (do NOT advance iteration)
            if chol_failed:
                reg = min(reg_max, reg * reg_up)
                # print(f'backward cholesky failed → increasing reg to {reg:.3e}')
                continue
            # forward pass with adaptive alpha (line search), adaptive alpha makes the DDP more stable!
            alpha = alpha_init
            accepted = False
            for _ in range(max_line_search_steps):
                X_new = np.zeros((self.nxl,self.N+1))
                U_new = np.zeros((self.nul,self.N))
                X_new[:,0:1] = np.reshape(xl_0,(self.nxl,1))
                cost_new = 0
                for k in range(self.N):
                    delta_x = np.reshape(X_new[:,k] - X_nominal[:,k],(self.nxl,1))
                    u_k     = np.reshape(U_nominal[:,k],(self.nul,1)) + K_fb[k]@delta_x + alpha*k_ff[k]
                    u_k     = np.reshape(u_k,(self.nul,1))
                    X_new[:,k+1:k+2]  = self.model_l_fn(xl0=np.reshape(X_new[:,k],(self.nxl,1)),ul0=u_k)['mdynlf'].full()
                    U_new[:,k:k+1]    = u_k
                    cost_new   += self.Jl_kfn(xl0=X_new[:,k],ul0=u_k,scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],
                                              scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],
                                              refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Jl_kf'].full()
                cost_new   += self.Jl_Nfn(xl0=X_new[:,-1],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],
                                          scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl], paral0=weight1,a0=i_admm)['Jl_Nf'].full()
                # Check if the cost decreased
                if cost_new < cost_prev:
                    # update the trajectories
                    X_nominal = X_new
                    U_nominal = U_new
                    accepted  = True
                    break
                alpha = np.clip(alpha*alpha_factor,alpha_min,alpha_init)  # Reduce alpha if cost did not improve

            # if nothing accepted, nudge reg up to help next backward factorization
            if not accepted:
                reg = min(reg_max, reg * reg_up)

            ratio = np.abs(cost_new-cost_prev)/np.abs(cost_prev)
            print('iteration:',iteration,'ratio=',ratio,'Qu_2=',Qu_2)

            cost_prev = cost_new
            iteration += 1
        
        opt_sol={"xl_traj":X_nominal.T,
                 "ul_traj":U_nominal.T,
                 "Vxx":Vxx,
                 "Vx":Vx,
                 "K_FB":K_fb,
                 "Hxx":Qxx_bar,
                 "Qxu":Qxu,
                 "Hxu":Qxu_bar,
                 "Huu":Quu_bar,
                 "Quu_inv":Quuinv,
                 "Fx":Fx,
                 "Fu":Fu}
        return opt_sol
    

    def DDP_Cable_ADMM_Subp1(self,xi_0,Ref_xi,Ref_ui,weight2,scxi,scui,scxI,scuI,max_iter,e_tol,i,i_admm):
        reg          = 1e-6 # Regularization term
        reg_max      = 1    # cap to avoid runaway
        reg_up       = 10.0 # how much to bump when ill-conditioned
        alpha_init   = 1 # Initial alpha for line search
        alpha_min    = 1e-2  # Minimum allowable alpha
        alpha_factor = 0.5 # 
        max_line_search_steps = 5
        iteration = 1
        ratio = 10
        X_nominal = np.zeros((self.nxi,self.N+1))
        U_nominal = np.zeros((self.nui,self.N))
        X_nominal[:,0:1] = np.reshape(xi_0,(self.nxi,1))
        
        # Initial trajectory and initial cost 
        cost_prev = 0
        if i ==0:
            for k in range(self.N):
                u_k    = np.reshape(Ref_ui,(self.nui,1))
                X_nominal[:,k+1:k+2] = self.model_i_fn(xi0=X_nominal[:,k],ui0=u_k)['mdynif'].full()
                U_nominal[:,k:k+1]   = u_k
                cost_prev     += self.Ji_k_fn(xi0=X_nominal[:,k],ui0=u_k,scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],
                                        scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],
                                        refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,parai0=weight2,a0=i_admm)['Ji_kf'].full()
            cost_prev += self.Ji_N_fn(xi0=X_nominal[:,-1],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
                                 scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi],parai0=weight2,a0=i_admm)['Ji_Nf'].full()
        else:
            for k in range(self.N):
                X_nominal[:,k+1:k+2] = np.reshape(scxi[k*self.nxi:(k+1)*self.nxi],(self.nxi,1))
                U_nominal[:,k:k+1]   = np.reshape(scui[k*self.nui:(k+1)*self.nui],(self.nui,1))
                cost_prev     += self.Ji_k_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],
                                        scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],
                                        refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,parai0=weight2,a0=i_admm)['Ji_kf'].full()
            cost_prev += self.Ji_N_fn(xi0=X_nominal[:,-1],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
                                 scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi],parai0=weight2,a0=i_admm)['Ji_Nf'].full()

        Qxx_bar     = self.N*[np.zeros((self.nxi,self.nxi))]
        Qxu_bar     = self.N*[np.zeros((self.nxi,self.nui))]
        Quu_bar     = self.N*[np.zeros((self.nui,self.nui))]
        Qxu         = self.N*[np.zeros((self.nxi,self.nui))]
        Quuinv      = self.N*[np.zeros((self.nui,self.nui))]
        Fx          = self.N*[np.zeros((self.nxi,self.nxi))]
        Fu          = self.N*[np.zeros((self.nxi,self.nui))]
        Vx          = (self.N+1)*[np.zeros((self.nxi,1))]
        Vxx         = (self.N+1)*[np.zeros((self.nxi,self.nxi))]
        K_fb        = self.N*[np.zeros((self.nui,self.nxi))] # feedback
        k_ff        = self.N*[np.zeros((self.nui,1))] # feedforward
        Qu_2        = 1000
        I_u         = np.identity(self.nui)

        while Qu_2>e_tol and iteration<=max_iter:
            Vx[self.N] = self.lxiN_fn(xi0=X_nominal[:,self.N],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],
                                      scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],parai0=weight2,a0=i_admm)['lxiN_f'].full()
            Vxx[self.N]= self.lxixiN_fn(parai0=weight2,a0=i_admm)['lxixiN_f'].full() 
            # backward pass
            Qu_2    = 0
            chol_failed = False
            for k in reversed(range(self.N)): # N-1, N-2,...,0
                Qx_k  = self.Qxi_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxi0=Vx[k+1],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,
                                    scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],parai0=weight2,a0=i_admm)['Qxi_f'].full()
                Qu_k  = self.Qui_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxi0=Vx[k+1],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,
                                    scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],parai0=weight2,a0=i_admm)['Qui_f'].full()
                Qxx_bar_k = self.Qxixi_bar_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxi0=Vx[k+1],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,
                                    scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],parai0=weight2,a0=i_admm)['Qxixi_bar_f'].full()
                Qxx_hat_k = self.Qxixi_hat_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxixi0=Vxx[k+1])['Qxixi_hat_f'].full()
                Qxx_k     = Qxx_bar_k + Qxx_hat_k
                Qxu_bar_k = self.Qxiui_bar_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxi0=Vx[k+1],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,
                                    scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],parai0=weight2,a0=i_admm)['Qxiui_bar_f'].full()
                Qxu_hat_k = self.Qxiui_hat_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxixi0=Vxx[k+1])['Qxiui_hat_f'].full()
                Qxu_k     = Qxu_bar_k + Qxu_hat_k
                Quu_bar_k = self.Quiui_bar_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxi0=Vx[k+1],refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,
                                    scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],parai0=weight2,a0=i_admm)['Quiui_bar_f'].full()
                Quu_hat_k = self.Quiui_hat_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],Vxixi0=Vxx[k+1])['Quiui_hat_f'].full()
                Quu_k     = Quu_bar_k + Quu_hat_k
                Quu_reg_k = Quu_k + reg*I_u
                L, _jitter   = self.try_cholesky(Quu_reg_k, jitter0=0.0)
                if L is None:
                    chol_failed = True
                    break
                Quu_inv      = self.chol_solve(L, I_u) # only for computing the gradients
                K_fb[k]      = self.chol_solve(L, -Qxu_k.T)
                k_ff[k]      = self.chol_solve(L, -Qu_k)
                Vx[k]        = Qx_k + Qxu_k @ k_ff[k]
                Vxx[k]       = self.symmetry(Qxx_k + Qxu_k @ K_fb[k])
                Fx[k]    = self.Fxi_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k])['Fxi_f'].full()
                Fu[k]    = self.Fui_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k])['Fui_f'].full()
                Qxx_bar[k]   = Qxx_bar_k
                Qxu_bar[k]   = Qxu_bar_k
                Quu_bar[k]   = Quu_bar_k
                Quuinv[k]    = Quu_inv
                Qxu[k]       = Qxu_k
                Qu_2         = max(Qu_2, (LA.norm(Qu_k)))
            # if backward failed, bump reg and retry (do NOT advance iteration)
            if chol_failed:
                reg = min(reg_max, reg * reg_up)
                # print(f'backward cholesky failed → increasing reg to {reg:.3e}')
                continue
            # forward pass with adaptive alpha (line search), adaptive alpha makes the DDP more stable!
            alpha = alpha_init
            accepted = False
            for _ in range(max_line_search_steps):
                X_new = np.zeros((self.nxi,self.N+1))
                U_new = np.zeros((self.nui,self.N))
                X_new[:,0:1] = np.reshape(xi_0,(self.nxi,1))
                cost_new = 0
                for k in range(self.N):
                    delta_x = np.reshape(X_new[:,k] - X_nominal[:,k],(self.nxi,1))
                    u_k     = np.reshape(U_nominal[:,k],(self.nui,1)) + K_fb[k]@delta_x + alpha*k_ff[k]
                    u_k     = np.reshape(u_k,(self.nui,1))
                    X_new[:,k+1:k+2]  = self.model_i_fn(xi0=np.reshape(X_new[:,k],(self.nxi,1)),ui0=u_k)['mdynif'].full()
                    U_new[:,k:k+1]    = u_k
                    cost_new   += self.Ji_k_fn(xi0=X_new[:,k],ui0=u_k,scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],
                                              scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],
                                              refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,parai0=weight2)['Ji_kf'].full()
                cost_new   += self.Ji_N_fn(xi0=X_new[:,-1],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
                                          scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi], parai0=weight2)['Ji_Nf'].full()
                # Check if the cost decreased
                if cost_new < cost_prev:
                    # update the trajectories
                    X_nominal = X_new
                    U_nominal = U_new
                    accepted  = True
                    break
                alpha = np.clip(alpha*alpha_factor,alpha_min,alpha_init)  # Reduce alpha if cost did not improve

            # if nothing accepted, nudge reg up to help next backward factorization
            if not accepted:
                reg = min(reg_max, reg * reg_up)

            ratio = np.abs(cost_new-cost_prev)/np.abs(cost_prev)
            print('iteration:',iteration,'ratio=',ratio,'Qu_2=',Qu_2)

            cost_prev = cost_new
            iteration += 1
        
        opt_sol={"xi_traj":X_nominal.T,
                 "ui_traj":U_nominal.T,
                 "Vxx":Vxx,
                 "Vx":Vx,
                 "K_FB":K_fb,
                 "Hxx":Qxx_bar,
                 "Qxu":Qxu,
                 "Hxu":Qxu_bar,
                 "Huu":Quu_bar,
                 "Quu_inv":Quuinv,
                 "Fx":Fx,
                 "Fu":Fu}
        return opt_sol


    def MPC_Cable_DDP_Planning_SubP1(self,ParaC): # checked, correct, Apr.1 2025
        xc_traj      = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        uc_traj      = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))]
        OPt_sol_c    = []
        max_iter     = 10
        e_tol        = 1e-2
        for i in range(int(self.nq)):
            Parai    = ParaC[i]
            xi_fb    = Parai[0:self.nxi]
            Ref_xi   = Parai[self.nxi:self.nxi*(self.N+2)]
            Ref_ui   = Parai[self.nxi+self.nxi*(self.N+1):self.nxi+self.nxi*(self.N+1)+self.nui]
            # Solve the DDP
            n_scxi_start = self.nxi*(self.N+2)+self.nui
            scxi         = Parai[n_scxi_start:n_scxi_start+self.nxi*(self.N+1)]
            n_scxI_start = n_scxi_start + self.nxi*(self.N+1)
            scxI         = Parai[n_scxI_start:n_scxI_start+self.nxi*(self.N+1)]
            n_scui_start = n_scxI_start + self.nxi*(self.N+1)
            scui         = Parai[n_scui_start:n_scui_start+self.nui*self.N]
            n_scuI_start = n_scui_start + self.nui*self.N
            scuI         = Parai[n_scuI_start:n_scuI_start+self.nui*self.N]
            n_weig_start = n_scuI_start + self.nui*self.N
            weight2   = Parai[n_weig_start:n_weig_start+self.npi]
            Iadmm     = Parai[-2]
            i_admm    = Parai[-1]
            opt_sol_i = self.DDP_Cable_ADMM_Subp1(xi_fb,Ref_xi,Ref_ui,weight2,scxi,scui,scxI,scuI,max_iter,e_tol,Iadmm,i_admm)
            OPt_sol_c += [opt_sol_i]
            xc_traj[i] = opt_sol_i['xi_traj']
            uc_traj[i] = opt_sol_i['ui_traj']
        # output
        opt_solc = {"xc_traj":xc_traj,
                   "uc_traj":uc_traj
                   }
        
        return opt_solc, OPt_sol_c

    
    def SetConstriants(self, pob1, pob2):
        # dynamic coupling constraint at each step k
        pl_k     = self.scxl[0:3]
        vl_k     = self.scxl[3:6]
        ql_k     = self.scxl[6:10]
        wl_k     = self.scxl[10:self.nxl]
        Fl_k     = self.scul[0:3] #{I}
        Ml_k     = self.scul[3:6]
        tc_k     = SX.sym('tc_k',3*int(self.nq),1)
        Rl_k     = self.q_2_rotation(ql_k)
        ql_knorm = ql_k.T@ql_k
        self.ql_n     = 1/(2*self.p_bar)*(ql_knorm-1)**2
        self.ql_fn    = Function('norm_ql',[self.scxl],[ql_knorm],['scxl0'],['norm_qlf'])
        k        = 0
        self.fi    = [] # list that stores all the quadrotor thruster limit constraints
        self.vi    = [] # list that stores all the quadrotor velocity limit constraints
        self.Gi1   = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 1st obstacle
        self.Gi2   = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 2nd obstacle
        self.Gij   = [] # list that stores all the safe inter-robot inequality constraints
        self.Di    = []
        self.sumfi = 0 # barrier functions of the quadrotor thrust limit
        self.sumvi = 0 # barrier functions of the quadrotor velocity limit
        self.gco   = 0 # barrier functions of the safe collision-avoidance constraints on quadrotors' planar positions
        self.gij   = 0 # barrier functions of the safe inter-robot constraints on quadrotors' planar positions
        self.Tcon  = 0 # barrier functions of the tension magnitude constraints
        self.din   = 0 # barrier functions of the cable direction normalization 
        for i in range(int(self.nq)):
            xi_k   = self.scxc[i*self.nxi:(i+1)*self.nxi]
            ui_k   = self.scuc[i*self.nui:(i+1)*self.nui]
            di_k   = xi_k[0:3] # world frame
            wi_k   = xi_k[3:6]
            ti_k   = xi_k[6]
            self.Tcon += -self.p_bar * log(ti_k-self.t_min)
            self.Tcon += -self.p_bar * log(self.t_max-ti_k)
            dwi_k  = ui_k[0:3] # cable angular acceleration
            ri   = np.reshape(self.ra[:,i],(3,1))
            pi_k = pl_k + Rl_k@ri + self.cl0*di_k # ith quadrotor's position in {I}
            diso1 = pi_k[0:2]-pob1
            go1  = diso1.T@diso1 - ((self.rq + self.ro)+self.rq/2)**2 # safe constriant between the obstacle 1 and the ith quadrotor, which should be positive. 0.1 is the safety margin.
            go1_fn = Function('go1'+str(i),[self.scxl,self.scxc],[go1],['scxl0','scxc0'],['go1f'+str(i)])
            self.gco += -self.p_bar * log(go1)
            self.Gi1 += [go1_fn]
            diso2 = pi_k[0:2]-pob2
            go2  = diso2.T@diso2 - ((self.rq + self.ro)+self.rq/2)**2 # safe constriant between the obstacle 2 and the ith quadrotor, which should be positive
            go2_fn = Function('go2'+str(i),[self.scxl,self.scxc],[go2],['scxl0','scxc0'],['go2f'+str(i)])
            self.gco += -self.p_bar * log(go2)
            self.Gi2 += [go2_fn]
            dnorm = di_k.T@di_k
            self.din += 1/(2*self.p_bar)*(dnorm-1)**2
            d_fn  = Function('dn'+str(i),[self.scxc],[dnorm],['scxc0'],['dn'+str(i)])
            self.Di += [d_fn]
            # Thrust constraints
            S_wl_k  = self.skew_sym(wl_k)
            S_wi_k  = self.skew_sym(wi_k)
            S_dwi_k = self.skew_sym(dwi_k)
            al_k    = -self.g*self.ez + Fl_k/self.ml
            awl_k   = LA.inv(self.Jl)@(Ml_k-S_wl_k@(self.Jlcom@wl_k))
            S_awl_k = self.skew_sym(awl_k)
            fi_k    = self.mq*(al_k+Rl_k@(S_wl_k@S_wl_k+S_awl_k)@ri+self.cl0*(S_dwi_k@di_k+S_wi_k@(S_wi_k@di_k))+self.g*self.ez) + di_k*ti_k
            norm_fi = fi_k.T@fi_k
            self.sumfi += -self.p_bar * log(self.fmax**2-norm_fi)
            norm_fi_fn = Function('norm_f'+str(i),[self.scxl,self.scul,self.scxc,self.scuc],[norm_fi],['scxl0','scul0','scxc0','scuc0'],['norm_ff'+str(i)])
            self.fi += [norm_fi_fn]
            # Velocity constraints: adding this leads to incorrect gradients!
            # vi_k    = vl_k + Rl_k@S_wl_k@ri + self.cl0*(S_wi_k@di_k)
            # norm_vi = vi_k.T@vi_k
            # self.sumvi += -self.p_bar * log(self.vmax**2-norm_vi)
            # norm_vi_fn = Function('norm_v'+str(i),[self.scxl,self.scxc],[norm_vi],['scxl0','scxc0'],['norm_vf'+str(i)])
            # self.vi += [norm_vi_fn]
            for j in range(i+1,int(self.nq)): # safe inter-robot separation constraints
                pi_I   = Rl_k@ri + self.cl0*di_k # world frame
                xj_k   = self.scxc[j*self.nxi:(j+1)*self.nxi]
                dj_k   = xj_k[0:3]
                rj     = np.reshape(self.ra[:,j],(3,1))
                pj_I   = Rl_k@rj + self.cl0*dj_k
                disij  = pi_I[0:2]-pj_I[0:2]
                gij    = disij.T@disij - (4*self.rq)**2
                self.gij += -self.p_bar * log(gij)
                gij_fn = Function('g'+str(k),[self.scxl,self.scxc],[gij],['scxl0','scxc0'],['gf'+str(k)])
                self.Gij += [gij_fn]
                k     += 1
            ti_kb = Rl_k.T@di_k*ti_k # cable tension vector in {B}
            tc_k[i*3:(i+1)*3] = ti_kb
        # control consensus constraint that maps tension forces to the load control wrench
        wrench   = vertcat(Rl_k.T@Fl_k,Ml_k) # body frame
        W_cons   = self.Pt@tc_k - wrench
        self.h_wcons  = 1/(2*self.p_bar)*W_cons.T@W_cons
        self.W_cons_fn = Function('W_cons',[self.scxl,self.scul,self.scxc],[W_cons],['scxl0','scul0','scxc0'],['W_consf'])


    def SetADMMSubP2_SoftCost_k(self):
        # at each step k
        self.J_2_soft_k    = self.Jl_P2_k + self.gco + self.gij + self.Tcon + self.ql_n + self.din + self.sumfi + self.h_wcons
        for i in range(int(self.nq)):
            xi      = self.xc[i*self.nxi:(i+1)*self.nxi]   # cable primal state
            scxi    = self.scxc[i*self.nxi:(i+1)*self.nxi] # safe copy state of each cable
            scxI    = self.scxC[i*self.nxi:(i+1)*self.nxi] # Lagrangian multiplier
            ui      = self.uc[i*self.nui:(i+1)*self.nui]   # cable primal control
            scui    = self.scuc[i*self.nui:(i+1)*self.nui] # safe copy control of each cable
            scuI    = self.scuC[i*self.nui:(i+1)*self.nui] # Lagrangian multiplier
            resid_x = xi - scxi + scxI/self.pi
            resid_u = ui - scui + scuI/self.pi
            self.J_2_soft_k    += self.pi/2*resid_x.T@resid_x + self.pi/2*resid_u.T@resid_u
        self.J_2_soft_k_orig =  self.gco + self.gij + self.Tcon + self.ql_n + self.din + self.sumfi + self.h_wcons 
    

    def SetADMMSubP2_SoftCost_N(self):
        # at the terminal step N
        self.J_2_soft_N    = self.Jl_P2_N  + self.gco + self.gij + self.Tcon + self.ql_n + self.din
        for i in range(int(self.nq)):
            xi      = self.xc[i*self.nxi:(i+1)*self.nxi]   # cable primal state
            scxi    = self.scxc[i*self.nxi:(i+1)*self.nxi] # safe copy state of each cable
            scxI    = self.scxC[i*self.nxi:(i+1)*self.nxi] # Lagrangian multiplier
            resid_x = xi - scxi + scxI/self.pi
            self.J_2_soft_N    += self.pi/2*resid_x.T@resid_x 
        self.J_2_soft_N_orig =  self.gco + self.gij + self.Tcon + self.ql_n + self.din


    

    def ADMM_SubP2_Init(self):
        # static optimization problem at step k 
        # start with an empty NLP
        w2        = [] # optimal trajectory list
        self.w02  = [] # initial guess list of optimal trajectory 
        self.lbw2 = [] # lower boundary list of optimal variables
        self.ubw2 = [] # upper boundary list of optimal variables
        g2        = [] # equality and inequality constraint list
        self.lbg2 = [] # lower boundary list of constraints
        self.ubg2 = [] # upper boundary list of constraints
        
        # hyperparameters + external signals
        Para2    = SX.sym('P2', (self.nxl # load primal state  
                                +self.nxl # load primal state's Lagrangian multiplier     
                                +self.nul # load primal control
                                +self.nul # load primal control's Lagrangian multuplier
                                +self.nxi*int(self.nq) # all the cable primal states
                                +self.nxi*int(self.nq) # all the cable primal states' Lagrangian multipliers
                                +self.nui*int(self.nq) # all the cable primal controls
                                +self.nui*int(self.nq) # all the cable primal controls' Lagrangian multipliers
                                +self.npl # load hyperparameters
                                +self.npi # cable shared hyperparameters     
                                +1 # the ADMM iteration index
                                )) 

        # formulate the NLP
        n_start_pl  = 2*(self.nxl+self.nul+self.nxi*int(self.nq)+self.nui*int(self.nq))
        para_l      = Para2[n_start_pl:n_start_pl+self.npl] 
        para_i      = Para2[n_start_pl+self.npl:n_start_pl+self.npl+self.npi]
        a           = Para2[-1]
        scxl_k      = SX.sym('scxl',self.nxl)
        w2         += [scxl_k]
        self.lbw2  += self.xl_lb
        self.ubw2  += self.xl_ub
        scul_k      = SX.sym('scul',self.nul)
        w2         += [scul_k]
        self.lbw2  += self.ul_lb
        self.ubw2  += self.ul_ub
        xl_k        = Para2[0:self.nxl]
        scxL_k      = Para2[self.nxl:2*self.nxl]
        ul_k        = Para2[2*self.nxl:2*self.nxl+self.nul]
        scuL_k      = Para2[2*self.nxl+self.nul:2*(self.nxl+self.nul)]
        # total cost at the step k that includes the load and all the cables
        J2          = self.Jl_P2_k_fn(xl0=xl_k,ul0=ul_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,paral0=para_l,a0=a)['Jl_P2_kf']
        scxc_k      = SX.sym('scxc',self.nxi*int(self.nq))
        scuc_k      = SX.sym('scuc',self.nui*int(self.nq))
        g2         += [self.ql_fn(scxl0=scxl_k)['norm_qlf']]
        self.lbg2  += [1]
        self.ubg2  += [1]
        for i in range(int(self.nq)):
            scxi_k  = SX.sym('scx'+str(i),self.nxi)
            w2     += [scxi_k]
            self.lbw2  += self.scxi_lb
            self.ubw2  += self.scxi_ub
            scxc_k[i*self.nxi:(i+1)*self.nxi] = scxi_k
            scui_k  = SX.sym('scu'+str(i),self.nui)
            w2     += [scui_k]
            self.lbw2  += self.ui_lb
            self.ubw2  += self.ui_ub
            scuc_k[i*self.nui:(i+1)*self.nui] = scui_k
            n_start_xi   = 2*(self.nxl+self.nul)
            xi_k    = Para2[n_start_xi+i*self.nxi:n_start_xi+(i+1)*self.nxi] # cable primal state
            n_start_scxI = n_start_xi + self.nxi*int(self.nq)
            scxI_k  = Para2[n_start_scxI+i*self.nxi:n_start_scxI+(i+1)*self.nxi] # cable primal state Lagrangian multiplier
            n_start_ui   = n_start_scxI + self.nxi*int(self.nq)
            ui_k    = Para2[n_start_ui+i*self.nui:n_start_ui+(i+1)*self.nui]
            n_start_scuI = n_start_ui + self.nui*int(self.nq)
            scuI_k  = Para2[n_start_scuI+i*self.nui:n_start_scuI+(i+1)*self.nui]
            J2     += self.Ji_P2_k_fn(xi0=xi_k,scxi0=scxi_k,scxI0=scxI_k,ui0=ui_k,scui0=scui_k,scuI0=scuI_k, parai0=para_i, a0=a)['Ji_P2_kf']
        
        for i in range(int(self.nq)):    
            # safe constriant between the obstacle 1 and the ith quadrotor
            goi1       = self.Gi1[i](scxl0=scxl_k,scxc0=scxc_k)['go1f'+str(i)]
            g2        += [goi1]
            self.lbg2 += [1e-2]
            self.ubg2 += [1e4] # add an upbound for numerical stability
            # safe constriant between the obstacle 2 and the ith quadrotor
            goi2       = self.Gi2[i](scxl0=scxl_k,scxc0=scxc_k)['go2f'+str(i)]
            g2        += [goi2]
            self.lbg2 += [1e-2]
            self.ubg2 += [1e4] # add an upbound for numerical stability
            # quadrotor's thrust limit
            gif        = self.fi[i](scxl0=scxl_k,scul0=scul_k,scxc0=scxc_k,scuc0=scuc_k)['norm_ff'+str(i)]
            g2        += [gif]
            self.lbg2 += [1e-2]
            self.ubg2 += [self.fmax**2] # tianchen's parameter
            # direction unit norm
            g2        += [self.Di[i](scxc0=scxc_k)['dn'+str(i)]]
            self.lbg2 += [1]
            self.ubg2 += [1] 
            # velocity limit
            # giv        = self.vi[i](scxl0=scxl_k,scxc0=scxc_k)['norm_vf'+str(i)]
            # g2        += [giv]
            # self.lbg2 += [0]
            # self.ubg2 += [self.vmax**2]
        
        for k in range(len(self.Gij)):
            gij        = self.Gij[k](scxl0=scxl_k,scxc0=scxc_k)['gf'+str(k)]
            g2        += [gij]
            self.lbg2 += [1e-2]
            self.ubg2 += [1e4]
        
        # control consensus constraint
        g_wc       = self.W_cons_fn(scxl0=scxl_k,scul0=scul_k,scxc0=scxc_k)['W_consf']
        g2        += [g_wc]
        self.lbg2 += self.nul*[0]
        self.ubg2 += self.nul*[0] 

        # create an NLP solver and solve it
        optsi2 = {}
        optsi2['ipopt.tol'] = 1e-8
        optsi2['ipopt.print_level'] = 0
        optsi2['print_time'] = 0
        optsi2['ipopt.warm_start_init_point']='yes'
        optsi2['ipopt.max_iter']=1e3
        optsi2['ipopt.acceptable_tol']=1e-8
        optsi2['ipopt.mu_strategy']='adaptive'
        optsi2['ipopt.bound_relax_factor']=1e-12
        optsi2['ipopt.limited_memory_max_history'] = 20
        optsi2['ipopt.nlp_scaling_method']='gradient-based'
        optsi2['ipopt.limited_memory_initialization'] = 'scalar1'
        prob2 = {'f': J2, 
                'x': vertcat(*w2), 
                'p': Para2,
                'g': vertcat(*g2)}
        self.solver2 = nlpsol('solver2', 'ipopt', prob2, optsi2)  



    def ADMM_SubP2_N_Init(self):
        # static optimization problem at step N (terminal) 
        # start with an empty NLP
        w2N        = [] # optimal trajectory list
        self.w02N  = [] # initial guess list of optimal trajectory 
        self.lbw2N = [] # lower boundary list of optimal variables
        self.ubw2N = [] # upper boundary list of optimal variables
        g2N        = [] # equality and inequality constraint list
        self.lbg2N = [] # lower boundary list of constraints
        self.ubg2N = [] # upper boundary list of constraints
        
        # hyperparameters + external signals
        Para2    = SX.sym('P2N', (self.nxl # load primal state at step N
                                +self.nxl  # load primal state's Lagrangian multiplier     
                                +self.nxi*int(self.nq) # all the cable primal states
                                +self.nxi*int(self.nq) # all the cable primal states' Lagrangian multipliers
                                +self.npl  # load hyperparameters
                                +self.npi  # cable hyperparameters
                                +1         # ADMM iteration index
                                )) 

        # formulate the NLP
        n_start_pl = 2*(self.nxl+self.nxi*int(self.nq))
        para_l     = Para2[n_start_pl:n_start_pl+self.npl] # penalty parameter of the load
        para_i     = Para2[n_start_pl+self.npl:n_start_pl+self.npl+self.npi]
        a          = Para2[-1]
        scxl_k     = SX.sym('scxl',self.nxl)
        w2N      += [scxl_k]
        self.lbw2N  += self.xl_lb
        self.ubw2N  += self.xl_ub
        xl_k     = Para2[0:self.nxl]
        scxL_k   = Para2[self.nxl:2*self.nxl]
        # total cost at the step k that includes the load and all the cables
        J2       = self.Jl_P2_N_fn(xl0=xl_k,scxl0=scxl_k,scxL0=scxL_k,paral0=para_l,a0=a)['Jl_P2_Nf']
        scxc_k   = SX.sym('scxc',self.nxi*int(self.nq))
        g2N        += [self.ql_fn(scxl0=scxl_k)['norm_qlf']]
        self.lbg2N  += [1]
        self.ubg2N  += [1]
        for i in range(int(self.nq)):
            scxi_k  = SX.sym('scx'+str(i),self.nxi)
            w2N     += [scxi_k]
            self.lbw2N  += self.scxi_lb
            self.ubw2N  += self.scxi_ub
            scxc_k[i*self.nxi:(i+1)*self.nxi] = scxi_k
            xi_k    = Para2[2*self.nxl+i*self.nxi:2*self.nxl+(i+1)*self.nxi] # cable primal state
            scxI_k  = Para2[2*self.nxl+self.nxi*int(self.nq)+i*self.nxi:2*self.nxl+self.nxi*int(self.nq)+(i+1)*self.nxi] # cable primal state Lagrangian multiplier
            J2     += self.Ji_P2_N_fn(xi0=xi_k,scxi0=scxi_k,scxI0=scxI_k,parai0=para_i,a0=a)['Jl_P2_Nf']
        
        for i in range(int(self.nq)):
            # safe constriant between the obstacle 1 and the ith quadrotor
            goi1        = self.Gi1[i](scxl0=scxl_k,scxc0=scxc_k)['go1f'+str(i)]
            g2N        += [goi1]
            self.lbg2N += [1e-2]
            self.ubg2N += [1e4] # add an upbound for numerical stability
            # safe constriant between the obstacle 2 and the ith quadrotor
            goi2        = self.Gi2[i](scxl0=scxl_k,scxc0=scxc_k)['go2f'+str(i)]
            g2N        += [goi2]
            self.lbg2N += [1e-2]
            self.ubg2N += [1e4] # add an upbound for numerical stability
            # direction unit norm
            g2N        += [self.Di[i](scxc0=scxc_k)['dn'+str(i)]]
            self.lbg2N += [1]
            self.ubg2N += [1]
            # velocity limit
            # giv         = self.vi[i](scxl0=scxl_k,scxc0=scxc_k)['norm_vf'+str(i)]
            # g2N        += [giv]
            # self.lbg2N += [0]
            # self.ubg2N += [self.vmax**2]
        
        for k in range(len(self.Gij)):
            gij         = self.Gij[k](scxl0=scxl_k,scxc0=scxc_k)['gf'+str(k)]
            g2N        += [gij]
            self.lbg2N += [1e-2]
            self.ubg2N += [1e4]
        

        # create an NLP solver and solve it
        optsi2 = {}
        optsi2['ipopt.tol'] = 1e-8
        optsi2['ipopt.print_level'] = 0
        optsi2['print_time'] = 0
        optsi2['ipopt.warm_start_init_point']='yes'
        optsi2['ipopt.max_iter']=1e3
        optsi2['ipopt.acceptable_tol']=1e-8
        optsi2['ipopt.mu_strategy']='adaptive'
        optsi2['ipopt.bound_relax_factor']=1e-12
        optsi2['ipopt.limited_memory_max_history'] = 20
        optsi2['ipopt.nlp_scaling_method']='gradient-based'
        optsi2['ipopt.limited_memory_initialization'] = 'scalar1'
        prob2N = {'f': J2, 
                'x': vertcat(*w2N), 
                'p': Para2,
                'g': vertcat(*g2N)}
        self.solver2N = nlpsol('solver2N', 'ipopt', prob2N, optsi2)  


    
    def ADMM_SubP2(self,Para2_cable):
        # Para2_cable = SX.sym('p2_cable',(self.nxl*(self.N+1) # load reference state for initialization
        #                                 +self.nul*self.N # load reference control for initialization 
        #                                 +self.nxi*self.nq*(self.N+1) # cables' reference states for initialization
        #                                 +self.nui*self.nq # cables' reference controls for initialization
        #---------------------------------------------------------------------------------------------------
        #                                 +self.nxl*(self.N+1) # load primal state trajectory
        #                                 +self.nxl*(self.N+1) # load primal state's Lagrangian multiplier trajectory
        #                                 +self.nul*self.N # load primal control trajectory
        #                                 +self.nul*self.N # load primal control's Lagrangian multiplier trajectory
        #                                 +self.nxi*self.nq*(self.N+1) # cables' primal state trajectories
        #                                 +self.nxi*self.nq*(self.N+1) # cables' primal state's Lagrangian multiplier trajectories
        #                                 +self.nui*self.nq*self.N # cables' primal control trajectories
        #                                 +self.nui*self.nq*self.N # cables' primal control's Lagrangian multiplier trajectories
        #                                 +self.npl # load hyperparameters
        #                                 +self.npi # cable hyperparameters
        #                                 +1 # ADMM iteration index
        #))
        scxl_traj    = np.zeros((self.N+1,self.nxl))
        scul_traj    = np.zeros((self.N,self.nul))
        scxc_traj    = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        scuc_traj    = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))]
        n_start_pl   = 3*self.nxl*(self.N+1)+3*self.nul*self.N+3*self.nxi*int(self.nq)*(self.N+1)+2*self.nui*int(self.nq)*self.N + self.nui*int(self.nq)
        para_l       = Para2_cable[n_start_pl:n_start_pl+self.npl] # load ADMM penalty parameter for load state
        para_i       = Para2_cable[n_start_pl+self.npl:n_start_pl+self.npl+self.npi]
        a            = Para2_cable[-1]
        for k in range(self.N):
            self.w02 = []
            xl_ref   = Para2_cable[k*self.nxl:(k+1)*self.nxl]
            ul_ref   = Para2_cable[2*self.nxl*(self.N+1)+k*self.nul:2*self.nxl*(self.N+1)+(k+1)*self.nul]
            scxl0    = []
            for j in range(self.nxl):
                scxl0 += [xl_ref[j]]
            self.w02 += scxl0
            scul0    = []
            for j in range(self.nul):
                scul0 += [ul_ref[j]]
            self.w02 += scul0
            n_start_xl   = self.nxl*(self.N+1)+self.nul*self.N+self.nxi*int(self.nq)*(self.N+1)+self.nui*int(self.nq)
            xl_k         = Para2_cable[n_start_xl+k*self.nxl:n_start_xl+(k+1)*self.nxl]
            n_start_scxL = n_start_xl + self.nxl*(self.N+1)
            scxL_k       = Para2_cable[n_start_scxL+k*self.nxl:n_start_scxL+(k+1)*self.nxl]
            n_start_ul   = n_start_scxL + self.nxl*(self.N+1)
            ul_k         = Para2_cable[n_start_ul+k*self.nul:n_start_ul+(k+1)*self.nul]
            n_start_scuL = n_start_ul + self.nul*self.N
            scuL_k       = Para2_cable[n_start_scuL+k*self.nul:n_start_scuL+(k+1)*self.nul]
            n_start_xc   = n_start_scuL + self.nul*self.N
            xc_k         = Para2_cable[n_start_xc+k*self.nxi*int(self.nq):n_start_xc+(k+1)*self.nxi*int(self.nq)]
            n_start_scxC = n_start_xc + self.nxi*int(self.nq)*(self.N+1)
            scxC_k       = Para2_cable[n_start_scxC+k*self.nxi*int(self.nq):n_start_scxC+(k+1)*self.nxi*int(self.nq)]
            n_start_uc   = n_start_scxC + self.nxi*int(self.nq)*(self.N+1)
            uc_k         = Para2_cable[n_start_uc+k*self.nui*int(self.nq):n_start_uc+(k+1)*self.nui*int(self.nq)]
            n_start_scuC = n_start_uc + self.nui*int(self.nq)*self.N
            scuC_k       = Para2_cable[n_start_scuC+k*self.nui*int(self.nq):n_start_scuC+(k+1)*self.nui*int(self.nq)]
            xq_ref_k     = Para2_cable[self.nxl*(self.N+1)+self.nul*self.N+k*self.nxi*int(self.nq):self.nxl*(self.N+1)+self.nul*self.N+(k+1)*self.nxi*int(self.nq)]
            
            for i in range(int(self.nq)):
                scxi0   = []
                xi_ref  = xq_ref_k[i*self.nxi:(i+1)*self.nxi]
                for j in range(self.nxi):
                    scxi0 +=[xi_ref[j]]
                self.w02 += scxi0
                scui0   = []
                ui_ref  = Para2_cable[self.nxl*(self.N+1)+self.nul*self.N+self.nxi*int(self.nq)*(self.N+1)+i*self.nui:self.nxl*(self.N+1)+self.nul*self.N+self.nxi*int(self.nq)*(self.N+1)+(i+1)*self.nui]
                for j in range(self.nui):
                    scui0 +=[ui_ref[j]]
                self.w02 += scui0
            para2   = np.concatenate((xl_k,scxL_k))
            para2   = np.concatenate((para2,ul_k))
            para2   = np.concatenate((para2,scuL_k))
            para2   = np.concatenate((para2,xc_k))
            para2   = np.concatenate((para2,scxC_k))
            para2   = np.concatenate((para2,uc_k))
            para2   = np.concatenate((para2,scuC_k))
            para2   = np.concatenate((para2,para_l))
            para2   = np.concatenate((para2,para_i))
            para2   = np.concatenate((para2,[a]))
            # Solve the NLP
            sol2 = self.solver2(x0=self.w02, 
                          lbx=self.lbw2, 
                          ubx=self.ubw2, 
                          p=para2,
                          lbg=self.lbg2, 
                          ubg=self.ubg2)
            w_opt2 = sol2['x'].full().flatten()
            # take the optimal control and state
            sol_traj = np.reshape(w_opt2, (-1, self.nxl + self.nul + (self.nxi+self.nui)*int(self.nq)))
            scxl_opt = sol_traj[:,0:self.nxl]
            scul_opt = sol_traj[:,self.nxl:self.nxl + self.nul]
            scc_opt  = sol_traj[:,self.nxl + self.nul:self.nxl + self.nul + (self.nxi+self.nui)*int(self.nq)]
            scxl_traj[k:k+1,:] = scxl_opt
            scul_traj[k:k+1,:] = scul_opt
            for i in range(int(self.nq)):
                scxc_traj[i][k:k+1,:]=scc_opt[:,i*(self.nxi+self.nui):i*(self.nxi+self.nui)+self.nxi]
                scuc_traj[i][k:k+1,:]=scc_opt[:,i*(self.nxi+self.nui)+self.nxi:(i+1)*(self.nxi+self.nui)]
        
        # terminal cost
        self.w02N = []
        xl_ref   = Para2_cable[self.N*self.nxl:(self.N+1)*self.nxl]
        scxl0N    = []
        for j in range(self.nxl):
            scxl0N += [xl_ref[j]]
        self.w02N += scxl0N
        xq_ref_N  = Para2_cable[self.nxl*(self.N+1)+self.nul*self.N+self.nxi*int(self.nq)*self.N:self.nxl*(self.N+1)+self.nul*self.N+self.nxi*int(self.nq)*(self.N+1)]
        for i in range(int(self.nq)):
            scxi0N   = []
            xi_ref  = xq_ref_N[i*self.nxi:(i+1)*self.nxi]
            for j in range(self.nxi):
                scxi0N +=[xi_ref[j]]
            self.w02N += scxi0N
        xl_N    = Para2_cable[n_start_xl+self.N*self.nxl:n_start_xl+(self.N+1)*self.nxl]
        scxL_N  = Para2_cable[n_start_scxL+self.N*self.nxl:n_start_scxL+(self.N+1)*self.nxl]
        xc_N    = Para2_cable[n_start_xc+self.N*self.nxi*int(self.nq):n_start_xc+(self.N+1)*self.nxi*int(self.nq)]
        scxC_N  = Para2_cable[n_start_scxC+self.N*self.nxi*int(self.nq):n_start_scxC+(self.N+1)*self.nxi*int(self.nq)]
        para2N  = np.concatenate((xl_N,scxL_N))
        para2N  = np.concatenate((para2N,xc_N))
        para2N  = np.concatenate((para2N,scxC_N))
        para2N  = np.concatenate((para2N,para_l))
        para2N  = np.concatenate((para2N,para_i))
        para2N  = np.concatenate((para2N,[a]))
        # Solve the NLP
        sol2N = self.solver2N(x0=self.w02N, 
                          lbx=self.lbw2N, 
                          ubx=self.ubw2N, 
                          p=para2N,
                          lbg=self.lbg2N, 
                          ubg=self.ubg2N)
        w_opt2N = sol2N['x'].full().flatten()
        sol_trajN = np.reshape(w_opt2N, (-1, self.nxl + self.nxi*int(self.nq)))
        scxl_optN = sol_trajN[:,0:self.nxl]
        scxc_optN = sol_trajN[:,self.nxl:self.nxl+ self.nxi*int(self.nq)]
        scxl_traj[self.N:self.N+1,:] = scxl_optN
        for i in range(int(self.nq)):
            scxc_traj[i][self.N:self.N+1,:]=scxc_optN[:,i*self.nxi:(i+1)*self.nxi]
        # output
        opt_sol2 = {"scxl_traj":scxl_traj,
                    "scul_traj":scul_traj,
                    "scxc_traj":scxc_traj,
                    "scuc_traj":scuc_traj
                    }
        
        return opt_sol2
    

    def system_derivatives_SubP2_ADMM_k(self):
        # gradients of the Lagrangian (augmented cost function with the soft constraints)
        self.Lscxl          = jacobian(self.J_2_soft_k,self.scxl)
        self.Lscul          = jacobian(self.J_2_soft_k,self.scul)
        self.Lscxc          = jacobian(self.J_2_soft_k,self.scxc)
        self.Lscuc          = jacobian(self.J_2_soft_k,self.scuc)
        # gradients of the original Lagrangian (augmented cost with the soft constraints but without the ADMM penalties)
        self.Lscxl_o        = jacobian(self.J_2_soft_k_orig,self.scxl)
        self.Lscul_o        = jacobian(self.J_2_soft_k_orig,self.scul)
        self.Lscxc_o        = jacobian(self.J_2_soft_k_orig,self.scxc)
        self.Lscuc_o        = jacobian(self.J_2_soft_k_orig,self.scuc)
        # hessians
        self.Lscxlscxl      = jacobian(self.Lscxl,self.scxl)
        self.Lscxlscxl_fn   = Function('Lscxlscxl',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxlscxl],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxlscxl_f'])
        self.Lscxlscul      = jacobian(self.Lscxl,self.scul)
        self.Lscxlscul_fn   = Function('Lscxlscul',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxlscul],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxlscul_f'])
        self.Lscxlscxc      = jacobian(self.Lscxl,self.scxc)
        self.Lscxlscxc_fn   = Function('Lscxlscxc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxlscxc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxlscxc_f'])
        self.Lscxlscuc      = jacobian(self.Lscxl,self.scuc)
        self.Lscxlscuc_fn   = Function('Lscxlscuc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxlscuc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxlscuc_f'])
        self.Lsculscul      = jacobian(self.Lscul,self.scul)
        self.Lsculscul_fn   = Function('Lsculscul',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lsculscul],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lsculscul_f'])
        self.Lsculscxc      = jacobian(self.Lscul,self.scxc)
        self.Lsculscxc_fn   = Function('Lsculscxc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lsculscxc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lsculscxc_f'])
        self.Lsculscuc      = jacobian(self.Lscul,self.scuc)
        self.Lsculscuc_fn   = Function('Lsculscuc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lsculscuc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lsculscuc_f'])
        self.Lscxcscxc      = jacobian(self.Lscxc,self.scxc)
        self.Lscxcscxc_fn   = Function('Lscxcscxc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxcscxc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxcscxc_f'])
        self.Lscxcscuc      = jacobian(self.Lscxc,self.scuc)
        self.Lscxcscuc_fn   = Function('Lscxcscuc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxcscuc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxcscuc_f'])
        self.Lscucscuc      = jacobian(self.Lscuc,self.scuc)
        self.Lscucscuc_fn   = Function('Lscucscuc',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscucscuc],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscucscuc_f'])
        # hessians of the original Lagrangian
        self.Lscxlscxl_o    = jacobian(self.Lscxl_o,self.scxl)
        self.Lscxlscxl_fno  = Function('Lscxlscxlo',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxlscxl_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxlscxlo_f'])
        self.Lscxlscul_o    = jacobian(self.Lscxl_o,self.scul)
        self.Lscxlscul_fno  = Function('Lscxlsculo',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxlscul_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxlsculo_f'])
        self.Lscxlscxc_o    = jacobian(self.Lscxl_o,self.scxc)
        self.Lscxlscxc_fno  = Function('Lscxlscxco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxlscxc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxlscxco_f'])
        self.Lscxlscuc_o    = jacobian(self.Lscxl_o,self.scuc)
        self.Lscxlscuc_fno  = Function('Lscxlscuco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxlscuc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxlscuco_f'])
        self.Lsculscul_o    = jacobian(self.Lscul_o,self.scul)
        self.Lsculscul_fno  = Function('Lsculsculo',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lsculscul_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lsculsculo_f'])
        self.Lsculscxc_o    = jacobian(self.Lscul_o,self.scxc)
        self.Lsculscxc_fno  = Function('Lsculscxco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lsculscxc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lsculscxco_f'])
        self.Lsculscuc_o    = jacobian(self.Lscul_o,self.scuc)
        self.Lsculscuc_fno  = Function('Lsculscuco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lsculscuc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lsculscuco_f'])
        self.Lscxcscxc_o    = jacobian(self.Lscxc_o,self.scxc)
        self.Lscxcscxc_fno  = Function('Lscxcscxco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxcscxc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxcscxco_f'])
        self.Lscxcscuc_o    = jacobian(self.Lscxc_o,self.scuc)
        self.Lscxcscuc_fno  = Function('Lscxcscuco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscxcscuc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscxcscuco_f'])
        self.Lscucscuc_o    = jacobian(self.Lscuc_o,self.scuc)
        self.Lscucscuc_fno  = Function('Lscucscuco',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC],[self.Lscucscuc_o],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0'],['Lscucscuco_f'])

        # hessians w.r.t. the hyperparameters
        self.Lscxlp         = jacobian(self.Lscxl,self.P_auto)
        self.Lscxlp_fn      = Function('Lscxlp',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxlp],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxlp_f'])
        self.Lsculp         = jacobian(self.Lscul,self.P_auto)
        self.Lsculp_fn      = Function('Lsculp',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lsculp],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lsculp_f'])
        self.Lscxcp         = jacobian(self.Lscxc,self.P_auto)
        self.Lscxcp_fn      = Function('Lscxcp',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscxcp],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscxcp_f'])
        self.Lscucp         = jacobian(self.Lscuc,self.P_auto)
        self.Lscucp_fn      = Function('Lscucp',[self.xl,self.ul,self.xc,self.uc,self.scxl,self.scxL,self.scul,self.scuL,self.scxc,self.scxC,self.scuc,self.scuC,self.P_auto,self.a],[self.Lscucp],
                                       ['xl0','ul0','xc0','uc0','scxl0','scxL0','scul0','scuL0','scxc0','scxC0','scuc0','scuC0','pauto0','a0'],['Lscucp_f'])

    def system_derivatives_SubP2_ADMM_N(self):
        # gradients of the Lagrangian (augmented cost function with the soft constraints)
        self.Lscxl_N        = jacobian(self.J_2_soft_N,self.scxl)
        self.Lscxc_N        = jacobian(self.J_2_soft_N,self.scxc)
        # gradients of the original Lagrangian (augmented cost with the soft constraints but without the ADMM penalties)
        self.Lscxl_N_o      = jacobian(self.J_2_soft_N_orig,self.scxl)
        self.Lscxc_N_o      = jacobian(self.J_2_soft_N_orig,self.scxc)
        # hessians
        self.Lscxlscxl_N    = jacobian(self.Lscxl_N,self.scxl)
        self.Lscxlscxl_N_fn = Function('LscxlscxlN',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC,self.P_auto,self.a],[self.Lscxlscxl_N],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0','pauto0','a0'],['LscxlscxlN_f'])
        self.Lscxlscxc_N    = jacobian(self.Lscxl_N,self.scxc)
        self.Lscxlscxc_N_fn = Function('LscxlscxcN',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC,self.P_auto,self.a],[self.Lscxlscxc_N],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0','pauto0','a0'],['LscxlscxcN_f'])
        self.Lscxcscxc_N    = jacobian(self.Lscxc_N,self.scxc)
        self.Lscxcscxc_N_fn = Function('LscxcscxcN',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC,self.P_auto,self.a],[self.Lscxcscxc_N],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0','pauto0','a0'],['LscxcscxcN_f'])
        # hessians of the original Lagrangian
        self.Lscxlscxl_No    = jacobian(self.Lscxl_N_o,self.scxl)
        self.Lscxlscxl_N_fno = Function('LscxlscxlNo',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC],[self.Lscxlscxl_No],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0'],['LscxlscxlNo_f'])
        self.Lscxlscxc_No    = jacobian(self.Lscxl_N_o,self.scxc)
        self.Lscxlscxc_N_fno = Function('LscxlscxcNo',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC],[self.Lscxlscxc_No],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0'],['LscxlscxcNo_f'])
        self.Lscxcscxc_No    = jacobian(self.Lscxc_N_o,self.scxc)
        self.Lscxcscxc_N_fno = Function('LscxcscxcNo',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC],[self.Lscxcscxc_No],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0'],['LscxcscxcNo_f'])
        # hessians w.r.t. the hyperparameters
        self.Lscxlp_N       = jacobian(self.Lscxl_N,self.P_auto)
        self.Lscxlp_N_fn    = Function('LscxlpN',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC,self.P_auto,self.a],[self.Lscxlp_N],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0','pauto0','a0'],['LscxlpN_f'])
        self.Lscxcp_N       = jacobian(self.Lscxc_N,self.P_auto)
        self.Lscxcp_N_fn    = Function('LscxcpN',[self.xl,self.xc,self.scxl,self.scxL,self.scxc,self.scxC,self.P_auto,self.a],[self.Lscxcp_N],
                                       ['xl0','xc0','scxl0','scxL0','scxc0','scxC0','pauto0','a0'],['LscxcpN_f'])



    def Get_AuxSys_SubP2(self,opt_sol1_l,opt_sol1_c,opt_sol2,scxL,scuL,scxC_list,scuC_list,Pauto,i_admm):
        xl      = opt_sol1_l['xl_traj']
        ul      = opt_sol1_l['ul_traj']
        xc_list      = opt_sol1_c['xc_traj'] # list that contains all the cables' states
        uc_list      = opt_sol1_c['uc_traj'] # list that contains all the cables' controls
        scxl    = opt_sol2['scxl_traj']
        scul    = opt_sol2['scul_traj']
        scxc_list    = opt_sol2['scxc_traj'] # list that contains all the cables' safe states
        scuc_list    = opt_sol2['scuc_traj'] # list that contains all the cables' safe controls
        Lscxlscxl    = (self.N+1)*[np.zeros((self.nxl,self.nxl))]
        Lscxlscul    = self.N*[np.zeros((self.nxl,self.nul))]
        Lscxlscxc    = (self.N+1)*[np.zeros((self.nxl,self.nxi*int(self.nq)))]
        Lscxlscuc    = self.N*[np.zeros((self.nxl,self.nui*int(self.nq)))]
        Lsculscul    = self.N*[np.zeros((self.nul,self.nul))]
        Lsculscxc    = self.N*[np.zeros((self.nul,self.nxi*int(self.nq)))]
        Lsculscuc    = self.N*[np.zeros((self.nul,self.nui*int(self.nq)))]
        Lscxcscxc    = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.nxi*int(self.nq)))]
        Lscxcscuc    = self.N*[np.zeros((self.nxi*int(self.nq),self.nui*int(self.nq)))]
        Lscucscuc    = self.N*[np.zeros((self.nui*int(self.nq),self.nui*int(self.nq)))]
        Lscxlp       = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        Lsculp       = self.N*[np.zeros((self.nul,self.n_Pauto))]
        Lscxcp       = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        Lscucp       = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        # hessians of the original Lagrangian for computing the minimal eigenvalue
        Lscxlscxl_o  = (self.N+1)*[np.zeros((self.nxl,self.nxl))]
        Lscxlscul_o  = self.N*[np.zeros((self.nxl,self.nul))]
        Lscxlscxc_o  = (self.N+1)*[np.zeros((self.nxl,self.nxi*int(self.nq)))]
        Lscxlscuc_o  = self.N*[np.zeros((self.nxl,self.nui*int(self.nq)))]
        Lsculscul_o  = self.N*[np.zeros((self.nul,self.nul))]
        Lsculscxc_o  = self.N*[np.zeros((self.nul,self.nxi*int(self.nq)))]
        Lsculscuc_o  = self.N*[np.zeros((self.nul,self.nui*int(self.nq)))]
        Lscxcscxc_o  = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.nxi*int(self.nq)))]
        Lscxcscuc_o  = self.N*[np.zeros((self.nxi*int(self.nq),self.nui*int(self.nq)))]
        Lscucscuc_o  = self.N*[np.zeros((self.nui*int(self.nq),self.nui*int(self.nq)))]
        for k in range(self.N):
            xl_k     = xl[k,:]
            ul_k     = ul[k,:]
            xc_k     = np.concatenate([xc_list[i][k,:] for i in range(int(self.nq))])
            uc_k     = np.concatenate([uc_list[i][k,:] for i in range(int(self.nq))])
            scxl_k   = scxl[k,:]
            scxL_k   = scxL[k,:]
            scul_k   = scul[k,:]
            scuL_k   = scuL[k,:]
            scxc_k   = np.concatenate([scxc_list[i][k,:] for i in range(int(self.nq))])
            scxC_k   = np.concatenate([scxC_list[i][k,:] for i in range(int(self.nq))])
            scuc_k   = np.concatenate([scuc_list[i][k,:] for i in range(int(self.nq))])
            scuC_k   = np.concatenate([scuC_list[i][k,:] for i in range(int(self.nq))])
            Lscxlscxl[k] = self.Lscxlscxl_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxlscxl_f'].full()
            Lscxlscul[k] = self.Lscxlscul_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxlscul_f'].full()
            Lscxlscxc[k] = self.Lscxlscxc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxlscxc_f'].full()
            Lscxlscuc[k] = self.Lscxlscuc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxlscuc_f'].full()
            Lsculscul[k] = self.Lsculscul_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lsculscul_f'].full()
            Lsculscxc[k] = self.Lsculscxc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lsculscxc_f'].full()
            Lsculscuc[k] = self.Lsculscuc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lsculscuc_f'].full()
            Lscxcscxc[k] = self.Lscxcscxc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxcscxc_f'].full()
            Lscxcscuc[k] = self.Lscxcscuc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxcscuc_f'].full()
            Lscucscuc[k] = self.Lscucscuc_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscucscuc_f'].full()
            Lscxlp[k]    = self.Lscxlp_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxlp_f'].full()
            Lsculp[k]    = self.Lsculp_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lsculp_f'].full()
            Lscxcp[k]    = self.Lscxcp_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscxcp_f'].full()
            Lscucp[k]    = self.Lscucp_fn(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k,pauto0=Pauto,a0=i_admm)['Lscucp_f'].full()
            # hessians of the original Lagrangian
            Lscxlscxl_o[k] = self.Lscxlscxl_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxlscxlo_f'].full()
            Lscxlscul_o[k] = self.Lscxlscul_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxlsculo_f'].full()
            Lscxlscxc_o[k] = self.Lscxlscxc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxlscxco_f'].full()
            Lscxlscuc_o[k] = self.Lscxlscuc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxlscuco_f'].full()
            Lsculscul_o[k] = self.Lsculscul_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lsculsculo_f'].full()
            Lsculscxc_o[k] = self.Lsculscxc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lsculscxco_f'].full()
            Lsculscuc_o[k] = self.Lsculscuc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lsculscuco_f'].full()
            Lscxcscxc_o[k] = self.Lscxcscxc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxcscxco_f'].full()
            Lscxcscuc_o[k] = self.Lscxcscuc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscxcscuco_f'].full()
            Lscucscuc_o[k] = self.Lscucscuc_fno(xl0=xl_k,ul0=ul_k,xc0=xc_k,uc0=uc_k,scxl0=scxl_k,scxL0=scxL_k,scul0=scul_k,scuL0=scuL_k,scxc0=scxc_k,scxC0=scxC_k,scuc0=scuc_k,scuC0=scuC_k)['Lscucscuco_f'].full()

        # ternimal hessians
        xl_N     = xl[-1,:]
        xc_N     = np.concatenate([xc_list[i][-1,:] for i in range(int(self.nq))])
        scxl_N   = scxl[-1,:]
        scxL_N   = scxL[-1,:]
        scxc_N   = np.concatenate([scxc_list[i][-1,:] for i in range(int(self.nq))])
        scxC_N   = np.concatenate([scxC_list[i][-1,:] for i in range(int(self.nq))])
        Lscxlscxl[self.N] = self.Lscxlscxl_N_fn(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N,pauto0=Pauto,a0=i_admm)['LscxlscxlN_f'].full()
        Lscxlscxc[self.N] = self.Lscxlscxc_N_fn(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N,pauto0=Pauto,a0=i_admm)['LscxlscxcN_f'].full()
        Lscxcscxc[self.N] = self.Lscxcscxc_N_fn(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N,pauto0=Pauto,a0=i_admm)['LscxcscxcN_f'].full()
        Lscxlp[self.N]    = self.Lscxlp_N_fn(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N,pauto0=Pauto,a0=i_admm)['LscxlpN_f'].full()
        Lscxcp[self.N]    = self.Lscxcp_N_fn(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N,pauto0=Pauto,a0=i_admm)['LscxcpN_f'].full()
        # hessians of the original Lagrangian
        Lscxlscxl_o[self.N] = self.Lscxlscxl_N_fno(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N)['LscxlscxlNo_f'].full()
        Lscxlscxc_o[self.N] = self.Lscxlscxc_N_fno(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N)['LscxlscxcNo_f'].full()
        Lscxcscxc_o[self.N] = self.Lscxcscxc_N_fno(xl0=xl_N,xc0=xc_N,scxl0=scxl_N,scxL0=scxL_N,scxc0=scxc_N,scxC0=scxC_N)['LscxcscxcNo_f'].full()

        auxsys2 = {
            "Lscxlscxl":Lscxlscxl,
            "Lscxlscul":Lscxlscul,
            "Lscxlscxc":Lscxlscxc,
            "Lscxlscuc":Lscxlscuc,
            "Lsculscul":Lsculscul,
            "Lsculscxc":Lsculscxc,
            "Lsculscuc":Lsculscuc,
            "Lscxcscxc":Lscxcscxc,
            "Lscxcscuc":Lscxcscuc,
            "Lscucscuc":Lscucscuc,
            "Lscxlp":Lscxlp,
            "Lsculp":Lsculp,
            "Lscxcp":Lscxcp,
            "Lscucp":Lscucp,
            "Lscxlscxl_o":Lscxlscxl_o,
            "Lscxlscul_o":Lscxlscul_o,
            "Lscxlscxc_o":Lscxlscxc_o,
            "Lscxlscuc_o":Lscxlscuc_o,
            "Lsculscul_o":Lsculscul_o,
            "Lsculscxc_o":Lsculscxc_o,
            "Lsculscuc_o":Lsculscuc_o,
            "Lscxcscxc_o":Lscxcscxc_o,
            "Lscxcscuc_o":Lscxcscuc_o,
            "Lscucscuc_o":Lscucscuc_o
        }

        return auxsys2

    
    def ADMM_SubP3(self,xl_traj,scxl_traj,scxL_traj,ul_traj,scul_traj,scuL_traj,xc_traj,scxc_traj,scxC_traj,uc_traj,scuc_traj,scuC_traj,p,pi,i_admm):
        scxL_traj_new = np.zeros((self.N+1,self.nxl))
        scuL_traj_new = np.zeros((self.N,self.nul))
        scxC_traj_new = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        scuC_traj_new = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))]
        dis_rn        = self.Discount_rate(i_admm) # the numerical value of the iteration-specific ADMM penalty parameter
        for k in range(self.N):
            scxL_new  = scxL_traj[k,:] + dis_rn*p*(xl_traj[k,:] - scxl_traj[k,:])
            scuL_new  = scuL_traj[k,:] + dis_rn*p*(ul_traj[k,:] - scul_traj[k,:])
            scxL_traj_new[k:k+1,:] = scxL_new
            scuL_traj_new[k:k+1,:] = scuL_new
            for i in range(int(self.nq)):
                scxI_new = scxC_traj[i][k,:] + dis_rn*pi*(xc_traj[i][k,:] - scxc_traj[i][k,:])
                scxC_traj_new[i][k:k+1,:] = scxI_new
                scuI_new = scuC_traj[i][k,:] + dis_rn*pi*(uc_traj[i][k,:] - scuc_traj[i][k,:])
                scuC_traj_new[i][k:k+1,:] = scuI_new
        scxL_new  = scxL_traj[self.N,:] + dis_rn*p*(xl_traj[self.N,:] - scxl_traj[self.N,:])
        scxL_traj_new[self.N:self.N+1,:] = scxL_new
        for i in range(int(self.nq)):
            scxI_new = scxC_traj[i][self.N,:] + dis_rn*pi*(xc_traj[i][self.N,:] - scxc_traj[i][self.N,:])
            scxC_traj_new[i][self.N:self.N+1,:] = scxI_new

        opt_sol3 = {"scxL_traj_new":scxL_traj_new,
                    "scuL_traj_new":scuL_traj_new,
                    "scxC_traj_new":scxC_traj_new,
                    "scuC_traj_new":scuC_traj_new
                    }
        
        return opt_sol3
    
    def system_derivatives_SubP3_ADMM(self):
        scxL_update = self.p*(self.xl - self.scxl)
        scuL_update = self.p*(self.ul - self.scul)
        scxC_update = self.pi*(self.xc - self.scxc)
        scuC_update = self.pi*(self.uc - self.scuc)
        self.dscxL_updatedp    = jacobian(scxL_update,self.P_auto)
        self.dscxL_updatedp_fn = Function('dscxL_updatedp',[self.xl,self.scxl,self.a],[self.dscxL_updatedp],['xl0','scxl0','a0'],['dscxL_updatedp_f'])
        self.dscuL_updatedp    = jacobian(scuL_update,self.P_auto)
        self.dscuL_updatedp_fn = Function('dscuL_updatedp',[self.ul,self.scul,self.a],[self.dscuL_updatedp],['ul0','scul0','a0'],['dscuL_updatedp_f'])
        self.dscxC_updatedp    = jacobian(scxC_update,self.P_auto)
        self.dscxC_updatedp_fn = Function('dscxC_updatedp',[self.xc,self.scxc,self.a],[self.dscxC_updatedp],['xc0','scxc0','a0'],['dscxC_updatedp_f'])
        self.dscuC_updatedp    = jacobian(scuC_update,self.P_auto)
        self.dscuC_updatedp_fn = Function('dscuC_updatedp',[self.uc,self.scuc,self.a],[self.dscuC_updatedp],['uc0','scuc0','a0'],['dscuC_updatedp_f'])


    def Get_AuxSys_SubP3(self,opt_sol1_l,opt_sol1_c,opt_sol2,i_admm):
        xl      = opt_sol1_l['xl_traj']
        ul      = opt_sol1_l['ul_traj']
        xc_list = opt_sol1_c['xc_traj']
        uc_list = opt_sol1_c['uc_traj']
        scxl    = opt_sol2['scxl_traj']
        scul    = opt_sol2['scul_traj']
        scxc_list    = opt_sol2['scxc_traj']
        scuc_list    = opt_sol2['scuc_traj']
        dscxL_updatedp = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        dscuL_updatedp = self.N*[np.zeros((self.nul,self.n_Pauto))]
        dscxC_updatedp = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        dscuC_updatedp = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        for k in range(self.N):
            xl_k     = xl[k,:]
            ul_k     = ul[k,:]
            xc_k     = np.concatenate([xc_list[i][k,:] for i in range(int(self.nq))])
            uc_k     = np.concatenate([uc_list[i][k,:] for i in range(int(self.nq))])
            scxl_k   = scxl[k,:]
            scul_k   = scul[k,:]
            scxc_k   = np.concatenate([scxc_list[i][k,:] for i in range(int(self.nq))])
            scuc_k   = np.concatenate([scuc_list[i][k,:] for i in range(int(self.nq))])
            dscxL_updatedp[k] = self.dscxL_updatedp_fn(xl0=xl_k,scxl0=scxl_k,a0=i_admm)['dscxL_updatedp_f'].full()
            dscuL_updatedp[k] = self.dscuL_updatedp_fn(ul0=ul_k,scul0=scul_k,a0=i_admm)['dscuL_updatedp_f'].full()
            dscxC_updatedp[k] = self.dscxC_updatedp_fn(xc0=xc_k,scxc0=scxc_k,a0=i_admm)['dscxC_updatedp_f'].full()
            dscuC_updatedp[k] = self.dscuC_updatedp_fn(uc0=uc_k,scuc0=scuc_k,a0=i_admm)['dscuC_updatedp_f'].full()
        xl_N     = xl[-1,:]
        scxl_N   = scxl[-1,:]
        xc_N     = np.concatenate([xc_list[i][-1,:] for i in range(int(self.nq))])
        scxc_N   = np.concatenate([scxc_list[i][-1,:] for i in range(int(self.nq))])
        dscxL_updatedp[self.N]= self.dscxL_updatedp_fn(xl0=xl_N,scxl0=scxl_N,a0=i_admm)['dscxL_updatedp_f'].full()
        dscxC_updatedp[self.N]= self.dscxC_updatedp_fn(xc0=xc_N,scxc0=scxc_N,a0=i_admm)['dscxC_updatedp_f'].full()

        auxSys3 = {
            "dscxL_updatedp":dscxL_updatedp,
            "dscuL_updatedp":dscuL_updatedp,
            "dscxC_updatedp":dscxC_updatedp,
            "dscuC_updatedp":dscuC_updatedp
        }
        return auxSys3


    def ADMM_forward_MPC(self,Ref_xl,Ref_ul,ref_xq,ref_uq,xl_fb,xq_fb,paral,paraC,max_iter_ADMM,adaptiveADMM):
        # initial guess of the safe copy variable trajectories
        scxl_traj_tp = np.zeros(((self.N+1)*self.nxl))
        scul_traj_tp = Ref_ul
        for k in range(self.N):
            scxl_traj_tp[k*self.nxl:(k+1)*self.nxl] = np.reshape(self.model_l_fn(xl0=scxl_traj_tp[k*self.nxl:(k+1)*self.nxl],ul0=Ref_ul[k*self.nul:(k+1)*self.nul])['mdynlf'].full(),self.nxl)
        scxl_traj_tp[self.N*self.nxl:(self.N+1)*self.nxl] = Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl]
        scxc_traj_tp = [np.zeros((self.N+1)*self.nxi) for _ in range(int(self.nq))]
        scuc_traj_tp = [np.zeros(self.N*self.nui)  for _ in range(int(self.nq))]
        for i in range(int(self.nq)):
            for k in range(self.N):
                scxc_traj_tp[i][k*self.nxi:(k+1)*self.nxi] = np.reshape(self.model_i_fn(xi0=scxc_traj_tp[i][k*self.nxi:(k+1)*self.nxi],ui0=ref_uq[i*self.nui:(i+1)*self.nui])['mdynif'].full(),self.nxi)   #ref_xq[i][k*self.nxi:(k+1)*self.nxi]
                scuc_traj_tp[i][k*self.nui:(k+1)*self.nui] = ref_uq[i*self.nui:(i+1)*self.nui]
            scxc_traj_tp[i][self.N*self.nxi:(self.N+1)*self.nxi] = ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi]
        # initial guess of the Lagrangian multiplier trajectories
        scxL_traj_tp = np.zeros(((self.N+1)*self.nxl)) # 1D array
        scuL_traj_tp = np.zeros((self.N*self.nul)) # 1D array
        scxC_traj_tp = [np.zeros(((self.N+1)*self.nxi)) for _ in range(int(self.nq))] # list of 1D array
        scuC_traj_tp = [np.zeros(self.N*self.nui)  for _ in range(int(self.nq))]
        scxL_traj    = np.zeros((self.N+1,self.nxl))
        scuL_traj    = np.zeros((self.N,self.nul))
        scxC_traj    = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        scuC_traj    = [np.zeros((self.N,self.nui))  for _ in range(int(self.nq))]
        max_iter     = 10 # 5 for training
        e_tol        = 1e-2 # 1e-2 for training
        Opt_Sol1_l = []
        Opt_Sol1_cddp = []
        Opt_Sol1_c = []
        Opt_Sol2   = []
        Opt_Sol3   = []
        self.max_iter_ADMM = max_iter_ADMM
        # initial guess for Subproblem2 IPOPT
        scxl0   = Ref_xl
        scul0   = Ref_ul
        scxc0   = np.zeros((self.N+1)*int(self.nq)*self.nxi)
        scuc0   = np.zeros(self.N*int(self.nq)*self.nxi)
        for k in range(self.N):
            ref_xc_k    = np.zeros(int(self.nq)*self.nxi)
            for i in range(int(self.nq)):
                ref_xc_k[i*self.nxi:(i+1)*self.nxi]    = ref_xq[i][k*self.nxi:(k+1)*self.nxi]
            scxc0[k*int(self.nq)*self.nxi:(k+1)*int(self.nq)*self.nxi]    = ref_xc_k
            scuc0[k*int(self.nq)*self.nui:(k+1)*int(self.nq)*self.nui]    = ref_uq
        ref_xc_N      = np.zeros(int(self.nq)*self.nxi)
        for i in range(int(self.nq)):
            ref_xc_N[i*self.nxi:(i+1)*self.nxi]    = ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi]
        scxc0[self.N*int(self.nq)*self.nxi:(self.N+1)*int(self.nq)*self.nxi]    = ref_xc_N
        Iadmm    = 0 # admm iteration index
        for i_admm in range(self.max_iter_ADMM):
            if adaptiveADMM == 'f':
                i_admm = 1e2 # a very large i_admm makes the sigmoid function almost 1
            # solve Subproblem 1-load (dynamic)
            start_time = TM.time()
            # opt_sol = self.MPC_Load_Planning_SubP1(Paral)
            opt_sol_l = self.DDP_Load_ADMM_Subp1(xl_fb,Ref_xl,Ref_ul,paral,scxl_traj_tp,scul_traj_tp,scxL_traj_tp,scuL_traj_tp,max_iter,e_tol,Iadmm,i_admm)
            mpctime = (TM.time() - start_time)*1000
            print('ADMM_iteration=',Iadmm+1,"subprblem1_load:--- %s ms ---" % format(mpctime,'.2f'))
            xl_traj = opt_sol_l['xl_traj']
            ul_traj = opt_sol_l['ul_traj']
            Kfbl_traj  = opt_sol_l['K_FB']
            xl_traj_tp = np.reshape(xl_traj,(self.N+1)*self.nxl)
            ul_traj_tp = np.reshape(ul_traj,self.N*self.nul)
            # solve Subproblem 1-cable (dynamic, across n cables)
            ParaC   = []
            for i in range(int(self.nq)):
                xi_fb  = xq_fb[i*self.nxi:(i+1)*self.nxi]
                ref_xi = ref_xq[i]
                ref_ui = ref_uq[i*self.nui:(i+1)*self.nui]
                scxi_traj = scxc_traj_tp[i]
                scxI_traj = scxC_traj_tp[i]
                scui_traj = scuc_traj_tp[i]
                scuI_traj = scuC_traj_tp[i]
                parai     = np.concatenate((xi_fb,ref_xi))
                parai     = np.concatenate((parai,ref_ui))
                parai     = np.concatenate((parai,scxi_traj))
                parai     = np.concatenate((parai,scxI_traj))
                parai     = np.concatenate((parai,scui_traj))
                parai     = np.concatenate((parai,scuI_traj))
                parai     = np.concatenate((parai,paraC))
                parai     = np.concatenate((parai,[Iadmm,i_admm]))
                ParaC  += [parai]
            start_time = TM.time()
            opt_solc, OPt_sol_c = self.MPC_Cable_DDP_Planning_SubP1(ParaC)
            mpctime = (TM.time() - start_time)*1000
            dis_rn     = self.Discount_rate(i_admm)
            print('ADMM_iteration=',Iadmm+1,"subproblem1_cables:--- %s ms ---" % format(mpctime,'.2f'),'current_pl=',dis_rn*paral[-1],'current_pi=',dis_rn*paraC[-1])
            xc_traj  = opt_solc['xc_traj']
            uc_traj  = opt_solc['uc_traj']
            # solve Subproblem 2 (static, N independent steps, each step is a centralized problem)
            xc_traj_tp2 = np.zeros((self.N+1)*int(self.nq)*self.nxi)
            uc_traj_tp2 = np.zeros(self.N*int(self.nq)*self.nui)
            for k in range(self.N):
                xc_traj_k   = np.zeros(int(self.nq)*self.nxi)
                uc_traj_k   = np.zeros(int(self.nq)*self.nui)
                for i in range(int(self.nq)):
                    xc_traj_k[i*self.nxi:(i+1)*self.nxi] = xc_traj[i][k,:]
                    uc_traj_k[i*self.nui:(i+1)*self.nui] = uc_traj[i][k,:]
                xc_traj_tp2[k*int(self.nq)*self.nxi:(k+1)*int(self.nq)*self.nxi] = xc_traj_k
                uc_traj_tp2[k*int(self.nq)*self.nui:(k+1)*int(self.nq)*self.nui] = uc_traj_k
            xc_traj_N   = np.zeros(int(self.nq)*self.nxi)
            for i in range(int(self.nq)):
                xc_traj_N[i*self.nxi:(i+1)*self.nxi] = xc_traj[i][self.N,:]
            xc_traj_tp2[self.N*int(self.nq)*self.nxi:(self.N+1)*int(self.nq)*self.nxi] = xc_traj_N
            scxC_traj_tp2 = np.zeros((self.N+1)*int(self.nq)*self.nxi)
            ref_xc_tp2    = np.zeros((self.N+1)*int(self.nq)*self.nxi)
            scuC_traj_tp2 = np.zeros(self.N*int(self.nq)*self.nui)
            for k in range(self.N):
                scxC_traj_k = np.zeros(int(self.nq)*self.nxi)
                ref_xc_k    = np.zeros(int(self.nq)*self.nxi)
                scuC_traj_k = np.zeros(int(self.nq)*self.nui)
                for i in range(int(self.nq)):
                    scxC_traj_k[i*self.nxi:(i+1)*self.nxi] = scxC_traj_tp[i][k*self.nxi:(k+1)*self.nxi]
                    ref_xc_k[i*self.nxi:(i+1)*self.nxi]    = ref_xq[i][k*self.nxi:(k+1)*self.nxi]
                    scuC_traj_k[i*self.nui:(i+1)*self.nui] = scuC_traj_tp[i][k*self.nui:(k+1)*self.nui]
                scxC_traj_tp2[k*int(self.nq)*self.nxi:(k+1)*int(self.nq)*self.nxi] = scxC_traj_k
                ref_xc_tp2[k*int(self.nq)*self.nxi:(k+1)*int(self.nq)*self.nxi]    = ref_xc_k
                scuC_traj_tp2[k*int(self.nq)*self.nui:(k+1)*int(self.nq)*self.nui] = scuC_traj_k
            scxC_traj_N   = np.zeros(int(self.nq)*self.nxi)
            ref_xc_N      = np.zeros(int(self.nq)*self.nxi)
            for i in range(int(self.nq)):
                scxC_traj_N[i*self.nxi:(i+1)*self.nxi] = scxC_traj_tp[i][self.N*self.nxi:(self.N+1)*self.nxi]
                ref_xc_N[i*self.nxi:(i+1)*self.nxi]    = ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi]
            scxC_traj_tp2[self.N*int(self.nq)*self.nxi:(self.N+1)*int(self.nq)*self.nxi] = scxC_traj_N
            ref_xc_tp2[self.N*int(self.nq)*self.nxi:(self.N+1)*int(self.nq)*self.nxi]    = ref_xc_N
            Para2_cable = np.concatenate((scxl0,scul0))
            Para2_cable = np.concatenate((Para2_cable,scxc0))
            Para2_cable = np.concatenate((Para2_cable,ref_uq))
            Para2_cable = np.concatenate((Para2_cable,xl_traj_tp))
            Para2_cable = np.concatenate((Para2_cable,scxL_traj_tp))
            Para2_cable = np.concatenate((Para2_cable,ul_traj_tp))
            Para2_cable = np.concatenate((Para2_cable,scuL_traj_tp))
            Para2_cable = np.concatenate((Para2_cable,xc_traj_tp2))
            Para2_cable = np.concatenate((Para2_cable,scxC_traj_tp2))
            Para2_cable = np.concatenate((Para2_cable,uc_traj_tp2))
            Para2_cable = np.concatenate((Para2_cable,scuC_traj_tp2))
            Para2_cable = np.concatenate((Para2_cable,paral))
            Para2_cable = np.concatenate((Para2_cable,paraC))
            Para2_cable = np.concatenate((Para2_cable,[i_admm]))
            start_time  = TM.time()
            opt_sol2    = self.ADMM_SubP2(Para2_cable)
            mpctime = (TM.time() - start_time)*1000
            print("subproblem2:--- %s ms ---" % format(mpctime,'.2f'))
            scxl_traj   = opt_sol2['scxl_traj']
            scul_traj   = opt_sol2['scul_traj']
            scxc_traj   = opt_sol2['scxc_traj']
            scuc_traj   = opt_sol2['scuc_traj']
            # solve Subproblem 3
            opt_sol3    = self.ADMM_SubP3(xl_traj,scxl_traj,scxL_traj,ul_traj,scul_traj,scuL_traj,xc_traj,scxc_traj,scxC_traj,uc_traj,scuc_traj,scuC_traj,paral[-1],paraC[-1],i_admm)
            scxL_traj   = opt_sol3['scxL_traj_new']
            scuL_traj   = opt_sol3['scuL_traj_new']
            scxC_traj   = opt_sol3['scxC_traj_new']
            scuC_traj   = opt_sol3['scuC_traj_new']
            # update trajectories
            scxl_traj_tp_new  = np.reshape(scxl_traj,(self.N+1)*self.nxl) # for Subproblem 1-load
            scul_traj_tp_new  = np.reshape(scul_traj,self.N*self.nul) # for Subproblem 1-load
            scxL_traj_tp  = np.reshape(scxL_traj,(self.N+1)*self.nxl) # for Subproblem 1-load and 2
            scuL_traj_tp  = np.reshape(scuL_traj,self.N*self.nul) # for Subproblem 1-load and 2
            xc_traj_tp    = [np.zeros((self.N+1)*self.nxi)  for _ in range(int(self.nq))] # for Subproblem 2-cable and 3
            uc_traj_tp    = [np.zeros(self.N*self.nui)  for _ in range(int(self.nq))]     # for Subproblem 2-cable and 3
            scxc_traj_tp_new  = [np.zeros((self.N+1)*self.nxi)  for _ in range(int(self.nq))] # for Subproblem 1-cable
            scxC_traj_tp  = [np.zeros((self.N+1)*self.nxi)  for _ in range(int(self.nq))] # for Subproblem 1-cable and 2
            scuc_traj_tp_new  = [np.zeros(self.N*self.nui)  for _ in range(int(self.nq))] # for Subproblem 1-cable
            scuC_traj_tp  = [np.zeros(self.N*self.nui)  for _ in range(int(self.nq))] # for Subproblem 1-cable and 2
            r_xc = [] # primal residual of cables' states
            r_uc = [] # primal residual of cables' controls
            s_xc = [] # dual residual of cables' states
            s_uc = [] # dual residual of cables' contorls
            for i in range(int(self.nq)):
                for k in range(self.N):
                    xc_traj_tp[i][k*self.nxi:(k+1)*self.nxi]       = xc_traj[i][k,:]
                    uc_traj_tp[i][k*self.nui:(k+1)*self.nui]       = uc_traj[i][k,:]
                    scxc_traj_tp_new[i][k*self.nxi:(k+1)*self.nxi] = scxc_traj[i][k,:]
                    scxC_traj_tp[i][k*self.nxi:(k+1)*self.nxi]     = scxC_traj[i][k,:]
                    scuc_traj_tp_new[i][k*self.nui:(k+1)*self.nui] = scuc_traj[i][k,:]
                    scuC_traj_tp[i][k*self.nui:(k+1)*self.nui]     = scuC_traj[i][k,:]
                xc_traj_tp[i][self.N*self.nxi:(self.N+1)*self.nxi]       = xc_traj[i][self.N,:]
                scxc_traj_tp_new[i][self.N*self.nxi:(self.N+1)*self.nxi] = scxc_traj[i][self.N,:]
                scxC_traj_tp[i][self.N*self.nxi:(self.N+1)*self.nxi]     = scxC_traj[i][self.N,:]
                r_xc += [LA.norm(xc_traj_tp[i]-scxc_traj_tp_new[i])]
                s_xc += [paral[-1]*LA.norm(scxc_traj_tp_new[i]-scxc_traj_tp[i])]
                r_uc += [LA.norm(uc_traj_tp[i]-scuc_traj_tp_new[i])]
                s_uc += [paral[-1]*LA.norm(scuc_traj_tp_new[i]-scuc_traj_tp[i])]
            Iadmm += 1
            # update the initial guess
            scxl0 = scxl_traj_tp_new 
            scul0 = scul_traj_tp_new
            scxc0 = np.zeros((self.N+1)*int(self.nq)*self.nxi)
            scuc0 = np.zeros((self.N)*int(self.nq)*self.nui)
            for k in range(self.N):
                scxc_k = np.zeros(int(self.nq)*self.nxi)
                scuc_k = np.zeros(int(self.nq)*self.nui)
                for i in range(int(self.nq)):
                    scxc_k[i*self.nxi:(i+1)*self.nxi]=scxc_traj[i][k,:]
                    scuc_k[i*self.nui:(i+1)*self.nui]=scuc_traj[i][k,:]
                scxc0[k*self.nxi*int(self.nq):(k+1)*self.nxi*int(self.nq)]=scxc_k
                scuc0[k*self.nui*int(self.nq):(k+1)*self.nui*int(self.nq)]=scuc_k

            # residuals
            r_xl = LA.norm(xl_traj_tp - scxl_traj_tp_new)
            r_ul = LA.norm(ul_traj_tp - scul_traj_tp_new)
            s_xl = paral[-1]*LA.norm(scxl_traj_tp_new - scxl_traj_tp)
            s_ul = paral[-1]*LA.norm(scul_traj_tp_new - scul_traj_tp)
            # print('ADMM_iteration=',i_ADMM,'p=',paral[-1],'r_xl=',r_xl,'r_ul=',r_ul,'s_xl=',s_xl,'s_ul=',s_ul)
            # for i in range(int(self.nq)):
            #     print('ADMM_iteration=',i_ADMM,'r_xc_'+str(i+1)+'=',r_xc[i],'s_xc_'+str(i+1)+'=',s_xc[i])
            #     print('ADMM_iteration=',i_ADMM,'r_uc_'+str(i+1)+'=',r_uc[i],'s_uc_'+str(i+1)+'=',s_uc[i])
            # update
            scxl_traj_tp = scxl_traj_tp_new
            scul_traj_tp = scul_traj_tp_new
            scxc_traj_tp = scxc_traj_tp_new
            scuc_traj_tp = scuc_traj_tp_new

            Opt_Sol1_l += [opt_sol_l]
            Opt_Sol1_cddp += [OPt_sol_c]
            Opt_Sol1_c += [opt_solc]
            Opt_Sol2   += [opt_sol2]
            Opt_Sol3   += [opt_sol3]
        
        opt_sol = {"xl_traj":xl_traj,
                   "Kfbl_traj":Kfbl_traj,
                   "scxl_traj":scxl_traj,
                   "xc_traj":xc_traj,
                   "uc_traj":uc_traj,
                   "scxc_traj":scxc_traj,
                    }
        
        return opt_sol, Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3
    

            
    def DDP_Load_Gradient(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, p, i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        dis_rn     = self.Discount_rate(i_admm)
        S          = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        S[self.N]  = HxNp + scxL_grad[self.N] - dis_rn*p*scxl_grad[self.N] # reduced to HxNp only in the single-agent problem
        v_FF       = self.N*[np.zeros((self.nul,self.n_Pauto))]
        xl_grad    = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] 
        ul_grad    = self.N*[np.zeros((self.nul,self.n_Pauto))]
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): 
            Hxp_k    = Hxp[k] + scxL_grad[k] - dis_rn*p*scxl_grad[k]
            Hup_k    = Hup[k] + scuL_grad[k] - dis_rn*p*scul_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            ul_grad[k]  = K_fb[k]@xl_grad[k]+v_FF[k]
            xl_grad[k+1]= F[k]@xl_grad[k]+G[k]@ul_grad[k]

        grad_outl ={"xl_grad":xl_grad,
                   "ul_grad":ul_grad
                }
        
        return grad_outl
    
    
    def Cao_Load_Gradient_s(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, p, i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        dis_rn      = self.Discount_rate(i_admm)
        S           = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] # Vxp
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        S[self.N]   = HxNp + scxL_grad[self.N] - dis_rn*p*scxl_grad[self.N]
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        v_FF        = self.N*[np.zeros((self.nul,self.n_Pauto))]
        xl_grad     = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] 
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        ul_grad     = self.N*[np.zeros((self.nul,self.n_Pauto))]
        
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): # N-1, N-2,...,0
            Hpp_k    = np.zeros((self.n_Pauto,self.n_Pauto))
            Hxp_k    = Hxp[k] + scxL_grad[k] - dis_rn*p*scxl_grad[k]
            Hup_k    = Hup[k] + scuL_grad[k] - dis_rn*p*scul_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            Vpp[k]   = Hpp_k + Vpp[k+1] + (Hup_k + G[k].T@S[k+1]).T@v_FF[k] # the augmented Riccati recursion, which is redundant
            S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            ul_grad[k]  = K_fb[k]@xl_grad[k]+v_FF[k]@p_grad[k] # expanding the augmented control law gives this form, which is exactly the same as ours
            xl_grad[k+1]= F[k]@xl_grad[k]+G[k]@ul_grad[k]
            p_grad[k+1] = p_grad[k] # the augmented dynamics, which is redundant
        
        grad_out_cao ={"xl_grad":xl_grad,
                   "ul_grad":ul_grad,
                   "p_grad":p_grad
                }
        
        return grad_out_cao
    
    def Cao_Load_Gradient(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, p,i_admm):
        # solve the augmented optimal problem using one-step DDP recursion
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysl['HxxN'], auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        # Vyy      = (self.N+1)*[np.zeros((self.n_Pauto+self.nxl,self.n_Pauto+self.nxl))] # a large matrix, leading to significant computation cost
        # we decompose Vyy into four smaller blocks
        dis_rn      = self.Discount_rate(i_admm)
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        Vpx         = (self.N+1)*[np.zeros((self.n_Pauto,self.nxl))]
        Vxp         = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        Vxx         = (self.N+1)*[np.zeros((self.nxl,self.nxl))]
        # Kfb_y    = self.N*[np.zeros((self.nul,self.n_Pauto+self.n_xl))] # augmented feedback gain
        Kfb_p       = self.N*[np.zeros((self.nul,self.n_Pauto))] # this matches exactly the feedforward gain!
        Kfb_x       = self.N*[np.zeros((self.nul,self.nxl))]    # this is the feedback gain
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        xl_grad     = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] 
        ul_grad     = self.N*[np.zeros((self.nul,self.n_Pauto))]
        # Vyy[self.N] = vertcat(
        #                         horzcat(np.zeros((self.n_Pauto,self.n_Pauto)),HxNp.T),
        #                         horzcat(HxNp,self.lxxN_fn(P1l0=weight1)['lxxNf'].full())
        #                     )
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        Vpx[self.N] = (HxNp + scxL_grad[self.N] - dis_rn*p*scxl_grad[self.N]).T
        Vxp[self.N] = HxNp + scxL_grad[self.N] - dis_rn*p*scxl_grad[self.N]
        Vxx[self.N] = HxxN
        
        for k in reversed(range(self.N)):
            # Hyy_k   = vertcat(
            #     horzcat(np.zeros((self.n_Pauto,self.n_Pauto)),(Hxp[k]+ scxL_grad[k] - p1*scxl_grad[k]).T),
            #     horzcat((Hxp[k]+ scxL_grad[k] - p1*scxl_grad[k]),Hxx[k])
            # )
            # F_bar   = vertcat(
            #     horzcat(np.identity(self.n_Pauto),np.zeros((self.n_Pauto,self.n_xl))),
            #     horzcat(np.zeros((self.n_Pauto,self.n_xl)).T,F[k])
            # )
            # G_bar   = vertcat(np.zeros((self.n_Pauto,self.n_Wl)),G[k])
            # Huy_k   = horzcat((Hup[k]+ scWL_grad[k] - p1*scWl_grad[k]),Hxu[k].T)
            # Qyy_k   = Hyy_k + F_bar.T@Vyy[k+1]@F_bar
            # Quy_k   = Huy_k + G_bar.T@Vyy[k+1]@F_bar
            # Quu_k   = Huu[k] + G_bar.T@Vyy[k+1]@G_bar
            # Kfb_y[k]=-LA.inv(Quu_k)@Quy_k
            # Vyy[k]  = Qyy_k + Quy_k.T@Kfb_y[k]
            # Hpp_k    = np.zeros((self.n_Pauto,self.n_Pauto))
            Hpx_k    = (Hxp[k]+ scxL_grad[k] - dis_rn*p*scxl_grad[k]).T
            # Hxp_k    = Hxp[k]+ scxL_grad[k] - dis_rn*p1*scxl_grad[k]
            Hxx_k    = Hxx[k]
            Hup_k    = Hup[k]+ scuL_grad[k] - dis_rn*p*scul_grad[k]
            Quu_k    = Huu[k]+G[k].T@Vxx[k+1]@G[k] 
            invQuu_k = LA.inv(Quu_k)
            Kfb_p[k] = -invQuu_k@(Hup_k+G[k].T@Vxp[k+1]) 
            Kfb_x[k] = -invQuu_k@(Hxu[k].T+G[k].T@Vxx[k+1]@F[k])
            Vpp[k]   = Vpp[k+1] + (Hup_k.T+Vpx[k+1]@G[k])@Kfb_p[k]
            Vpx[k]   = Hpx_k + Vpx[k+1]@F[k] + Kfb_p[k].T@(Hxu[k]+F[k].T@Vxx[k+1]@G[k]).T
            # Vxp[k]   = Hxp_k + F[k].T@Vxp[k+1] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_p[k]
            Vxp[k]   = Vpx[k].T
            Vxx[k]   = Hxx_k + F[k].T@Vxx[k+1]@F[k] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_x[k]

        for k in range(self.N):
            ul_grad[k]   = Kfb_p[k]@p_grad[k] + Kfb_x[k]@xl_grad[k]
            xl_grad[k+1] = F[k]@xl_grad[k]+G[k]@ul_grad[k]
            p_grad[k+1]  = p_grad[k]
        grad_out_cao ={"xl_grad":xl_grad,
                   "ul_grad":ul_grad,
                   "p_grad":p_grad
                }
        
        return grad_out_cao
    

    def PDP_Load_Gradient(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, p, i_admm):
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysl['HxxN'], auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        dis_rn     = self.Discount_rate(i_admm)
        P          = (self.N+1)*[np.zeros((self.nxl,self.nxl))]
        S          = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        A          = self.N*[np.zeros((self.nxl,self.nxl))]
        R          = self.N*[np.zeros((self.nxl,self.nxl))]
        M_p        = self.N*[np.zeros((self.nxl,self.n_Pauto))]
        invHuu     = self.N*[np.zeros((self.nul,self.nul))]
        PinvIRP    = self.N*[np.zeros((self.nxl,self.nxl))]
        P[self.N]  = HxxN
        S[self.N]  = HxNp  + scxL_grad[self.N] - dis_rn*p*scxl_grad[self.N]
        xl_grad    = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] 
        ul_grad    = self.N*[np.zeros((self.nul,self.n_Pauto))]
        I          = np.identity(self.nxl)
        Iu         = np.identity(self.nul)
        for k in reversed(range(self.N)):# N-1, N-2,...,0
            P_next      = P[k+1]
            S_next      = S[k+1]
            invHuu[k]   = LA.inv(Huu[k])
            GinvHuu     = G[k]@invHuu[k]
            HxuinvHuu   = Hxu[k]@invHuu[k]
            A[k]        = F[k]-GinvHuu@Hxu[k].T
            R[k]        = GinvHuu@G[k].T
            M_p[k]      = -GinvHuu@(Hup[k] + scuL_grad[k] - dis_rn*p*scul_grad[k])
            Q_k         = Hxx[k]-HxuinvHuu@Hxu[k].T
            N_p_k       = Hxp[k]+ scxL_grad[k] - dis_rn*p*scxl_grad[k] - HxuinvHuu@(Hup[k] + scuL_grad[k] - dis_rn*p*scul_grad[k])
            PinvIRP[k]  = P_next@LA.inv(I+R[k]@P_next)
            P_curr      = Q_k + A[k].T@PinvIRP[k]@A[k]
            S_curr      = A[k].T@PinvIRP[k]@(M_p[k] - R[k]@S_next) + A[k].T@S_next + N_p_k
            P[k]        = P_curr
            S[k]        = S_curr
        
        for k in range(self.N):
            ul_grad[k]  = -invHuu[k]@((Hxu[k].T+G[k].T@PinvIRP[k]@A[k])@xl_grad[k] + G[k].T@PinvIRP[k]@(M_p[k]- R[k]@ S[k+1]) + G[k].T@S[k+1] + (Hup[k] + scuL_grad[k] - dis_rn*p*scul_grad[k]))
            xl_grad[k+1] = F[k]@xl_grad[k] + G[k]@ul_grad[k]

        grad_out ={"xl_grad":xl_grad,
                   "ul_grad":ul_grad
                }
        
        return grad_out
    

    
    def DDP_Cable_Gradient(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pi, i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        dis_rn     = self.Discount_rate(i_admm)
        S          = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))]
        S[self.N]  = HxNp + scxI_grad[self.N] - dis_rn*pi*scxi_grad[self.N] # reduced to HxNp only in the single-agent problem
        v_FF       = self.N*[np.zeros((self.nui,self.n_Pauto))]
        xi_grad    = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))] 
        ui_grad    = self.N*[np.zeros((self.nui,self.n_Pauto))]
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): 
            Hxp_k    = Hxp[k] + scxI_grad[k] - dis_rn*pi*scxi_grad[k]
            Hup_k    = Hup[k] + scuI_grad[k] - dis_rn*pi*scui_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            ui_grad[k]  = K_fb[k]@xi_grad[k]+v_FF[k]
            xi_grad[k+1]= F[k]@xi_grad[k]+G[k]@ui_grad[k]

        grad_outi ={"xi_grad":xi_grad,
                   "ui_grad":ui_grad
                }
        
        return grad_outi
    
    def Cao_Cable_Gradient_s(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pi, i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        dis_rn      = self.Discount_rate(i_admm)
        S           = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))] # Vxp
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        S[self.N]   = HxNp + scxI_grad[self.N] - dis_rn*pi*scxi_grad[self.N]
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        v_FF        = self.N*[np.zeros((self.nui,self.n_Pauto))]
        xi_grad     = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))] 
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        ui_grad     = self.N*[np.zeros((self.nui,self.n_Pauto))]
        
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): # N-1, N-2,...,0
            Hpp_k    = np.zeros((self.n_Pauto,self.n_Pauto))
            Hxp_k    = Hxp[k] + scxI_grad[k] - dis_rn*pi*scxi_grad[k]
            Hup_k    = Hup[k] + scuI_grad[k] - dis_rn*pi*scui_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            Vpp[k]   = Hpp_k + Vpp[k+1] + (Hup_k + G[k].T@S[k+1]).T@v_FF[k] # the augmented Riccati recursion, which is redundant
            S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            ui_grad[k]  = K_fb[k]@xi_grad[k]+v_FF[k]@p_grad[k] # expanding the augmented control law gives this form, which is exactly the same as ours
            xi_grad[k+1]= F[k]@xi_grad[k]+G[k]@ui_grad[k]
            p_grad[k+1] = p_grad[k] # the augmented dynamics, which is redundant
        
        grad_out_cao ={"xi_grad":xi_grad,
                   "ui_grad":ui_grad,
                   "p_grad":p_grad
                }
        
        return grad_out_cao

    def Cao_Cable_Gradient(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pi,i_admm):
        # solve the augmented optimal problem using one-step DDP recursion
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysi['HxxN'], auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        # Vyy      = (self.N+1)*[np.zeros((self.n_Pauto+self.nxl,self.n_Pauto+self.nxl))] # a large matrix, leading to significant computation cost
        # we decompose Vyy into four smaller blocks
        dis_rn      = self.Discount_rate(i_admm)
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        Vpx         = (self.N+1)*[np.zeros((self.n_Pauto,self.nxi))]
        Vxp         = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))]
        Vxx         = (self.N+1)*[np.zeros((self.nxi,self.nxi))]
        # Kfb_y    = self.N*[np.zeros((self.nul,self.n_Pauto+self.n_xl))] # augmented feedback gain
        Kfb_p       = self.N*[np.zeros((self.nui,self.n_Pauto))] # this matches exactly the feedforward gain!
        Kfb_x       = self.N*[np.zeros((self.nui,self.nxi))]    # this is the feedback gain
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        xi_grad     = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))] 
        ui_grad     = self.N*[np.zeros((self.nui,self.n_Pauto))]
        # Vyy[self.N] = vertcat(
        #                         horzcat(np.zeros((self.n_Pauto,self.n_Pauto)),HxNp.T),
        #                         horzcat(HxNp,self.lxxN_fn(P1l0=weight1)['lxxNf'].full())
        #                     )
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        Vpx[self.N] = (HxNp + scxI_grad[self.N] - dis_rn*pi*scxi_grad[self.N]).T
        Vxp[self.N] = HxNp + scxI_grad[self.N] - dis_rn*pi*scxi_grad[self.N]
        Vxx[self.N] = HxxN
        
       
        for k in reversed(range(self.N)):
            # Hyy_k   = vertcat(
            #     horzcat(np.zeros((self.n_Pauto,self.n_Pauto)),(Hxp[k]+ scxL_grad[k] - p1*scxl_grad[k]).T),
            #     horzcat((Hxp[k]+ scxL_grad[k] - p1*scxl_grad[k]),Hxx[k])
            # )
            # F_bar   = vertcat(
            #     horzcat(np.identity(self.n_Pauto),np.zeros((self.n_Pauto,self.n_xl))),
            #     horzcat(np.zeros((self.n_Pauto,self.n_xl)).T,F[k])
            # )
            # G_bar   = vertcat(np.zeros((self.n_Pauto,self.n_Wl)),G[k])
            # Huy_k   = horzcat((Hup[k]+ scWL_grad[k] - p1*scWl_grad[k]),Hxu[k].T)
            # Qyy_k   = Hyy_k + F_bar.T@Vyy[k+1]@F_bar
            # Quy_k   = Huy_k + G_bar.T@Vyy[k+1]@F_bar
            # Quu_k   = Huu[k] + G_bar.T@Vyy[k+1]@G_bar
            # Kfb_y[k]=-LA.inv(Quu_k)@Quy_k
            # Vyy[k]  = Qyy_k + Quy_k.T@Kfb_y[k]
            # Hpp_k    = np.zeros((self.n_Pauto,self.n_Pauto))
            Hpx_k    = (Hxp[k]+ scxI_grad[k] - dis_rn*pi*scxi_grad[k]).T
            # Hxp_k    = Hxp[k]+ scxL_grad[k] - dis_rn*p1*scxl_grad[k]
            Hxx_k    = Hxx[k]
            Hup_k    = Hup[k]+ scuI_grad[k] - dis_rn*pi*scui_grad[k]
            Quu_k    = Huu[k]+G[k].T@Vxx[k+1]@G[k]
            invQuu_k = LA.inv(Quu_k)
            Kfb_p[k] = -invQuu_k@(Hup_k+G[k].T@Vxp[k+1]) 
            Kfb_x[k] = -invQuu_k@(Hxu[k].T+G[k].T@Vxx[k+1]@F[k])
            Vpp[k]   = Vpp[k+1] + (Hup_k.T+Vpx[k+1]@G[k])@Kfb_p[k]
            Vpx[k]   = Hpx_k + Vpx[k+1]@F[k] + Kfb_p[k].T@(Hxu[k]+F[k].T@Vxx[k+1]@G[k]).T
            # Vxp[k]   = Hxp_k + F[k].T@Vxp[k+1] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_p[k]
            Vxp[k]   = Vpx[k].T
            Vxx[k]   = Hxx_k + F[k].T@Vxx[k+1]@F[k] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_x[k]

        for k in range(self.N):
            ui_grad[k]   = Kfb_p[k]@p_grad[k] + Kfb_x[k]@xi_grad[k]
            xi_grad[k+1] = F[k]@xi_grad[k]+G[k]@ui_grad[k]
            p_grad[k+1]  = p_grad[k]
        grad_out_cao ={"xi_grad":xi_grad,
                   "ui_grad":ui_grad,
                   "p_grad":p_grad
                }
        
        return grad_out_cao
    

    
    def PDP_Cable_Gradient(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pi, i_admm):
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysi['HxxN'], auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        dis_rn     = self.Discount_rate(i_admm)
        P          = (self.N+1)*[np.zeros((self.nxi,self.nxi))]
        S          = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))]
        A          = self.N*[np.zeros((self.nxi,self.nxi))]
        R          = self.N*[np.zeros((self.nxi,self.nxi))]
        M_p        = self.N*[np.zeros((self.nxi,self.n_Pauto))]
        invHuu     = self.N*[np.zeros((self.nui,self.nui))]
        PinvIRP    = self.N*[np.zeros((self.nxi,self.nxi))]
        P[self.N]  = HxxN
        S[self.N]  = HxNp  + scxI_grad[self.N] - dis_rn*pi*scxi_grad[self.N]
        xi_grad    = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))] 
        ui_grad    = self.N*[np.zeros((self.nui,self.n_Pauto))]
        I          = np.identity(self.nxi)
        for k in reversed(range(self.N)):
            P_next      = P[k+1]
            S_next      = S[k+1]
            invHuu[k]   = LA.inv(Huu[k])
            GinvHuu     = G[k]@invHuu[k]
            HxuinvHuu   = Hxu[k]@invHuu[k]
            A[k]        = F[k]-GinvHuu@Hxu[k].T
            R[k]        = GinvHuu@G[k].T
            M_p[k]      = -GinvHuu@(Hup[k] + scuI_grad[k] - dis_rn*pi*scui_grad[k])
            Q_k         = Hxx[k]-HxuinvHuu@Hxu[k].T
            N_p_k       = Hxp[k]+ scxI_grad[k] - dis_rn*pi*scxi_grad[k] - HxuinvHuu@(Hup[k] + scuI_grad[k] - dis_rn*pi*scui_grad[k])
            PinvIRP[k]  = P_next@LA.inv(I+R[k]@P_next)
            P_curr      = Q_k + A[k].T@PinvIRP[k]@A[k]
            S_curr      = A[k].T@PinvIRP[k]@(M_p[k] - R[k]@S_next) + A[k].T@S_next + N_p_k
            P[k]        = P_curr
            S[k]        = S_curr
        
        for k in range(self.N):
            ui_grad[k]  = -invHuu[k]@((Hxu[k].T+G[k].T@PinvIRP[k]@A[k])@xi_grad[k] + G[k].T@PinvIRP[k]@(M_p[k]- R[k]@ S[k+1]) + G[k].T@S[k+1] + (Hup[k] + scuI_grad[k] - dis_rn*pi*scui_grad[k]))
            xi_grad[k+1] = F[k]@xi_grad[k] + G[k]@ui_grad[k]

        grad_out ={"xi_grad":xi_grad,
                   "ui_grad":ui_grad
                }
        
        return grad_out
    
    

    def SubP2_Gradient(self,auxSys2,grad_outl,grad_outc,scxL_grad,scuL_grad,scxC_grad,scuC_grad,p,pi,i_admm):
        xl_grad      = grad_outl['xl_grad']
        ul_grad      = grad_outl['ul_grad']
        Lscxlscxl    = auxSys2['Lscxlscxl']
        Lscxlscul    = auxSys2['Lscxlscul']
        Lscxlscxc    = auxSys2['Lscxlscxc']
        Lscxlscuc    = auxSys2['Lscxlscuc']
        Lsculscul    = auxSys2['Lsculscul']
        Lsculscxc    = auxSys2['Lsculscxc']
        Lsculscuc    = auxSys2['Lsculscuc']
        Lscxcscxc    = auxSys2['Lscxcscxc']
        Lscxcscuc    = auxSys2['Lscxcscuc']
        Lscucscuc    = auxSys2['Lscucscuc']
        Lscxlp       = auxSys2['Lscxlp']
        Lsculp       = auxSys2['Lsculp']
        Lscxcp       = auxSys2['Lscxcp']
        Lscucp       = auxSys2['Lscucp']
        # hessians of the original Lagrangian
        Lscxlscxl_o  = auxSys2['Lscxlscxl_o']
        Lscxlscul_o  = auxSys2['Lscxlscul_o']
        Lscxlscxc_o  = auxSys2['Lscxlscxc_o']
        Lscxlscuc_o  = auxSys2['Lscxlscuc_o']
        Lsculscul_o  = auxSys2['Lsculscul_o']
        Lsculscxc_o  = auxSys2['Lsculscxc_o']
        Lsculscuc_o  = auxSys2['Lsculscuc_o']
        Lscxcscxc_o  = auxSys2['Lscxcscxc_o']
        Lscxcscuc_o  = auxSys2['Lscxcscuc_o']
        Lscucscuc_o  = auxSys2['Lscucscuc_o']
        I_hessian    = np.identity(self.nxl+self.nul+(self.nxi+self.nui)*int(self.nq))
        I_hess2      = np.identity(self.nxl+self.nxi*int(self.nq))
        scxl_grad    = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        scul_grad    = self.N*[np.zeros((self.nul,self.n_Pauto))]
        scxc_grad    = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        scuc_grad    = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        MIN_eigen    = []
        dis_rn       = self.Discount_rate(i_admm)
        for k in range(self.N):
            L_hessian_k = vertcat(
                                horzcat(Lscxlscxl[k],   Lscxlscul[k],   Lscxlscxc[k],   Lscxlscuc[k]),
                                horzcat(Lscxlscul[k].T, Lsculscul[k],   Lsculscxc[k],   Lsculscuc[k]),
                                horzcat(Lscxlscxc[k].T, Lsculscxc[k].T, Lscxcscxc[k],   Lscxcscuc[k]),
                                horzcat(Lscxlscuc[k].T, Lsculscuc[k].T, Lscxcscuc[k].T, Lscucscuc[k])
                                )
            xl_grad_k   = xl_grad[k]
            ul_grad_k   = ul_grad[k]
            scxL_grad_k = scxL_grad[k]
            scuL_grad_k = scuL_grad[k]
            scxC_grad_k = scxC_grad[k]
            scuC_grad_k = scuC_grad[k]
            xc_grad_k   = grad_outc[0]['xi_grad'][k]
            uc_grad_k   = grad_outc[0]['ui_grad'][k]
            for i in range(1,int(self.nq)):
                xc_grad_k = np.vstack((xc_grad_k,grad_outc[i]['xi_grad'][k]))
                uc_grad_k = np.vstack((uc_grad_k,grad_outc[i]['ui_grad'][k]))
            L_trajp_k   = vertcat(
                                horzcat(Lscxlp[k] - dis_rn*p*xl_grad_k - scxL_grad_k),
                                horzcat(Lsculp[k] - dis_rn*p*ul_grad_k - scuL_grad_k),
                                horzcat(Lscxcp[k] - dis_rn*pi*xc_grad_k - scxC_grad_k),
                                horzcat(Lscucp[k] - dis_rn*pi*uc_grad_k - scuC_grad_k)
                                )
            L_hessian_ko = vertcat(
                                horzcat(Lscxlscxl_o[k],   Lscxlscul_o[k],   Lscxlscxc_o[k],   Lscxlscuc_o[k]),
                                horzcat(Lscxlscul_o[k].T, Lsculscul_o[k],   Lsculscxc_o[k],   Lsculscuc_o[k]),
                                horzcat(Lscxlscxc_o[k].T, Lsculscxc_o[k].T, Lscxcscxc_o[k],   Lscxcscuc_o[k]),
                                horzcat(Lscxlscuc_o[k].T, Lsculscuc_o[k].T, Lscxcscuc_o[k].T, Lscucscuc_o[k])
                                )       
            min_eigval = np.min(LA.eigvalsh(L_hessian_ko))
            MIN_eigen += [min_eigval]
            if min_eigval<0:
                reg = -min_eigval+1e-4
            else:
                reg = 0
            L_hessian_k_sym = L_hessian_k + reg*I_hessian
            L, _jitter      = self.try_cholesky(L_hessian_k_sym, jitter0=0.0)
            grad_subp2_k    = self.chol_solve(L, -L_trajp_k)
            # grad_subp2_k    = LA.solve(L_hessian_k_sym,-L_trajp_k)
            scxl_grad[k]    = grad_subp2_k[0:self.nxl,:]
            scul_grad[k]    = grad_subp2_k[self.nxl:(self.nxl+self.nul),:]
            scxc_grad[k]    = grad_subp2_k[(self.nxl+self.nul):(self.nxl+self.nul+self.nxi*int(self.nq)),:]
            scuc_grad[k]    = grad_subp2_k[(self.nxl+self.nul+self.nxi*int(self.nq)):(self.nxl+self.nul+self.nxi*int(self.nq)+self.nui*int(self.nq)),:]
        # terminal gradients
        L_hessian_N = vertcat(
                                horzcat(Lscxlscxl[self.N],   Lscxlscxc[self.N]),
                                horzcat(Lscxlscxc[self.N].T, Lscxcscxc[self.N])
                                )
        xl_grad_N   = xl_grad[self.N]
        xc_grad_N   = grad_outc[0]['xi_grad'][self.N]
        for i in range(1,int(self.nq)):
            xc_grad_N = np.vstack((xc_grad_N,grad_outc[i]['xi_grad'][self.N]))
        scxL_grad_N = scxL_grad[self.N]
        scxC_grad_N = scxC_grad[self.N]
        L_trajp_N   = vertcat(
                            horzcat(Lscxlp[self.N] - dis_rn*p*xl_grad_N - scxL_grad_N),
                            horzcat(Lscxcp[self.N] - dis_rn*pi*xc_grad_N - scxC_grad_N)
                            )
        L_hessian_No = vertcat(
                                horzcat(Lscxlscxl_o[self.N],   Lscxlscxc_o[self.N]),
                                horzcat(Lscxlscxc_o[self.N].T, Lscxcscxc_o[self.N])
                                )
        min_eigval = np.min(LA.eigvalsh(L_hessian_No))
        MIN_eigen += [min_eigval]
        if min_eigval<0:
            reg = -min_eigval+1e-4
        else:
            reg = 0
        L_hessian_N_sym = L_hessian_N + reg*I_hess2 
        L, _jitter      = self.try_cholesky(L_hessian_N_sym, jitter0=0.0)   
        grad_subp2_N    = self.chol_solve(L, -L_trajp_N)
        scxl_grad[self.N] = grad_subp2_N[0:self.nxl,:]
        scxc_grad[self.N] = grad_subp2_N[self.nxl:(self.nxl+self.nxi*int(self.nq)),:]
        print('min_eigen=',np.min(MIN_eigen)) 
        grad_out2 = {
                    "scxl_grad":scxl_grad,
                    "scul_grad":scul_grad,
                    "scxc_grad":scxc_grad,
                    "scuc_grad":scuc_grad
                    }
        
        return grad_out2
    

    def SubP3_Gradient(self,auxSys3,grad_outl,grad_outc,grad_out2,scxL_grad,scuL_grad,scxC_grad,scuC_grad,p,pi,i_admm):
        xl_grad         = grad_outl['xl_grad']
        ul_grad         = grad_outl['ul_grad']
        scxl_grad       = grad_out2['scxl_grad']
        scul_grad       = grad_out2['scul_grad']
        scxc_grad       = grad_out2['scxc_grad']
        scuc_grad       = grad_out2['scuc_grad']
        dscxL_updatedp  = auxSys3['dscxL_updatedp']
        dscuL_updatedp  = auxSys3['dscuL_updatedp']
        dscxC_updatedp  = auxSys3['dscxC_updatedp']
        dscuC_updatedp  = auxSys3['dscuC_updatedp']
        scxL_grad_new   = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        scuL_grad_new   = self.N*[np.zeros((self.nul,self.n_Pauto))]
        scxC_grad_new   = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        scuC_grad_new   = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        dis_rn          = self.Discount_rate(i_admm)
        for k in range(self.N):
            xc_grad_k   = grad_outc[0]['xi_grad'][k]
            uc_grad_k   = grad_outc[0]['ui_grad'][k]
            for i in range(1,int(self.nq)):
                xc_grad_k = np.vstack((xc_grad_k,grad_outc[i]['xi_grad'][k]))
                uc_grad_k = np.vstack((uc_grad_k,grad_outc[i]['ui_grad'][k]))
            scxL_grad_new[k] = scxL_grad[k] + dis_rn*p*(xl_grad[k] - scxl_grad[k]) + dscxL_updatedp[k]
            scuL_grad_new[k] = scuL_grad[k] + dis_rn*p*(ul_grad[k] - scul_grad[k]) + dscuL_updatedp[k]
            scxC_grad_new[k] = scxC_grad[k] + dis_rn*pi*(xc_grad_k - scxc_grad[k]) + dscxC_updatedp[k]
            scuC_grad_new[k] = scuC_grad[k] + dis_rn*pi*(uc_grad_k - scuc_grad[k]) + dscuC_updatedp[k]
        # terminal gradients
        xc_grad_N   = grad_outc[0]['xi_grad'][self.N]
        for i in range(1,int(self.nq)):
            xc_grad_N = np.vstack((xc_grad_N,grad_outc[i]['xi_grad'][self.N]))
        scxL_grad_new[self.N]= scxL_grad[self.N] + dis_rn*p*(xl_grad[self.N] - scxl_grad[self.N]) + dscxL_updatedp[self.N]
        scxC_grad_new[self.N]= scxC_grad[self.N] + dis_rn*pi*(xc_grad_N - scxc_grad[self.N]) + dscxC_updatedp[self.N]

        grad_out3 = {
            "scxL_grad":scxL_grad_new,
            "scuL_grad":scuL_grad_new,
            "scxC_grad":scxC_grad_new,
            "scuC_grad":scuC_grad_new
        }

        return grad_out3
    

    def ADMM_Gradient_Solver(self,Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3, Ref_xl, Ref_ul, ref_xq, ref_uq, weight1, weight2, adaptiveADMM):
        # initialize the gradient trajectories of SubP2 and SubP3
        scxl_grad  = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        scul_grad  = self.N*[np.zeros((self.nul,self.n_Pauto))]
        scxL_grad  = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        scuL_grad  = self.N*[np.zeros((self.nul,self.n_Pauto))]
        scxc_grad  = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        scuc_grad  = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        scxC_grad  = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        scuC_grad  = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        # initial trajectories, same as those used in the ADMM recursion in the forward pass
        scxl       = np.zeros((self.N+1,self.nxl))
        scul       = np.zeros((self.N,self.nul))
        for k in range(self.N):
            scul[k,:] = Ref_ul[k*self.nul:(k+1)*self.nul]
            scxl[k,:] = np.reshape(self.model_l_fn(xl0=scxl[k,:],ul0=scul[k,:])['mdynlf'].full(),self.nxl)
        scxl[self.N,:]= Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl]
        scxc       = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))] 
        scuc       = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))] 
        for i in range(int(self.nq)):
            for k in range(self.N):
                scuc[i][k,:] = ref_uq[i*self.nui:(i+1)*self.nui]
                scxc[i][k,:] = np.reshape(self.model_i_fn(xi0=scxc[i][k,:],ui0=scuc[i][k,:])['mdynif'].full(),self.nxi)
            scxc[i][self.N,:] = ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi]
        scxL       = np.zeros((self.N+1,self.nxl))
        scuL       = np.zeros((self.N,self.nul))
        scxC       = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))] 
        scuC       = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))] 
        # lists for storing gradient trajectories
        Grad_Out1l     = []
        Grad_Out1c     = []
        Grad_Out2      = []
        Grad_Out3      = []
        GradTime       = []
        GradTimeCao    = []
        GradTimeCaos   = []
        GradTimePDP    = []
        GradTime_c     = []
        GradTimeCaos_c = []
        GradTimeCao_c  = []
        GradTimePDP_c  = []
        MeanerrorCao   = [] # error between gradRe and gradPDP
        MeanerrorPDP   = [] # error between gradRe and gradCao
        MeanerrorCao_c = []
        MeanerrorPDP_c = []
        Pauto      = np.concatenate((weight1,weight2))
        Iadmm      = 0 # admm iteration index
        for i_admm in range(self.max_iter_ADMM):
            if adaptiveADMM == 'f':
                i_admm = 1e2 # a very large i_admm makes the sigmoid function almost 1
            # gradients of Subproblem1
            opt_sol        = Opt_Sol1_l[Iadmm]
            auxSysl        = self.Get_AuxSys_DDP_Load(opt_sol,Ref_xl,Ref_ul,scxl,scul,scxL,scuL,weight1,i_admm)
            start_time     = TM.time()
            grad_outl      = self.DDP_Load_Gradient(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-1],i_admm)
            gradtimeOur    = (TM.time() - start_time)*1000
            start_time     = TM.time()
            grad_outl_Caos = self.Cao_Load_Gradient_s(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-1], i_admm)
            gradtimeCaos   = (TM.time() - start_time)*1000
            start_time     = TM.time()
            grad_outl_Cao  = self.Cao_Load_Gradient(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-1],i_admm)
            gradtimeCao    = (TM.time() - start_time)*1000
            start_time     = TM.time()
            grad_outl_PDP  = self.PDP_Load_Gradient(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-1], i_admm)
            gradtimePDP    = (TM.time() - start_time)*1000
            grad_outc = []
            grad_outcCao = []
            grad_outcPDP = []
            gradtimeOur_sum   = 0
            gradtimeCaos_sum  = 0
            gradtimeCao_sum   = 0
            gradtimePDP_sum   = 0
            for i in range(int(self.nq)):
                opt_solc  = Opt_Sol1_cddp[Iadmm]
                Ref_xi    = ref_xq[i]
                Ref_ui    = ref_uq[i*self.nui:(i+1)*self.nui]
                scxi      = scxc[i]
                scui      = scuc[i]
                scxI      = scxC[i]
                scuI      = scuC[i]
                auxSysi   = self.Get_AuxSys_DDP_Cable(opt_solc[i],Ref_xi,Ref_ui,scxi,scui,scxI,scuI,weight2,i_admm)
                scxi_grad = (self.N+1)*[np.zeros((self.nxi, self.n_Pauto))]
                scxI_grad = (self.N+1)*[np.zeros((self.nxi, self.n_Pauto))]
                scui_grad = self.N*[np.zeros((self.nui, self.n_Pauto))]
                scuI_grad = self.N*[np.zeros((self.nui, self.n_Pauto))]
                for k in range(self.N):
                    scxi_grad[k] = np.reshape(scxc_grad[k][i*self.nxi:(i+1)*self.nxi,:],(self.nxi,self.n_Pauto))
                    scxI_grad[k] = np.reshape(scxC_grad[k][i*self.nxi:(i+1)*self.nxi,:],(self.nxi,self.n_Pauto))
                    scui_grad[k] = np.reshape(scuc_grad[k][i*self.nui:(i+1)*self.nui,:],(self.nui,self.n_Pauto))
                    scuI_grad[k] = np.reshape(scuC_grad[k][i*self.nui:(i+1)*self.nui,:],(self.nui,self.n_Pauto))
                scxi_grad[self.N]= np.reshape(scxc_grad[self.N][i*self.nxi:(i+1)*self.nxi,:],(self.nxi,self.n_Pauto))
                scxI_grad[self.N]= np.reshape(scxC_grad[self.N][i*self.nxi:(i+1)*self.nxi,:],(self.nxi,self.n_Pauto))
                start_time           = TM.time()
                grad_outi            = self.DDP_Cable_Gradient(opt_solc[i],auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, weight2[-1],i_admm)
                gradtimeOur_cable    = (TM.time() - start_time)*1000
                gradtimeOur_sum     += gradtimeOur_cable
                grad_outc           += [grad_outi]
                start_time           = TM.time()
                grad_outi_Caos       = self.Cao_Cable_Gradient_s(opt_solc[i],auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, weight2[-1], i_admm)
                gradtimeCaos_cable   = (TM.time() - start_time)*1000
                gradtimeCaos_sum    += gradtimeCaos_cable
                start_time           = TM.time()
                grad_outi_Cao        = self.Cao_Cable_Gradient(opt_solc[i],auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, weight2[-1],i_admm)
                gradtimeCao_cable    = (TM.time() - start_time)*1000
                gradtimeCao_sum     += gradtimeCao_cable
                grad_outcCao        += [grad_outi_Cao]
                start_time           = TM.time()
                grad_outi_PDP        = self.PDP_Cable_Gradient(opt_solc[i],auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, weight2[-1], i_admm)
                gradtimePDP_cable    = (TM.time() - start_time)*1000
                gradtimePDP_sum     += gradtimePDP_cable
                grad_outcPDP        += [grad_outi_PDP]
            gradtimeOur_avgcable  = gradtimeOur_sum/self.nq
            gradtimeCaos_avgcable = gradtimeCaos_sum/self.nq
            gradtimeCao_avgcable  = gradtimeCao_sum/self.nq
            gradtimePDP_avgcable  = gradtimePDP_sum/self.nq
                
            # gradients of Subproblem2
            opt_sol1_c = Opt_Sol1_c[Iadmm]
            opt_sol2   = Opt_Sol2[Iadmm]
            auxSys2    = self.Get_AuxSys_SubP2(opt_sol,opt_sol1_c,opt_sol2,scxL,scuL,scxC,scuC,Pauto,i_admm)
            grad_out2  = self.SubP2_Gradient(auxSys2,grad_outl,grad_outc,scxL_grad,scuL_grad,scxC_grad,scuC_grad,weight1[-1],weight2[-1],i_admm) 
            # gradients of Subproblem3
            auxSys3    = self.Get_AuxSys_SubP3(opt_sol,opt_sol1_c,opt_sol2,i_admm)
            grad_out3  = self.SubP3_Gradient(auxSys3,grad_outl,grad_outc,grad_out2,scxL_grad,scuL_grad,scxC_grad,scuC_grad,weight1[-1],weight2[-1],i_admm)
            # update
            scxl       = opt_sol2['scxl_traj']
            scul       = opt_sol2['scul_traj']
            scxc       = opt_sol2['scxc_traj']
            scuc       = opt_sol2['scuc_traj']
            opt_sol3   = Opt_Sol3[Iadmm]
            scxL       = opt_sol3['scxL_traj_new']
            scuL       = opt_sol3['scuL_traj_new']
            scxC       = opt_sol3['scxC_traj_new']
            scuC       = opt_sol3['scuC_traj_new']
            scxl_grad  = grad_out2['scxl_grad']
            scul_grad  = grad_out2['scul_grad']
            scxc_grad  = grad_out2['scxc_grad']
            scuc_grad  = grad_out2['scuc_grad']
            scxL_grad  = grad_out3['scxL_grad']
            scuL_grad  = grad_out3['scuL_grad']
            scxC_grad  = grad_out3['scxC_grad']
            scuC_grad  = grad_out3['scuC_grad']
            # save the results
            Grad_Out1l     += [grad_outl]
            Grad_Out1c     += [grad_outc]
            Grad_Out2      += [grad_out2]
            Grad_Out3      += [grad_out3]
            GradTime       += [gradtimeOur]
            GradTimeCaos   += [gradtimeCaos]
            GradTimeCao    += [gradtimeCao]
            GradTimePDP    += [gradtimePDP]
            GradTime_c     += [gradtimeOur_avgcable]
            GradTimeCaos_c += [gradtimeCaos_avgcable]
            GradTimeCao_c  += [gradtimeCao_avgcable]
            GradTimePDP_c  += [gradtimePDP_avgcable] 
            

            xl_grad    = grad_outl['xl_grad']
            xl_gradCao = grad_outl_Cao['xl_grad']
            xl_gradPDP = grad_outl_PDP['xl_grad']
            
            Error1     = 0
            Error2     = 0
            Error1_c   = 0
            Error2_c   = 0
            for k in range(self.N):
                error1 = xl_grad[k+1] - xl_gradCao[k+1]
                Error1 += (LA.norm(error1,ord='fro')/LA.norm(xl_grad[k+1],ord='fro'))
                error2 = xl_grad[k+1] - xl_gradPDP[k+1]
                Error2 += (LA.norm(error2,ord='fro')/LA.norm(xl_grad[k+1],ord='fro'))
                for i in range(int(self.nq)):
                    error1_c = grad_outc[i]['xi_grad'][k+1] - grad_outcCao[i]['xi_grad'][k+1]
                    Error1_c += (LA.norm(error1_c,ord='fro')/LA.norm(grad_outc[i]['xi_grad'][k+1],ord='fro'))
                    error2_c = grad_outc[i]['xi_grad'][k+1] - grad_outcPDP[i]['xi_grad'][k+1]
                    Error2_c += (LA.norm(error2_c,ord='fro')/LA.norm(grad_outc[i]['xi_grad'][k+1],ord='fro'))
            meanerror1 = Error1/self.N
            meanerror2 = Error2/self.N    
            MeanerrorCao += [meanerror1]
            MeanerrorPDP += [meanerror2]
            meanerror1_c = Error1_c/(self.N*self.nq)
            meanerror2_c = Error2_c/(self.N*self.nq)
            MeanerrorCao_c += [meanerror1_c]
            MeanerrorPDP_c += [meanerror2_c]
            if Iadmm == self.max_iter_ADMM-1:
                print("g_Our:--- %s ms ---" % format(gradtimeOur,'.2f'))
                print("g_Cao_s:--- %s ms ---" % format(gradtimeCaos,'.2f'))
                print("g_Cao:--- %s ms ---" % format(gradtimeCao,'.2f'))
                print("g_PDP:--- %s ms ---" % format(gradtimePDP,'.2f'))
                print("g_Our_cable:--- %s ms ---" % format(gradtimeOur_avgcable,'.2f'))
                print("g_Cao_s_cable:--- %s ms ---" % format(gradtimeCaos_avgcable,'.2f'))
                print("g_Cao_cable:--- %s ms ---" % format(gradtimeCao_avgcable,'.2f'))
                print("g_PDP_cable:--- %s ms ---" % format(gradtimePDP_avgcable,'.2f'))
                print('meanerrorCao=',meanerror1,'meanerrorPDP=',meanerror2)
                print('meanerrorCao_c=',meanerror1_c,'meanerrorPDP_c=',meanerror2_c)
            Iadmm += 1
        
        return Grad_Out1l, Grad_Out1c, Grad_Out2, Grad_Out3, GradTime, GradTimeCaos, GradTimeCao,  GradTimePDP,  GradTime_c, GradTimeCaos_c, GradTimeCao_c, GradTimePDP_c,  MeanerrorCao, MeanerrorPDP, MeanerrorCao_c, MeanerrorPDP_c 
    
    

class Gradient_Solver:
    def __init__(self, sysm_para, horizon, xl, ul, scxl, scul, xi, ui, scxi, scui, P_auto, weight1, weight2):
        """
        [3]Kendall, A., Gal, Y. and Cipolla, R., 2018. 
        Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. 
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7482-7491).
        """
        self.nxl    = xl.numel()
        self.nul    = ul.numel()
        self.nxi    = xi.numel()
        self.nui    = ui.numel()
        self.n_Pauto= P_auto.numel()
        self.npl    = weight1.numel()
        self.npi    = weight2.numel()
        self.nq     = int(sysm_para[6])
        self.N      = horizon
        self.xl     = xl
        self.ul     = ul
        self.xi     = xi
        self.ui     = ui
        self.scxl   = scxl
        self.scul   = scul
        self.scxi   = scxi
        self.scui   = scui
        self.Pauto  = P_auto
        self.xl_ref = SX.sym('xl_ref',self.nxl)
        self.xi_ref = SX.sym('xi_ref',self.nxi)
        # boundaries of the hyperparameters
        self.p_min  = 1e-3 
        self.p_max  = 1e3
        #------------- loss definition -------------#
        # tracking loss
        track_error_l = self.xl - self.xl_ref
        track_error_i = self.xi - self.xi_ref
        self.loss_track_l = track_error_l.T@track_error_l
        self.weight_i     = np.diag(np.array([1,1,1,0,0,0,1,0]))
        self.loss_track_i = track_error_i.T@self.weight_i@track_error_i
        # primal residual loss
        r_primal_xl     = self.xl - self.scxl
        r_primal_ul     = self.ul - self.scul
        self.loss_rpl   = r_primal_xl.T@r_primal_xl + r_primal_ul.T@r_primal_ul
        self.loss_rpl_N = r_primal_xl.T@r_primal_xl
        r_primal_xi     = self.xi - self.scxi
        r_primal_ui     = self.ui - self.scui
        self.loss_rpi   = r_primal_xi.T@r_primal_xi + r_primal_ui.T@r_primal_ui
        self.loss_rpi_N = r_primal_xi.T@r_primal_xi
        #-------------We use ideas from [3] to auto-tune the weights in the meta-loss-----------#
        # self.wt         = SX.sym('wt',1)   # wt = log w1^2, the log-version weight for the tracking errors
        # self.wrp        = SX.sym('wrp',1)  # wrp = log w2^2, the log-version weight for the ADMM primal residuals 
        # self.L_t        = SX.sym('Lt',1)   # Loss_track
        # self.L_rp       = SX.sym('L_rp',1) # Loss_residual
        # self.scale      = 5                # a constant scaling factor
        # self.Meta_loss  = self.scale/2*(exp(-self.wt)*self.L_t + exp(-self.wrp)*self.L_rp + self.wt + self.wrp)
        # self.Meta_L_fn  = Function('Meta_l',[self.wt,self.wrp,self.L_t,self.L_rp],[self.Meta_loss],['wt0','wrp0','Lt0','Lrp0'],['Meta_lf'])

    def adaptive_meta_loss_weights(self,loss_t,loss_rp,wt):
        if loss_t > 1.25*loss_rp:
            wt_new = np.clip(2*wt,0.1,10)
        elif loss_rp > 1.25*loss_t:
            wt_new = np.clip(wt/2,0.1,10)
        else:
            wt_new = wt
        return wt_new


    def Set_Parameters(self,tunable_para,p_min):
        weight       = np.zeros(self.n_Pauto)
        for j in range(self.n_Pauto):
            if (j==2*self.nxl+self.nul):
                weight[j]= p_min + (self.p_max - p_min) * 1/(1+np.exp(-tunable_para[j]))
            elif (j== 2*self.nxl+self.nul+2*self.nxi+self.nui + 1):
                weight[j]= p_min + (self.p_max - p_min) * 1/(1+np.exp(-tunable_para[j]))
            else:
                weight[j]= self.p_min + (self.p_max - self.p_min) * 1/(1+np.exp(-tunable_para[j])) # sigmoid boundedness
        return weight
    

    def Set_Parameters_nn_l(self,tunable_para,pl_min):
        weight       = np.zeros(self.npl)
        for k in range(self.npl):
            if (k==self.npl-1):
                weight[k]= pl_min + (self.p_max - pl_min) * tunable_para[0,k]
            else:
                weight[k]= self.p_min + (self.p_max - self.p_min) * tunable_para[0,k] # sigmoid boundedness
        return weight
    
    def Set_Parameters_nn_i(self,tunable_para,pi_min):
        weight       = np.zeros(self.npi)
        for k in range(self.npi):
            if (k== self.npi-1):
                weight[k]= pi_min + (self.p_max - pi_min) * tunable_para[0,k]
            else:
                weight[k]= self.p_min + (self.p_max - self.p_min) * tunable_para[0,k] # sigmoid boundedness
        return weight
    

    def ChainRule_Gradient(self,tunable_para,p_min):
        Tunable      = SX.sym('Tp',1,self.n_Pauto)
        Weight       = SX.sym('wp',1,self.n_Pauto)
        for k in range(self.n_Pauto):
            if k==2*self.nxl+self.nul:
                Weight[k]= p_min + (self.p_max - p_min) * 1/(1+exp(-Tunable[k]))
            elif k== 2*self.nxl+self.nul+2*self.nxi+self.nui + 1:
                Weight[k]= p_min + (self.p_max - p_min) * 1/(1+exp(-Tunable[k]))
            else:
                Weight[k]= self.p_min + (self.p_max - self.p_min) * 1/(1 + exp(-Tunable[k])) # sigmoid boundedness
        dWdT         = jacobian(Weight,Tunable)
        dWdT_fn      = Function('dWdT',[Tunable],[dWdT],['Tp0'],['dWdT_f'])
        weight_grad  = dWdT_fn(Tp0=tunable_para)['dWdT_f'].full()

        return weight_grad
    
    def ChainRule_Gradient_nn_l(self,tunable_para,pl_min):
        Tunable      = SX.sym('Tp',1,self.npl)
        Weight       = SX.sym('wp',1,self.npl)
        for k in range(self.npl):
            if k==2*self.nxl+self.nul:
                Weight[k]= pl_min + (self.p_max - pl_min) * Tunable[k]
            else:
                Weight[k]= self.p_min + (self.p_max - self.p_min) * Tunable[k] # sigmoid boundedness
        dWdT         = jacobian(Weight,Tunable)
        dWdT_fn      = Function('dWdT',[Tunable],[dWdT],['Tp0'],['dWdT_f'])
        weight_grad  = dWdT_fn(Tp0=tunable_para)['dWdT_f'].full()
        return weight_grad
    
    def ChainRule_Gradient_nn_i(self,tunable_para,pi_min):
        Tunable      = SX.sym('Tp',1,self.npi)
        Weight       = SX.sym('wp',1,self.npi)
        for k in range(self.npi):
            if k== 2*self.nxi+self.nui:
                Weight[k]= pi_min + (self.p_max - pi_min) * Tunable[k]
            else:
                Weight[k]= self.p_min + (self.p_max - self.p_min) * Tunable[k] # sigmoid boundedness
        dWdT         = jacobian(Weight,Tunable)
        dWdT_fn      = Function('dWdT',[Tunable],[dWdT],['Tp0'],['dWdT_f'])
        weight_grad  = dWdT_fn(Tp0=tunable_para)['dWdT_f'].full()
        return weight_grad
    

    def loss(self,Opt_Sol1_l,Opt_Sol1_c,Opt_Sol2,Ref_xl,ref_xq, wt, wrp):
        xl_traj   = Opt_Sol1_l[-1]['xl_traj']
        ul_traj   = Opt_Sol1_l[-1]['ul_traj']
        xc_list   = Opt_Sol1_c[-1]['xc_traj']
        uc_list   = Opt_Sol1_c[-1]['uc_traj']
        scxl_traj = Opt_Sol2[-1]['scxl_traj']
        scul_traj = Opt_Sol2[-1]['scul_traj']
        scxc_traj = Opt_Sol2[-1]['scxc_traj'] # list
        scuc_traj = Opt_Sol2[-1]['scuc_traj'] # list
        loss_track = 0
        loss_resid = 0
        for k in range(self.N):
            xl_k        = np.reshape(xl_traj[k,:],(self.nxl,1))
            ul_k        = np.reshape(ul_traj[k,:],(self.nul,1))
            scxl_k      = np.reshape(scxl_traj[k,:],(self.nxl,1))
            scul_k      = np.reshape(scul_traj[k,:],(self.nul,1))
            refxl_k     = np.reshape(Ref_xl[k*self.nxl:(k+1)*self.nxl],(self.nxl,1))
            error_k     = xl_k - refxl_k # load tracking error
            resid_xk    = xl_k - scxl_k  # load primal state residual
            resid_uk    = ul_k - scul_k  # load primal control residual
            loss_track += error_k.T@error_k # load tracking loss at k
            loss_resid += resid_xk.T@resid_xk + resid_uk.T@resid_uk
            for i in range(self.nq):
                xi_k        = np.reshape(xc_list[i][k,:],(self.nxi,1))
                ui_k        = np.reshape(uc_list[i][k,:],(self.nui,1))
                scxi_k      = np.reshape(scxc_traj[i][k,:],(self.nxi,1))
                scui_k      = np.reshape(scuc_traj[i][k,:],(self.nui,1))
                refxi_k     = np.reshape(ref_xq[i][k*self.nxi:(k+1)*self.nxi],(self.nxi,1))
                error_ik    = xi_k - refxi_k
                resid_xik   = xi_k - scxi_k
                resid_uik   = ui_k - scui_k
                loss_track += error_ik.T@self.weight_i@error_ik
                loss_resid += resid_xik.T@resid_xik + resid_uik.T@resid_uik
        xl_N        = np.reshape(xl_traj[self.N,:],(self.nxl,1))
        scxl_N      = np.reshape(scxl_traj[self.N,:],(self.nxl,1))
        refxl_N     = np.reshape(Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],(self.nxl,1))
        error_N     = xl_N - refxl_N
        resid_xN    = xl_N - scxl_N
        loss_track += error_N.T@error_N
        loss_resid += resid_xN.T@resid_xN
        for i in range(self.nq):
            xi_N        = np.reshape(xc_list[i][self.N,:],(self.nxi,1))
            scxi_N      = np.reshape(scxc_traj[i][self.N,:],(self.nxi,1))
            refxi_N     = np.reshape(ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi],(self.nxi,1))
            error_iN    = xi_N - refxi_N
            resid_xiN   = xi_N - scxi_N
            loss_track += error_iN.T@self.weight_i@error_iN
            loss_resid += resid_xiN.T@resid_xiN
        
        loss = wt*loss_track + wrp*loss_resid
        return loss, loss_track, loss_resid
    

    def ChainRule(self,Opt_Sol1_l,Opt_Sol1_c,Opt_Sol2,Ref_xl,ref_xq,Grad_Out1l,Grad_Out1c,Grad_Out2,wt,wrp):
        dltdxl          = jacobian(self.loss_track_l,self.xl)
        dltdxl_fn       = Function('dltdxl',[self.xl,self.xl_ref],[dltdxl],['xl0','refxl0'],['dltdxl_f'])
        dltdxi          = jacobian(self.loss_track_i,self.xi)
        dltdxi_fn       = Function('dltdxi',[self.xi,self.xi_ref],[dltdxi],['xi0','refxi0'],['dltdxi_f'])
        dlrpdxl         = jacobian(self.loss_rpl,self.xl)
        dlrpdxl_fn      = Function('dlrpdxl',[self.xl,self.scxl,self.ul,self.scul],[dlrpdxl],['xl0','scxl0','ul0','scul0'],['dlrpdxl_f'])
        dlrpdul         = jacobian(self.loss_rpl,self.ul)
        dlrpdul_fn      = Function('dlrpdul',[self.xl,self.scxl,self.ul,self.scul],[dlrpdul],['xl0','scxl0','ul0','scul0'],['dlrpdul_f'])
        dlrpdscxl       = jacobian(self.loss_rpl,self.scxl)
        dlrpdscxl_fn    = Function('dlrpdscxl',[self.xl,self.scxl,self.ul,self.scul],[dlrpdscxl],['xl0','scxl0','ul0','scul0'],['dlrpdscxl_f'])
        dlrpdscul       = jacobian(self.loss_rpl,self.scul)
        dlrpdscul_fn    = Function('dlrpdscul',[self.xl,self.scxl,self.ul,self.scul],[dlrpdscul],['xl0','scxl0','ul0','scul0'],['dlrpdscul_f'])
        dlrpdxi         = jacobian(self.loss_rpi,self.xi)
        dlrpdxi_fn      = Function('dlrpdxi',[self.xi,self.scxi,self.ui,self.scui],[dlrpdxi],['xi0','scxi0','ui0','scui0'],['dlrpdxi_f'])
        dlrpdui         = jacobian(self.loss_rpi,self.ui)
        dlrpdui_fn      = Function('dlrpdui',[self.xi,self.scxi,self.ui,self.scui],[dlrpdui],['xi0','scxi0','ui0','scui0'],['dlrpdui_f'])
        dlrpdscxi       = jacobian(self.loss_rpi,self.scxi)
        dlrpdscxi_fn    = Function('dlrpdscxi',[self.xi,self.scxi,self.ui,self.scui],[dlrpdscxi],['xi0','scxi0','ui0','scui0'],['dlrpdscxi_f'])
        dlrpdscui       = jacobian(self.loss_rpi,self.scui)
        dlrpdscui_fn    = Function('dlrpdscui',[self.xi,self.scxi,self.ui,self.scui],[dlrpdscui],['xi0','scxi0','ui0','scui0'],['dlrpdscui_f'])
        dlrpdxlN        = jacobian(self.loss_rpl_N,self.xl)
        dlrpdxlN_fn     = Function('dlrpdxlN',[self.xl,self.scxl],[dlrpdxlN],['xl0','scxl0'],['dlrpdxlN_f'])
        dlrpdscxlN      = jacobian(self.loss_rpl_N,self.scxl)
        dlrpdscxlN_fn   = Function('dlrpdscxlN',[self.xl,self.scxl],[dlrpdscxlN],['xl0','scxl0'],['dlrpdscxlN_f'])
        dlrpdxiN        = jacobian(self.loss_rpi_N,self.xi)
        dlrpdxiN_fn     = Function('dlrpdxiN',[self.xi,self.scxi],[dlrpdxiN],['xi0','scxi0'],['dlrpdxiN_f'])
        dlrpdscxiN      = jacobian(self.loss_rpi_N,self.scxi)
        dlrpdscxiN_fn   = Function('dlrpdscxiN',[self.xi,self.scxi],[dlrpdscxiN],['xi0','scxi0'],['dlrpdscxiN_f'])
        dltdw           = 0 # gradient of the tracking errors
        dlrpdw          = 0 # gradient of the ADMM primal residuals
        # load trajectories
        k_admm          = -1
        xl_traj         = Opt_Sol1_l[k_admm]['xl_traj']
        ul_traj         = Opt_Sol1_l[k_admm]['ul_traj']
        scxl_traj       = Opt_Sol2[k_admm]['scxl_traj']
        scul_traj       = Opt_Sol2[k_admm]['scul_traj']
        # load gradient trajectories
        xl_grad         = Grad_Out1l[k_admm]['xl_grad']
        ul_grad         = Grad_Out1l[k_admm]['ul_grad']
        scxl_grad       = Grad_Out2[k_admm]['scxl_grad']
        scul_grad       = Grad_Out2[k_admm]['scul_grad']
        # cable trajectories
        xc_traj         = Opt_Sol1_c[k_admm]['xc_traj'] # a list
        uc_traj         = Opt_Sol1_c[k_admm]['uc_traj'] # a list
        scxc_traj       = Opt_Sol2[k_admm]['scxc_traj'] # a list
        scuc_traj       = Opt_Sol2[k_admm]['scuc_traj'] # a list
        # cable gradient trajectories
        grad_outc       = Grad_Out1c[k_admm] # a list that contains both state and control gradients
        scxc_grad       = Grad_Out2[k_admm]['scxc_grad']
        scuc_grad       = Grad_Out2[k_admm]['scuc_grad']
        # meta-loss
        loss, loss_track, loss_resid   = self.loss(Opt_Sol1_l,Opt_Sol1_c,Opt_Sol2,Ref_xl,ref_xq,wt,wrp)
        
        for k in range(self.N):
            # gradient of the load tracking errors
            dltdxl_k    = dltdxl_fn(xl0=xl_traj[k,:],refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl])['dltdxl_f'].full()
            dltldw      = dltdxl_k@xl_grad[k]
            # print('dltldwr1=',dltldw[0,2*self.nxl],'dltldwpi=',dltldw[0,self.n_Pauto-1])
            dltdw      += dltldw
            # gradient of the load primal residuals
            dlrpdxl_k   = dlrpdxl_fn(xl0=xl_traj[k,:],scxl0=scxl_traj[k,:],ul0=ul_traj[k,:],scul0=scul_traj[k,:])['dlrpdxl_f'].full()
            dlrpdscxl_k = dlrpdscxl_fn(xl0=xl_traj[k,:],scxl0=scxl_traj[k,:],ul0=ul_traj[k,:],scul0=scul_traj[k,:])['dlrpdscxl_f'].full()
            dlrpdul_k   = dlrpdul_fn(xl0=xl_traj[k,:],scxl0=scxl_traj[k,:],ul0=ul_traj[k,:],scul0=scul_traj[k,:])['dlrpdul_f'].full()
            dlrpdscul_k = dlrpdscul_fn(xl0=xl_traj[k,:],scxl0=scxl_traj[k,:],ul0=ul_traj[k,:],scul0=scul_traj[k,:])['dlrpdscul_f'].full()
            dlrpdw     += dlrpdxl_k@xl_grad[k] + dlrpdscxl_k@scxl_grad[k] + dlrpdul_k@ul_grad[k] + dlrpdscul_k@scul_grad[k]
            for i in range(self.nq):
                # gradient of the cable tracking errors
                xi_traj     = xc_traj[i]
                ui_traj     = uc_traj[i]
                scxi_traj   = scxc_traj[i]
                scui_traj   = scuc_traj[i]
                refxi_k     = ref_xq[i][k*self.nxi:(k+1)*self.nxi]
                dltdxi_k    = dltdxi_fn(xi0=xi_traj[k,:],refxi0=refxi_k)['dltdxi_f'].full()
                grad_outi   = grad_outc[i]
                xi_grad     = grad_outi['xi_grad']
                dltidw      = dltdxi_k@xi_grad[k]
                # print('dltidwr4=',dltidw[0,2*self.nxl+self.nul+2*self.nxi+self.nui],'dltidwpl=',dltidw[0,2*self.nxl+self.nul],'dltidwpi=',dltidw[0,2*self.nxl+self.nul+2*self.nxi+self.nui+1])
                dltdw      += dltidw
                # gradient of the cable primal residuals
                ui_grad     = grad_outi['ui_grad']
                scxi_grad_k = scxc_grad[k][i*self.nxi:(i+1)*self.nxi,:]
                scui_grad_k = scuc_grad[k][i*self.nui:(i+1)*self.nui,:]
                dlrpdxi_k   = dlrpdxi_fn(xi0=xi_traj[k,:],scxi0=scxi_traj[k,:],ui0=ui_traj[k,:],scui0=scui_traj[k,:])['dlrpdxi_f'].full()
                dlrpdscxi_k = dlrpdscxi_fn(xi0=xi_traj[k,:],scxi0=scxi_traj[k,:],ui0=ui_traj[k,:],scui0=scui_traj[k,:])['dlrpdscxi_f'].full()
                dlrpdui_k   = dlrpdui_fn(xi0=xi_traj[k,:],scxi0=scxi_traj[k,:],ui0=ui_traj[k,:],scui0=scui_traj[k,:])['dlrpdui_f'].full()
                dlrpdscui_k = dlrpdscui_fn(xi0=xi_traj[k,:],scxi0=scxi_traj[k,:],ui0=ui_traj[k,:],scui0=scui_traj[k,:])['dlrpdscui_f'].full()
                dlrpdw     += dlrpdxi_k@xi_grad[k] + dlrpdscxi_k@scxi_grad_k + dlrpdui_k@ui_grad[k] + dlrpdscui_k@scui_grad_k
        # -----terminal gradients-----#
        dltdxl_N    = dltdxl_fn(xl0=xl_traj[self.N,:],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl])['dltdxl_f'].full()
        dltdw      += dltdxl_N@xl_grad[self.N]
        dlrpdxl_N   = dlrpdxlN_fn(xl0=xl_traj[self.N,:],scxl0=scxl_traj[self.N,:])['dlrpdxlN_f'].full()
        dlrpdscxl_N = dlrpdscxlN_fn(xl0=xl_traj[self.N,:],scxl0=scxl_traj[self.N,:])['dlrpdscxlN_f'].full()
        dlrpdw     += dlrpdxl_N@xl_grad[self.N] + dlrpdscxl_N@scxl_grad[self.N]
        for i in range(self.nq):
            xi_traj     = xc_traj[i]
            scxi_traj   = scxc_traj[i]
            refxi_N     = ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi]
            dltdxi_N    = dltdxi_fn(xi0=xi_traj[self.N,:],refxi0=refxi_N)['dltdxi_f'].full()
            grad_outi   = grad_outc[i]
            xi_grad     = grad_outi['xi_grad']
            dltdw      += dltdxi_N@xi_grad[self.N]
            scxi_grad_N = scxc_grad[self.N][i*self.nxi:(i+1)*self.nxi,:]
            dlrpdxi_N   = dlrpdxiN_fn(xi0=xi_traj[self.N,:],scxi0=scxi_traj[self.N,:])['dlrpdxiN_f'].full()
            dlrpdscxi_N = dlrpdscxiN_fn(xi0=xi_traj[self.N,:],scxi0=scxi_traj[self.N,:])['dlrpdscxiN_f'].full()
            dlrpdw     += dlrpdxi_N@xi_grad[self.N] + dlrpdscxi_N@scxi_grad_N
        # total gradient
        dldw        = wt*dltdw + wrp*dlrpdw

        return dldw, loss, loss_track, loss_resid
  





















    





        

    

                


                    







    


        



    


    

    

    

        
        
            





    