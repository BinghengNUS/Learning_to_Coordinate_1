import os

_DEFAULT_JAX_CPU_DEVICE_COUNT = "256"
_force_host_flag = f"--xla_force_host_platform_device_count={_DEFAULT_JAX_CPU_DEVICE_COUNT}"
_existing_xla_flags = os.environ.get("XLA_FLAGS", "")
if _force_host_flag not in _existing_xla_flags:
    os.environ["XLA_FLAGS"] = f"{_existing_xla_flags} {_force_host_flag}".strip()

for _env_name, _env_value in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ[_env_name] = _env_value
os.environ.setdefault("JAX_ENABLE_X64", "True")

from casadi import *
import numpy as np
from numpy import linalg as LA
import math
from scipy.spatial.transform import Rotation as Rot
from scipy import linalg as sLA
from scipy.linalg import null_space
import time as TM
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import ArpackNoConvergence

import jax
import jax.numpy as jnp
from jax import vmap, pmap
from jax.scipy.linalg import solve_triangular
jax.config.update("jax_enable_x64", True)


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
        # Cable and obstacle's parameters
        self.cl0    = sysm_para[10] # the cable length [m]
        self.ro     = sysm_para[11] # the radius of obstacle [m]
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
        # lower bound of the ADMM penalty parameter
        self.p_min  = 1e-3
        # Optional one-shot timing comparison for cable-gradient execution.
        self.enable_cable_parallel_diagnostics = False
        _all_local_devices = list(jax.local_devices())
        # SubP1 cable-gradient math stays unchanged; this fixed launch axis keeps
        # agent-count changes from changing the pmap shape for nq=4/8/12.
        self.jax_subp1_agent_devices = tuple(_all_local_devices[:max(1, min(12, len(_all_local_devices)))])
        # For diagnostics/legacy paths we keep one cable shard per device.
        self.jax_cable_pmap_devices = max(1, min(int(self.nq), jax.local_device_count()))
        self.jax_cable_devices = tuple(jax.local_devices()[:self.jax_cable_pmap_devices])
        self.jax_subp2_devices = tuple(_all_local_devices)
        self.jax_subp3_devices = tuple(_all_local_devices)
        self.jax_subp3_load_devices = self.jax_subp3_devices
        self.jax_subp3_cable_devices = self.jax_subp3_devices
        # SubP3 is elementwise/lightweight; using all 200+ CPU devices adds more
        # launch/sharding overhead than useful parallel work.
        self._subp3_max_devices = max(1, min(32, len(self.jax_subp3_devices)))
        self._subp3_disjoint_device_groups = False
        self._subp2_max_workers = max(1, os.cpu_count() or 1)
        self._subp2_executor = ThreadPoolExecutor(max_workers=self._subp2_max_workers) if self._subp2_max_workers > 1 else None
        self._subp3_executor = None
        self.debug_train_iter = None
        self.debug_task_idx = None
        self.enable_subp2_la_diagnostics = True
        self.enable_subp2_prepare_diagnostics = False
        self.enable_subp3_parallel_diagnostics = False
    
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
        self.t_max = 5 # the maximum tension force
        self.scxi_lb = [-1e19,-1e19,-1e19, -1e19,-1e19,-1e19, -1e19,-1e19,-1e19, -1e19,-1e19,-1e19, self.t_min,-1e19]
        self.scxi_ub = [1e19,1e19,1e19, 1e19,1e19,1e19, 1e19,1e19,1e19, 1e19,1e19,1e19, self.t_max, 1e19]


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
        self.ui_bound =1e3
        self.ui_lb = self.nui*[-self.ui_bound]
        self.ui_ub = self.nui*[self.ui_bound]

    def SetDyns(self, model_l, model_i):
        self.model_l = self.xl + self.dt*model_l # 4th-order Runge-Kutta discrete-time load dynamics model
        self.model_i = self.xi + self.dt*model_i # 4th-order Runge-Kutta discrete-time cable dynamics model
        self.model_l_fn = Function('mdynl',[self.xl, self.ul],[self.model_l],['xl0','ul0'],['mdynlf'])
        self.model_i_fn = Function('mdyni',[self.xi, self.ui],[self.model_i],['xi0','ui0'],['mdynif'])

    def SetWeightPara(self):
        # self.nwsl    = self.nxl
        self.para_l  = SX.sym('paral',1,(2*self.nxl+self.nul+4)) # including the ADMM penalty parameter, px, pu, gammax, gammau
        self.npl     = self.para_l.numel()
        self.para_i  = SX.sym('parai',1,(2*self.nxi+self.nui+4)) # including the ADMM penalty parameter, pix, piu, gammaix, gammaiu
        self.npi     = self.para_i.numel()
        self.P_auto  = horzcat(self.para_l,self.para_i)
        self.n_Pauto = self.P_auto.numel()

    def Discount_rate(self,gamma,a,ADMM_max):
        dis = 1/(1+exp(-gamma*(a - int(ADMM_max/2))))
        return dis

    def open_loop_penalty(self,rho,gamma,a,ADMM_max,b=0.5):
        # rho_a = self.p_min + (rho - self.p_min) * 1/(1 + exp(-gamma*(a/(int(ADMM_max)-1)-b))) # iteration-dependent open-loop penalty policy
        rho_a = self.p_min + (rho - self.p_min) * 1/(1+exp(-gamma*(a - (ADMM_max-1)/2)))
        return rho_a

    def open_loop_penalty_jax(self, rho, gamma, a, ADMM_max, b=0.5):
        del b
        return self.p_min + (rho - self.p_min) * 1 / (1 + jnp.exp(-gamma * (a - (ADMM_max - 1) / 2)))

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

    def SetPayloadCostDyn(self,ADMM_max):
        self.ref_xl   = SX.sym('refxl',self.nxl,1)
        self.ref_ul   = SX.sym('reful',self.nul,1) 
        track_error_l = self.xl - self.ref_xl
        ctrl_error_l  = self.ul - self.ref_ul
        self.a        = SX.sym('a',1) # the ADMM iteration index
        self.Ql_k     = diag(self.para_l[0,0:self.nxl])
        self.Ql_N     = diag(self.para_l[0,self.nxl:2*self.nxl])
        self.Rl_k     = diag(self.para_l[0,2*self.nxl:2*self.nxl+self.nul])
        self.px_dis   = self.open_loop_penalty(self.para_l[0,-4],self.para_l[0,-2],self.a,ADMM_max)
        self.pu_dis   = self.open_loop_penalty(self.para_l[0,-3],self.para_l[0,-1],self.a,ADMM_max)
        # path cost
        self.resid_xl = self.xl - self.scxl + self.scxL/self.px_dis
        self.resid_ul = self.ul - self.scul + self.scuL/self.pu_dis
        self.Jl_k     = 1/2 * (track_error_l.T@self.Ql_k@track_error_l + ctrl_error_l.T@self.Rl_k@ctrl_error_l) + self.px_dis/2*self.resid_xl.T@self.resid_xl + self.pu_dis/2*self.resid_ul.T@self.resid_ul
        self.Jl_kfn   = Function('Jl_k',[self.xl, self.ul, self.scxl, self.scxL, self.scul, self.scuL, self.ref_xl, self.ref_ul, self.para_l, self.a],[self.Jl_k],['xl0', 'ul0', 'scxl0', 'scxL0', 'scul0', 'scuL0', 'refxl0', 'reful0', 'paral0', 'a0'],['Jl_kf'])
        # terminal cost
        self.Jl_N     = 1/2 * track_error_l.T@self.Ql_N@track_error_l + self.px_dis/2*self.resid_xl.T@self.resid_xl
        self.Jl_Nfn   = Function('Jl_N',[self.xl, self.ref_xl, self.scxl, self.scxL, self.para_l, self.a],[self.Jl_N],['xl0', 'refxl0', 'scxl0', 'scxL0', 'paral0', 'a0'],['Jl_Nf'])
        # path cost of ADMM subproblem2
        self.Jl_P2_k  = self.px_dis/2*self.resid_xl.T@self.resid_xl + self.pu_dis/2*self.resid_ul.T@self.resid_ul 
        self.Jl_P2_k_fn = Function('Jl_P2_k',[self.xl, self.ul, self.scxl, self.scxL, self.scul, self.scuL, self.para_l, self.a],[self.Jl_P2_k],['xl0', 'ul0', 'scxl0', 'scxL0', 'scul0', 'scuL0', 'paral0', 'a0'],['Jl_P2_kf'])
        # terminal cost of ADMM subproblem2
        self.Jl_P2_N  = self.px_dis/2*self.resid_xl.T@self.resid_xl 
        self.Jl_P2_N_fn = Function('Jl_P2_N',[self.xl, self.scxl, self.scxL, self.para_l, self.a],[self.Jl_P2_N],['xl0', 'scxl0', 'scxL0', 'paral0', 'a0'],['Jl_P2_Nf'])


    def SetCableCostDyn(self,ADMM_max):
        self.ref_xi   = SX.sym('refxi',self.nxi,1)
        self.ref_ui   = SX.sym('refui',self.nui,1)
        track_error_i = self.xi - self.ref_xi
        ctrl_error_i  = self.ui - self.ref_ui
        self.Qi_k     = diag(self.para_i[0,0:self.nxi])
        self.Qi_N     = diag(self.para_i[0,self.nxi:2*self.nxi])
        self.Ri_k     = diag(self.para_i[0,2*self.nxi:2*self.nxi+self.nui])
        self.pix_dis  = self.open_loop_penalty(self.para_i[0,-4],self.para_i[0,-2],self.a,ADMM_max) # \rho_i
        self.piu_dis  = self.open_loop_penalty(self.para_i[0,-3],self.para_i[0,-1],self.a,ADMM_max) # \sigma_i
        # path cost
        self.resid_xi = self.xi - self.scxi + self.scxI/self.pix_dis
        self.resid_ui = self.ui - self.scui + self.scuI/self.piu_dis  
        self.Ji_k     = 1/2 * (track_error_i.T@self.Qi_k@track_error_i + ctrl_error_i.T@self.Ri_k@ctrl_error_i) + self.pix_dis/2*self.resid_xi.T@self.resid_xi + self.piu_dis/2*self.resid_ui.T@self.resid_ui 
        self.Ji_k_fn  = Function('Ji_k',[self.xi, self.ui, self.scxi, self.scxI, self.scui, self.scuI, self.ref_xi, self.ref_ui, self.para_i, self.a],[self.Ji_k],['xi0', 'ui0', 'scxi0', 'scxI0', 'scui0', 'scuI0', 'refxi0', 'refui0', 'parai0', 'a0'],['Ji_kf'])
        # terminal cost
        self.Ji_N     = 1/2 * track_error_i.T@self.Qi_N@track_error_i + self.pix_dis/2*self.resid_xi.T@self.resid_xi 
        self.Ji_N_fn  = Function('Ji_N',[self.xi, self.ref_xi, self.scxi, self.scxI, self.para_i, self.a],[self.Ji_N],['xi0', 'refxi0', 'scxi0', 'scxI0', 'parai0', 'a0'],['Ji_Nf'])
        # path cost of ADMM subproblem2
        self.Ji_P2_k  = self.pix_dis/2*self.resid_xi.T@self.resid_xi + self.piu_dis/2*self.resid_ui.T@self.resid_ui 
        self.Ji_P2_k_fn = Function('Ji_P2_k',[self.xi, self.scxi, self.scxI, self.ui, self.scui, self.scuI, self.para_i, self.a],[self.Ji_P2_k],['xi0', 'scxi0', 'scxI0', 'ui0', 'scui0', 'scuI0', 'parai0', 'a0'],['Ji_P2_kf'])
        # terminal cost of ADMM subproblem2
        self.Ji_P2_N  = self.pix_dis/2*self.resid_xi.T@self.resid_xi 
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
        self.lxixi    = jacobian(self.lxi,self.xi) # already includes \rho_i
        self.lxixiN   = jacobian(self.lxiN,self.xi)
        self.lxixiN_fn= Function('lxixiN',[self.para_i,self.a],[self.lxixiN],['parai0','a0'],['lxixiN_f'])
        self.lxiui    = jacobian(self.lxi,self.ui)
        # print('lxiui=',self.lxiui)
        self.luiui    = jacobian(self.lui,self.ui) # already includes \sigma_i
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

   
    def DDP_Load_ADMM_Subp1(self,xl_0,Ref_xl,Ref_ul,weight1,scxl,scul,scxL,scuL,max_iter,e_tol,i_admm):
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
        # if i_admm ==0:
        for k in range(self.N):
            u_k    = np.reshape(Ref_ul[k*self.nul:(k+1)*self.nul],(self.nul,1))
            # X_nominal[:,k:k+1] = np.reshape(Ref_xl[k*self.nxl:(k+1)*self.nxl],(self.nxl,1))
            X_nominal[:,k:k+1] = self.model_l_fn(xl0=X_nominal[:,k],ul0=u_k)['mdynlf'].full() # start from a bad state
            U_nominal[:,k:k+1]   = u_k
            cost_prev     += self.Jl_kfn(xl0=X_nominal[:,k],ul0=u_k,scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],
                                        scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],
                                        refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Jl_kf'].full()
        cost_prev += self.Jl_Nfn(xl0=X_nominal[:,-1],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],
                                 scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl],paral0=weight1,a0=i_admm)['Jl_Nf'].full()
        # else:
        #     for k in range(self.N):
        #         X_nominal[:,k+1:k+2] = np.reshape(scxl[k*self.nxl:(k+1)*self.nxl],(self.nxl,1))
        #         U_nominal[:,k:k+1]   = np.reshape(scul[k*self.nul:(k+1)*self.nul],(self.nul,1))
        #         cost_prev     += self.Jl_kfn(xl0=X_nominal[:,k],ul0=U_nominal[:,k],scxl0=scxl[k*self.nxl:(k+1)*self.nxl],scxL0=scxL[k*self.nxl:(k+1)*self.nxl],
        #                                 scul0=scul[k*self.nul:(k+1)*self.nul],scuL0=scuL[k*self.nul:(k+1)*self.nul],
        #                                 refxl0=Ref_xl[k*self.nxl:(k+1)*self.nxl],reful0=Ref_ul[k*self.nul:(k+1)*self.nul],paral0=weight1,a0=i_admm)['Jl_kf'].full()
        #     cost_prev += self.Jl_Nfn(xl0=X_nominal[:,-1],refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],
        #                          scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl],paral0=weight1,a0=i_admm)['Jl_Nf'].full()    

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
            Vx[self.N] = self.lxlN_fn(xl0=X_nominal[:,self.N],
                                      refxl0=Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl],
                                      scxl0=scxl[self.N*self.nxl:(self.N+1)*self.nxl],
                                      scxL0=scxL[self.N*self.nxl:(self.N+1)*self.nxl],
                                      paral0=weight1,
                                      a0=i_admm)['lxlN_f'].full()
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
                try:
                    L, _jitter = self.try_cholesky(Quu_reg_k, jitter0=0.0)
                except LA.LinAlgError:
                    chol_failed = True
                    break
                Quu_inv      = self.chol_solve(L, I_u) # only for computing the gradients
                K_fb[k]      = self.chol_solve(L, -Qxu_k.T)
                k_ff[k]      = self.chol_solve(L, -Qu_k)
                Vx[k]        = Qx_k + Qxu_k @ k_ff[k]
                Vxx[k]       = self.symmetry(Qxx_k + Qxu_k @ K_fb[k])
                Fx[k]        = self.Fxl_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k])['Fxl_f'].full()
                Fu[k]        = self.Ful_fn(xl0=X_nominal[:,k],ul0=U_nominal[:,k])['Ful_f'].full()
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
    

    def DDP_Cable_ADMM_Subp1(self,xi_0,Ref_xi,Ref_ui,weight2,scxi,scui,scxI,scuI,max_iter,e_tol,i_admm):
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
        # if i_admm ==0:
        for k in range(self.N):
            u_k    = np.reshape(Ref_ui,(self.nui,1))
                # X_nominal[:,k:k+1] = np.reshape(Ref_xi[k*self.nxi:(k+1)*self.nxi],(self.nxi,1))
            X_nominal[:,k:k+1] = self.model_i_fn(xi0=X_nominal[:,k],ui0=u_k)['mdynif'].full() # start from a bad state
            U_nominal[:,k:k+1]   = u_k
            cost_prev     += self.Ji_k_fn(xi0=X_nominal[:,k],ui0=u_k,scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],
                                        scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],
                                        refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,parai0=weight2,a0=i_admm)['Ji_kf'].full()
        cost_prev += self.Ji_N_fn(xi0=X_nominal[:,-1],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
                                 scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi],parai0=weight2,a0=i_admm)['Ji_Nf'].full()
        # else:
        #     for k in range(self.N):
        #         X_nominal[:,k:k+1] = np.reshape(scxi[k*self.nxi:(k+1)*self.nxi],(self.nxi,1))
        #         U_nominal[:,k:k+1]   = np.reshape(scui[k*self.nui:(k+1)*self.nui],(self.nui,1))
        #         cost_prev     += self.Ji_k_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],
        #                                 scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],
        #                                 refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,parai0=weight2,a0=i_admm)['Ji_kf'].full()
        #     cost_prev += self.Ji_N_fn(xi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
        #                          scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi],parai0=weight2,a0=i_admm)['Ji_Nf'].full()

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
            Vx[self.N] = self.lxiN_fn(xi0=X_nominal[:,self.N],
                                      refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],
                                      scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
                                      scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi],
                                      parai0=weight2,
                                      a0=i_admm)['lxiN_f'].full()
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
                try:
                    L, _jitter = self.try_cholesky(Quu_reg_k, jitter0=0.0)
                except LA.LinAlgError:
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
                                              refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,parai0=weight2,a0=i_admm)['Ji_kf'].full()
                cost_new   += self.Ji_N_fn(xi0=X_new[:,-1],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
                                          scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi], parai0=weight2,a0=i_admm)['Ji_Nf'].full()
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
            i_admm    = Parai[-1]
            opt_sol_i = self.DDP_Cable_ADMM_Subp1(xi_fb,Ref_xi,Ref_ui,weight2,scxi,scui,scxI,scuI,max_iter,e_tol,i_admm)
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
        k           = 0
        self.fi     = [] # list that stores all the quadrotor thruster limit constraints
        self.Gi1    = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 1st obstacle
        self.Gi2    = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 2nd obstacle
        self.Gij    = [] # list that stores all the safe inter-robot inequality constraints
        self.Gio    = []
        self.Di     = []
        self.sumfi  = 0 # barrier functions of the quadrotor thrust limit
        self.gco    = 0 # barrier functions of the safe collision-avoidance constraints on quadrotors' planar positions
        self.G_lo   = 0
        self.gij    = 0 # barrier functions of the safe inter-robot constraints on quadrotors' planar positions
        self.Tcon   = 0 # barrier functions of the tension magnitude constraints
        self.Uicon  = 0 # barrier functions of the cable control inputs
        self.din    = 0 # barrier functions of the cable direction normalization 
        self.Ei_Pil = 0
        dis_two     = 2*self.rl*math.sin(self.alpha/2) # distance between two neighbour cable attachment points
        num_dis     = int(self.cl0/(dis_two)) # discretization number
        
        po1   = (pl_k[0:2]-pob1).T@(pl_k[0:2]-pob1) - ((self.ro)+self.rq/2)**2
        self.po1_fn= Function('pl1_admm',[self.scxl],[po1],['scxl0'],['po1f_admm'])
        self.G_lo += -self.p_bar * log(po1)
        po2   = (pl_k[0:2]-pob2).T@(pl_k[0:2]-pob2) - ((self.ro)+self.rq/2)**2
        self.po2_fn= Function('pl2_admm',[self.scxl],[po2],['scxl0'],['po2f_admm'])
        self.G_lo += -self.p_bar * log(po2)
        for kc in range(1,num_dis+1):
            for i in range(int(self.nq)):
                ri     = np.reshape(self.ra[:,i],(3,1))
                ei     = ri/norm_2(ri)
                xi_k   = self.scxc[i*self.nxi:(i+1)*self.nxi]
                ui_k   = self.scuc[i*self.nui:(i+1)*self.nui]
                di_k   = xi_k[0:3] # world frame
                pib_k  = ri + (kc/(num_dis))*self.cl0*Rl_k.T@di_k # body frame
                if kc == (num_dis):
                    wi_k   = xi_k[3:6]
                    dwi_k  = xi_k[6:9] # cable angular acceleration
                    ti_k   = xi_k[12]  # cable tension magnitude
                    self.Tcon += -self.p_bar * log(ti_k-self.t_min)
                    self.Tcon += -self.p_bar * log(self.t_max-ti_k)
                    for i_u in range(self.nui):
                        self.Uicon += -self.p_bar * log(self.ui_bound - ui_k[i_u])
                        self.Uicon += -self.p_bar * log(ui_k[i_u] + self.ui_bound)
                    pi_k = pl_k + Rl_k@ri + (kc/(num_dis))*self.cl0*di_k # ith quadrotor's position in {I}
                    diso1 = pi_k[0:2]-pob1
                    go1  = diso1.T@diso1 - ((self.rq + self.ro)+self.rq)**2 # safe constriant between the obstacle 1 and the ith quadrotor, which should be positive. 
                    go1_fn = Function('go1'+str(i),[self.scxl,self.scxc],[go1],['scxl0','scxc0'],['go1f'+str(i)])
                    self.gco += -self.p_bar * log(go1)
                    self.Gi1 += [go1_fn]
                    diso2 = pi_k[0:2]-pob2
                    go2  = diso2.T@diso2 - ((self.rq + self.ro)+self.rq)**2 # safe constriant between the obstacle 2 and the ith quadrotor, which should be positive
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
                    awl_k   = LA.inv(self.Jl)@(Ml_k-S_wl_k@(self.Jl@wl_k))
                    S_awl_k = self.skew_sym(awl_k)
                    fi_k    = self.mq*(al_k+Rl_k@(S_wl_k@S_wl_k+S_awl_k)@ri+self.cl0*(S_dwi_k@di_k+S_wi_k@(S_wi_k@di_k))+self.g*self.ez) + di_k*ti_k
                    norm_fi = fi_k.T@fi_k
                    self.sumfi += -self.p_bar * log(self.fmax**2-norm_fi)
                    norm_fi_fn = Function('norm_f'+str(i),[self.scxl,self.scul,self.scxc,self.scuc],[norm_fi],['scxl0','scul0','scxc0','scuc0'],['norm_ff'+str(i)])
                    self.fi += [norm_fi_fn]
                    ti_kb = Rl_k.T@di_k*ti_k # cable tension vector in {B}
                    tc_k[i*3:(i+1)*3] = ti_kb
                    # cross cable safe constraintss
                    eiPil= ei[0:2].T@pib_k[0:2]
                    eiPil_fn = Function('gc'+str(i),[self.scxl,self.scxc],[eiPil],['scxl0','scxc0'],['gcf'+str(i)])
                    self.Ei_Pil += -self.p_bar * log(eiPil + self.rl)
                    self.Ei_Pil += -self.p_bar * log(self.cl0+self.rl - eiPil)
                    self.Gio    += [eiPil_fn ]

                for j in range(i+1,int(self.nq)): # safe inter-robot separation constraints
                    xj_k   = self.scxc[j*self.nxi:(j+1)*self.nxi]
                    dj_k   = xj_k[0:3]
                    rj     = np.reshape(self.ra[:,j],(3,1))
                    pjb_k  = rj + (kc/(num_dis))*self.cl0*Rl_k.T@dj_k # body frame
                    disij  = pib_k[0:2]-pjb_k[0:2]
                    gij    = disij.T@disij - (kc/(num_dis)*4*self.rq)**2 # 4rq in training
                    self.gij += -self.p_bar * log(gij)
                    gij_fn = Function('g'+str(k),[self.scxl,self.scxc],[gij],['scxl0','scxc0'],['gf'+str(k)])
                    self.Gij += [gij_fn]
                
                    k     += 1
            
        # control consensus constraint that maps tension forces to the load control wrench
        wrench   = vertcat(Rl_k.T@Fl_k,Ml_k) # body frame
        W_cons   = self.Pt@tc_k - wrench
        self.h_wcons  = 1/(2*self.p_bar)*W_cons.T@W_cons
        self.W_cons_fn = Function('W_cons',[self.scxl,self.scul,self.scxc],[W_cons],['scxl0','scul0','scxc0'],['W_consf'])


    def SetADMMSubP2_SoftCost_k(self):
        # at each step k
        self.J_2_soft_k    = self.Jl_P2_k  + self.gco + self.gij + self.Tcon + self.ql_n + self.din + self.sumfi + self.h_wcons + self.Uicon + self.G_lo + self.Ei_Pil
        for i in range(int(self.nq)):
            xi      = self.xc[i*self.nxi:(i+1)*self.nxi]   # cable primal state
            scxi    = self.scxc[i*self.nxi:(i+1)*self.nxi] # safe copy state of each cable
            scxI    = self.scxC[i*self.nxi:(i+1)*self.nxi] # Lagrangian multiplier
            ui      = self.uc[i*self.nui:(i+1)*self.nui]   # cable primal control
            scui    = self.scuc[i*self.nui:(i+1)*self.nui] # safe copy control of each cable
            scuI    = self.scuC[i*self.nui:(i+1)*self.nui] # Lagrangian multiplier
            resid_x = xi - scxi + scxI/self.pix_dis
            resid_u = ui - scui + scuI/self.piu_dis
            self.J_2_soft_k    += self.pix_dis/2*resid_x.T@resid_x + self.piu_dis/2*resid_u.T@resid_u
        self.J_2_soft_k_orig =   self.gco + self.gij + self.Tcon + self.ql_n + self.din + self.sumfi + self.h_wcons + self.Uicon + self.G_lo + self.Ei_Pil 
    

    def SetADMMSubP2_SoftCost_N(self):
        # at the terminal step N
        self.J_2_soft_N    = self.Jl_P2_N   + self.gco + self.gij + self.Tcon + self.ql_n + self.din + self.G_lo + self.Ei_Pil 
        for i in range(int(self.nq)):
            xi      = self.xc[i*self.nxi:(i+1)*self.nxi]   # cable primal state
            scxi    = self.scxc[i*self.nxi:(i+1)*self.nxi] # safe copy state of each cable
            scxI    = self.scxC[i*self.nxi:(i+1)*self.nxi] # Lagrangian multiplier
            resid_x = xi - scxi + scxI/self.pix_dis
            self.J_2_soft_N    += self.pix_dis/2*resid_x.T@resid_x 
        self.J_2_soft_N_orig =  self.gco + self.gij + self.Tcon + self.ql_n + self.din  + self.G_lo + self.Ei_Pil 


    

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
        g2         += [self.po1_fn(scxl0=scxl_k)['po1f_admm']]
        self.lbg2  += [1e-2]
        self.ubg2  += [1e4]
        g2         += [self.po2_fn(scxl0=scxl_k)['po2f_admm']]
        self.lbg2  += [1e-2]
        self.ubg2  += [1e4]
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
            # avoidance of cable-crossing constraints
            gio        = self.Gio[i](scxl0=scxl_k,scxc0=scxc_k)['gcf'+str(i)]
            g2        += [gio]
            self.lbg2 += [-self.rl]
            self.ubg2 += [self.rl+self.cl0] 
           
            
        
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
        # optsi2 = {}
        # optsi2['ipopt.tol'] = 1e-8
        # optsi2['ipopt.print_level'] = 0
        # optsi2['print_time'] = 0
        # optsi2['ipopt.warm_start_init_point']='yes'
        # optsi2['ipopt.max_iter']=2e3
        # optsi2['ipopt.acceptable_tol']=1e-8
        # optsi2['ipopt.mu_strategy']='adaptive'
        # optsi2['ipopt.diverging_iterates_tol'] = 1e8


        optsi2 = {
        'ipopt.print_level': 0,
        'print_time': 0,
        # 'ipopt.mu_strategy': 'adaptive',
        'ipopt.tol': 1e-8,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.max_iter': 2e3,
        'ipopt.warm_start_init_point': 'yes'
        # 'ipopt.diverging_iterates_tol':1e5,
        # 'ipopt.acceptable_iter':20,
        # 'ipopt.bound_relax_factor':1e-8 # this value is very important for stability!
        # 'ipopt.nlp_scaling_method': 'gradient-based',
        # 'ipopt.nlp_scaling_max_gradient': 100.0
        }


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
        g2N         += [self.po1_fn(scxl0=scxl_k)['po1f_admm']]
        self.lbg2N  += [1e-2]
        self.ubg2N  += [1e4]
        g2N         += [self.po2_fn(scxl0=scxl_k)['po2f_admm']]
        self.lbg2N  += [1e-2]
        self.ubg2N  += [1e4]
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
            # avoidance of cable-crossing constraints
            gio        = self.Gio[i](scxl0=scxl_k,scxc0=scxc_k)['gcf'+str(i)]
            g2N       += [gio]
            self.lbg2N += [-self.rl]
            self.ubg2N += [self.rl+self.cl0] 
        
        for k in range(len(self.Gij)):
            gij         = self.Gij[k](scxl0=scxl_k,scxc0=scxc_k)['gf'+str(k)]
            g2N        += [gij]
            self.lbg2N += [1e-2]
            self.ubg2N += [1e4]
        

        # create an NLP solver and solve it
        # optsi2N = {}
        # optsi2N['ipopt.tol'] = 1e-8
        # optsi2N['ipopt.print_level'] = 0
        # optsi2N['print_time'] = 0
        # optsi2N['ipopt.warm_start_init_point']='yes'
        # optsi2N['ipopt.max_iter']=2e3
        # optsi2N['ipopt.acceptable_tol']=1e-8
        # optsi2N['ipopt.mu_strategy']='adaptive'
        # optsi2['ipopt.bound_relax_factor']=1e-12
        # optsi2['ipopt.limited_memory_max_history'] = 20
        # optsi2['ipopt.nlp_scaling_method']='gradient-based'
        # optsi2['ipopt.limited_memory_initialization'] = 'scalar1'

        optsi2N = {
        'ipopt.print_level': 0,
        'print_time': 0,
        # 'ipopt.mu_strategy': 'adaptive',
        'ipopt.tol': 1e-8,
        'ipopt.acceptable_tol': 1e-8,# default value 1e-6
        'ipopt.max_iter': 2e3,
        'ipopt.warm_start_init_point': 'yes'
        # 'ipopt.diverging_iterates_tol':1e5, # default value 1e20
        # 'ipopt.acceptable_iter':20, # default value 15
        # 'ipopt.bound_relax_factor':1e-9 # default value 1e-8
        # 'ipopt.nlp_scaling_method': 'gradient-based',
        # 'ipopt.nlp_scaling_max_gradient': 100.0 # default value 1e2
        }

        prob2N = {'f': J2, 
                'x': vertcat(*w2N), 
                'p': Para2,
                'g': vertcat(*g2N)}
        self.solver2N = nlpsol('solver2N', 'ipopt', prob2N, optsi2N)  


    
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

    
    def ADMM_SubP3(self,xl_traj,scxl_traj,scxL_traj,ul_traj,scul_traj,scuL_traj,xc_traj,scxc_traj,scxC_traj,uc_traj,scuc_traj,scuC_traj,px,pu,gammax,gammau,pix,piu,gammaix,gammaiu,ADMM_max,i_admm):
        scxL_traj_new = np.zeros((self.N+1,self.nxl))
        scuL_traj_new = np.zeros((self.N,self.nul))
        scxC_traj_new = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        scuC_traj_new = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))]
        px_dis        = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis        = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        pix_dis       = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis       = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        for k in range(self.N):
            scxL_new  = scxL_traj[k,:] + px_dis*(xl_traj[k,:] - scxl_traj[k,:])
            scuL_new  = scuL_traj[k,:] + pu_dis*(ul_traj[k,:] - scul_traj[k,:])
            scxL_traj_new[k:k+1,:] = scxL_new
            scuL_traj_new[k:k+1,:] = scuL_new
            for i in range(int(self.nq)):
                scxI_new = scxC_traj[i][k,:] + pix_dis*(xc_traj[i][k,:] - scxc_traj[i][k,:])
                scxC_traj_new[i][k:k+1,:] = scxI_new
                scuI_new = scuC_traj[i][k,:] + piu_dis*(uc_traj[i][k,:] - scuc_traj[i][k,:])
                scuC_traj_new[i][k:k+1,:] = scuI_new
        #----terminal-----#
        scxL_new  = scxL_traj[self.N,:] + px_dis*(xl_traj[self.N,:] - scxl_traj[self.N,:])
        scxL_traj_new[self.N:self.N+1,:] = scxL_new
        for i in range(int(self.nq)):
            scxI_new = scxC_traj[i][self.N,:] + pix_dis*(xc_traj[i][self.N,:] - scxc_traj[i][self.N,:])
            scxC_traj_new[i][self.N:self.N+1,:] = scxI_new

        opt_sol3 = {"scxL_traj_new":scxL_traj_new,
                    "scuL_traj_new":scuL_traj_new,
                    "scxC_traj_new":scxC_traj_new,
                    "scuC_traj_new":scuC_traj_new
                    }
        
        return opt_sol3
    
    def system_derivatives_SubP3_ADMM(self):
        scxL_update = self.px_dis*(self.xl - self.scxl)
        scuL_update = self.pu_dis*(self.ul - self.scul)
        scxC_update = self.pix_dis*(self.xc - self.scxc)
        scuC_update = self.piu_dis*(self.uc - self.scuc)
        self.dscxL_updatedp    = jacobian(scxL_update,self.P_auto)
        self.dscxL_updatedp_fn = Function('dscxL_updatedp',[self.xl,self.scxl,self.para_l,self.a],[self.dscxL_updatedp],['xl0','scxl0','paral0','a0'],['dscxL_updatedp_f'])
        self.dscuL_updatedp    = jacobian(scuL_update,self.P_auto)
        self.dscuL_updatedp_fn = Function('dscuL_updatedp',[self.ul,self.scul,self.para_l,self.a],[self.dscuL_updatedp],['ul0','scul0','paral0','a0'],['dscuL_updatedp_f'])
        self.dscxC_updatedp    = jacobian(scxC_update,self.P_auto)
        self.dscxC_updatedp_fn = Function('dscxC_updatedp',[self.xc,self.scxc,self.para_i,self.a],[self.dscxC_updatedp],['xc0','scxc0','parai0','a0'],['dscxC_updatedp_f'])
        self.dscuC_updatedp    = jacobian(scuC_update,self.P_auto)
        self.dscuC_updatedp_fn = Function('dscuC_updatedp',[self.uc,self.scuc,self.para_i,self.a],[self.dscuC_updatedp],['uc0','scuc0','parai0','a0'],['dscuC_updatedp_f'])


    def Get_AuxSys_SubP3(self,opt_sol1_l,opt_sol1_c,opt_sol2,weight1,weight2,i_admm):
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
            dscxL_updatedp[k] = self.dscxL_updatedp_fn(xl0=xl_k,scxl0=scxl_k,paral0=weight1,a0=i_admm)['dscxL_updatedp_f'].full()
            dscuL_updatedp[k] = self.dscuL_updatedp_fn(ul0=ul_k,scul0=scul_k,paral0=weight1,a0=i_admm)['dscuL_updatedp_f'].full()
            dscxC_updatedp[k] = self.dscxC_updatedp_fn(xc0=xc_k,scxc0=scxc_k,parai0=weight2,a0=i_admm)['dscxC_updatedp_f'].full()
            dscuC_updatedp[k] = self.dscuC_updatedp_fn(uc0=uc_k,scuc0=scuc_k,parai0=weight2,a0=i_admm)['dscuC_updatedp_f'].full()
        xl_N     = xl[-1,:]
        scxl_N   = scxl[-1,:]
        xc_N     = np.concatenate([xc_list[i][-1,:] for i in range(int(self.nq))])
        scxc_N   = np.concatenate([scxc_list[i][-1,:] for i in range(int(self.nq))])
        dscxL_updatedp[self.N]= self.dscxL_updatedp_fn(xl0=xl_N,scxl0=scxl_N,paral0=weight1,a0=i_admm)['dscxL_updatedp_f'].full()
        dscxC_updatedp[self.N]= self.dscxC_updatedp_fn(xc0=xc_N,scxc0=scxc_N,parai0=weight2,a0=i_admm)['dscxC_updatedp_f'].full()

        auxSys3 = {
            "dscxL_updatedp":dscxL_updatedp,
            "dscuL_updatedp":dscuL_updatedp,
            "dscxC_updatedp":dscxC_updatedp,
            "dscuC_updatedp":dscuC_updatedp
        }
        return auxSys3
    



    def ADMM_forward_MPC(self,Ref_xl,Ref_ul,ref_xq,ref_uq,xl_fb,xq_fb,paral,paraC,max_iter_ADMM):
        # initial guess of the safe copy variable trajectories
        scxl_traj_tp = np.zeros(((self.N+1)*self.nxl))
        scul_traj_tp = Ref_ul
        for k in range(self.N):
            # scxl_traj_tp[k*self.nxl:(k+1)*self.nxl] = Ref_xl[k*self.nxl:(k+1)*self.nxl]
            scxl_traj_tp[(k)*self.nxl:(k+1)*self.nxl] = np.reshape(self.model_l_fn(xl0=scxl_traj_tp[k*self.nxl:(k+1)*self.nxl],ul0=Ref_ul[k*self.nul:(k+1)*self.nul])['mdynlf'].full(),self.nxl) # start from a bad initial state
        # scxl_traj_tp[self.N*self.nxl:(self.N+1)*self.nxl] = Ref_xl[self.N*self.nxl:(self.N+1)*self.nxl]
        scxc_traj_tp = [np.zeros((self.N+1)*self.nxi) for _ in range(int(self.nq))]
        scuc_traj_tp = [np.zeros(self.N*self.nui)  for _ in range(int(self.nq))]
        for i in range(int(self.nq)):
            for k in range(self.N):
                # scxc_traj_tp[i][k*self.nxi:(k+1)*self.nxi] = ref_xq[i][k*self.nxi:(k+1)*self.nxi]
                scxc_traj_tp[i][(k)*self.nxi:(k+1)*self.nxi] = np.reshape(self.model_i_fn(xi0=scxc_traj_tp[i][k*self.nxi:(k+1)*self.nxi],ui0=ref_uq[i*self.nui:(i+1)*self.nui])['mdynif'].full(),self.nxi)   #ref_xq[i][k*self.nxi:(k+1)*self.nxi]
                scuc_traj_tp[i][k*self.nui:(k+1)*self.nui] = ref_uq[i*self.nui:(i+1)*self.nui]
            # scxc_traj_tp[i][self.N*self.nxi:(self.N+1)*self.nxi] = ref_xq[i][self.N*self.nxi:(self.N+1)*self.nxi]
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
        
        for i_admm in range(self.max_iter_ADMM):
            # solve Subproblem 1-load (dynamic)
            start_time = TM.time()
            # opt_sol = self.MPC_Load_Planning_SubP1(Paral)
            opt_sol_l = self.DDP_Load_ADMM_Subp1(xl_fb,Ref_xl,Ref_ul,paral,scxl_traj_tp,scul_traj_tp,scxL_traj_tp,scuL_traj_tp,max_iter,e_tol,i_admm)
            mpctime = (TM.time() - start_time)*1000
            print('ADMM_iteration=',i_admm+1,"subprblem1_load:--- %s ms ---" % format(mpctime,'.2f'))
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
                parai     = np.concatenate((parai,[i_admm]))
                ParaC  += [parai]
            start_time = TM.time()
            opt_solc, OPt_sol_c = self.MPC_Cable_DDP_Planning_SubP1(ParaC)
            mpctime = (TM.time() - start_time)*1000
            px_dis      = self.open_loop_penalty(paral[-4],paral[-2],i_admm,max_iter_ADMM)
            pu_dis      = self.open_loop_penalty(paral[-3],paral[-1],i_admm,max_iter_ADMM)
            pix_dis     = self.open_loop_penalty(paraC[-4],paraC[-2],i_admm,max_iter_ADMM)
            piu_dis     = self.open_loop_penalty(paraC[-3],paraC[-1],i_admm,max_iter_ADMM)
            print('ADMM_iteration=',i_admm+1,"subproblem1_cables:--- %s ms ---" % format(mpctime,'.2f'),'current_plx=',px_dis,'current_plu=',pu_dis,'current_pix=',pix_dis,'current_piu=',piu_dis)
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
            opt_sol3    = self.ADMM_SubP3(xl_traj,scxl_traj,scxL_traj,ul_traj,scul_traj,scuL_traj,xc_traj,scxc_traj,scxC_traj,uc_traj,scuc_traj,scuC_traj,paral[-4],paral[-3],paral[-2],paral[-1],paraC[-4],paraC[-3],paraC[-2],paraC[-1],max_iter_ADMM,i_admm)
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
                s_xc += [parai[-4]*LA.norm(scxc_traj_tp_new[i]-scxc_traj_tp[i])]
                r_uc += [LA.norm(uc_traj_tp[i]-scuc_traj_tp_new[i])]
                s_uc += [parai[-3]*LA.norm(scuc_traj_tp_new[i]-scuc_traj_tp[i])]
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
                   "ul_traj":ul_traj,
                   "Kfbl_traj":Kfbl_traj,
                   "scxl_traj":scxl_traj,
                   "scul_traj":scul_traj,
                   "xc_traj":xc_traj,
                   "uc_traj":uc_traj,
                   "scxc_traj":scxc_traj,
                   "scuc_traj":scuc_traj
                    }
        
        return opt_sol, Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3
    

            
    def DDP_Load_Gradient(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, px,pu, gammax,gammau,ADMM_max, i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        px_dis     = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis     = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        S          = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        S[self.N]  = HxNp + scxL_grad[self.N] - px_dis*scxl_grad[self.N] # reduced to HxNp only in the single-agent problem
        v_FF       = self.N*[np.zeros((self.nul,self.n_Pauto))]
        xl_grad    = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] 
        ul_grad    = self.N*[np.zeros((self.nul,self.n_Pauto))]
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): # N-1, N-2, ....., 0
            Hxp_k    = Hxp[k] + scxL_grad[k] - px_dis*scxl_grad[k]
            Hup_k    = Hup[k] + scuL_grad[k] - pu_dis*scul_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            # S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
            S[k]     = Hxp_k + F[k].T@S[k+1] + K_fb[k].T@(Hup_k + G[k].T@S[k+1]) # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            ul_grad[k]  = K_fb[k]@xl_grad[k]+v_FF[k]
            xl_grad[k+1]= F[k]@xl_grad[k]+G[k]@ul_grad[k]

        grad_outl ={"xl_grad":xl_grad,
                   "ul_grad":ul_grad
                }
        
        return grad_outl
    
    
    def Cao_Load_Gradient_s(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, px,pu, gammax,gammau,ADMM_max,i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        px_dis     = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis     = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        S           = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] # Vxp
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        S[self.N]   = HxNp + scxL_grad[self.N] - px_dis*scxl_grad[self.N]
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        v_FF        = self.N*[np.zeros((self.nul,self.n_Pauto))]
        xl_grad     = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] 
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        ul_grad     = self.N*[np.zeros((self.nul,self.n_Pauto))]
        
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): # N-1, N-2,...,0
            Hpp_k    = np.zeros((self.n_Pauto,self.n_Pauto))
            Hxp_k    = Hxp[k] + scxL_grad[k] - px_dis*scxl_grad[k]
            Hup_k    = Hup[k] + scuL_grad[k] - pu_dis*scul_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            Vpp[k]   = Hpp_k + Vpp[k+1] + (Hup_k + G[k].T@S[k+1]).T@v_FF[k] # the augmented Riccati recursion, which is redundant
            # S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
            S[k]     = Hxp_k + F[k].T@S[k+1] + K_fb[k].T@(Hup_k + G[k].T@S[k+1]) # s[0] not used
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
    
    def Cao_Load_Gradient(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, px,pu,gammax,gammau,ADMM_max,i_admm):
        # solve the augmented optimal problem using one-step DDP recursion
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysl['HxxN'], auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        # Vyy      = (self.N+1)*[np.zeros((self.n_Pauto+self.nxl,self.n_Pauto+self.nxl))] # a large matrix, leading to significant computation cost
        # we decompose Vyy into four smaller blocks
        px_dis     = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis     = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
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
        Vpx[self.N] = (HxNp + scxL_grad[self.N] - px_dis*scxl_grad[self.N]).T
        Vxp[self.N] = HxNp + scxL_grad[self.N] - px_dis*scxl_grad[self.N]
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
            Hpx_k    = (Hxp[k]+ scxL_grad[k] - px_dis*scxl_grad[k]).T
            # Hxp_k    = Hxp[k]+ scxL_grad[k] - dis_rn*p1*scxl_grad[k]
            Hxx_k    = Hxx[k]
            Hup_k    = Hup[k]+ scuL_grad[k] - pu_dis*scul_grad[k]
            Quu_k    = Huu[k]+G[k].T@Vxx[k+1]@G[k] 
            invQuu_k = LA.inv(Quu_k)
            Kfb_p[k] = -invQuu_k@(Hup_k+G[k].T@Vxp[k+1]) 
            Kfb_x[k] = -invQuu_k@(Hxu[k].T+G[k].T@Vxx[k+1]@F[k])
            Vpp[k]   = Vpp[k+1] + (Hup_k.T+Vpx[k+1]@G[k])@Kfb_p[k]
            Vpx[k]   = Hpx_k + Vpx[k+1]@F[k] + Kfb_p[k].T@(Hxu[k]+F[k].T@Vxx[k+1]@G[k]).T
            # Vxp[k]   = Hxp_k + F[k].T@Vxp[k+1] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_p[k]
            Vxp[k]   = Vpx[k].T
            Vxx[k]   = Hxx_k + F[k].T@Vxx[k+1]@F[k] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_x[k] # large matrix multiplication will take longer time!!!

        for k in range(self.N):
            ul_grad[k]   = Kfb_p[k]@p_grad[k] + Kfb_x[k]@xl_grad[k]
            xl_grad[k+1] = F[k]@xl_grad[k]+G[k]@ul_grad[k]
            p_grad[k+1]  = p_grad[k]
        grad_out_cao ={"xl_grad":xl_grad,
                   "ul_grad":ul_grad,
                   "p_grad":p_grad
                }
        
        return grad_out_cao
    

    def PDP_Load_Gradient(self,opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, px,pu, gammax,gammau,ADMM_max,i_admm):
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysl['HxxN'], auxSysl['HxNp'], auxSysl['Hxp'], auxSysl['Hup']
        px_dis      = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis      = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        P           = (self.N+1)*[np.zeros((self.nxl,self.nxl))]
        S           = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        A           = self.N*[np.zeros((self.nxl,self.nxl))]
        R           = self.N*[np.zeros((self.nxl,self.nxl))]
        M_p         = self.N*[np.zeros((self.nxl,self.n_Pauto))]
        invHuu      = self.N*[np.zeros((self.nul,self.nul))]
        PinvIRP     = self.N*[np.zeros((self.nxl,self.nxl))]
        P[self.N]   = HxxN
        S[self.N]   = HxNp  + scxL_grad[self.N] - px_dis*scxl_grad[self.N]
        xl_grad     = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))] 
        ul_grad     = self.N*[np.zeros((self.nul,self.n_Pauto))]
        I           = np.identity(self.nxl)
        Iu          = np.identity(self.nul)
        for k in reversed(range(self.N)):# N-1, N-2,...,0
            P_next      = P[k+1]
            S_next      = S[k+1]
            invHuu[k]   = LA.inv(Huu[k])
            GinvHuu     = G[k]@invHuu[k]
            HxuinvHuu   = Hxu[k]@invHuu[k]
            A[k]        = F[k]-GinvHuu@Hxu[k].T
            R[k]        = GinvHuu@G[k].T
            M_p[k]      = -GinvHuu@(Hup[k] + scuL_grad[k] - pu_dis*scul_grad[k])
            Q_k         = Hxx[k]-HxuinvHuu@Hxu[k].T
            N_p_k       = Hxp[k]+ scxL_grad[k] - px_dis*scxl_grad[k] - HxuinvHuu@(Hup[k] + scuL_grad[k] - pu_dis*scul_grad[k])
            PinvIRP[k]  = P_next@LA.inv(I+R[k]@P_next)
            P_curr      = Q_k + A[k].T@PinvIRP[k]@A[k]
            S_curr      = A[k].T@PinvIRP[k]@(M_p[k] - R[k]@S_next) + A[k].T@S_next + N_p_k
            P[k]        = P_curr
            S[k]        = S_curr
        
        for k in range(self.N):
            ul_grad[k]  = -invHuu[k]@((Hxu[k].T+G[k].T@PinvIRP[k]@A[k])@xl_grad[k] + G[k].T@PinvIRP[k]@(M_p[k]- R[k]@ S[k+1]) + G[k].T@S[k+1] + (Hup[k] + scuL_grad[k] - pu_dis*scul_grad[k]))
            xl_grad[k+1] = F[k]@xl_grad[k] + G[k]@ul_grad[k]

        grad_out ={"xl_grad":xl_grad,
                   "ul_grad":ul_grad
                }
        
        return grad_out
    

    
    def DDP_Cable_Gradient(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix,piu, gammaix,gammaiu,ADMM_max, i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        pix_dis     = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis     = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        S           = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))]
        S[self.N]   = HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N] # reduced to HxNp only in the single-agent problem
        v_FF        = self.N*[np.zeros((self.nui,self.n_Pauto))]
        xi_grad     = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))]
        ui_grad     = self.N*[np.zeros((self.nui,self.n_Pauto))]
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): 
            Hxp_k    = Hxp[k] + scxI_grad[k] - pix_dis*scxi_grad[k]
            Hup_k    = Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            # S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
            S[k]     = Hxp_k + F[k].T@S[k+1] + K_fb[k].T@(Hup_k + G[k].T@S[k+1]) # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            ui_grad[k]  = K_fb[k]@xi_grad[k]+v_FF[k]
            xi_grad[k+1]= F[k]@xi_grad[k]+G[k]@ui_grad[k]

        grad_outi ={"xi_grad":xi_grad,
                   "ui_grad":ui_grad
                }
        
        return grad_outi
    
    def Cao_Cable_Gradient_s(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix,piu, gammaix,gammaiu,ADMM_max,i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        pix_dis     = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis     = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        S           = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))] # Vxp
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        S[self.N]   = HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        v_FF        = self.N*[np.zeros((self.nui,self.n_Pauto))]
        xi_grad     = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))]
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        ui_grad     = self.N*[np.zeros((self.nui,self.n_Pauto))]
        
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): # N-1, N-2,...,0
            Hpp_k    = np.zeros((self.n_Pauto,self.n_Pauto))
            Hxp_k    = Hxp[k] + scxI_grad[k] - pix_dis*scxi_grad[k]
            Hup_k    = Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            Vpp[k]   = Hpp_k + Vpp[k+1] + (Hup_k + G[k].T@S[k+1]).T@v_FF[k] # the augmented Riccati recursion, which is redundant
            # S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
            S[k]     = Hxp_k + F[k].T@S[k+1] + K_fb[k].T@(Hup_k + G[k].T@S[k+1]) # s[0] not used
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

    def Cao_Cable_Gradient(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix,piu,gammaix,gammaiu,ADMM_max,i_admm):
        # solve the augmented optimal problem using one-step DDP recursion
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysi['HxxN'], auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        # Vyy      = (self.N+1)*[np.zeros((self.n_Pauto+self.nxl,self.n_Pauto+self.nxl))] # a large matrix, leading to significant computation cost
        # we decompose Vyy into four smaller blocks
        pix_dis     = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis     = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
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
        Vpx[self.N] = (HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]).T
        Vxp[self.N] = HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]
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
            Hpx_k    = (Hxp[k]+ scxI_grad[k] - pix_dis*scxi_grad[k]).T
            # Hxp_k    = Hxp[k]+ scxL_grad[k] - dis_rn*p1*scxl_grad[k]
            Hxx_k    = Hxx[k]
            Hup_k    = Hup[k]+ scuI_grad[k] - piu_dis*scui_grad[k]
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
    

    
    def PDP_Cable_Gradient(self,opt_sol,auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix,piu, gammaix,gammaiu,ADMM_max,i_admm):
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysi['HxxN'], auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        pix_dis     = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis     = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        P           = (self.N+1)*[np.zeros((self.nxi,self.nxi))]
        S           = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))]
        A           = self.N*[np.zeros((self.nxi,self.nxi))]
        R           = self.N*[np.zeros((self.nxi,self.nxi))]
        M_p         = self.N*[np.zeros((self.nxi,self.n_Pauto))]
        invHuu      = self.N*[np.zeros((self.nui,self.nui))]
        PinvIRP     = self.N*[np.zeros((self.nxi,self.nxi))]
        P[self.N]   = HxxN
        S[self.N]   = HxNp  + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]
        xi_grad     = (self.N+1)*[np.zeros((self.nxi,self.n_Pauto))]
        ui_grad     = self.N*[np.zeros((self.nui,self.n_Pauto))]
        I           = np.identity(self.nxi)
        for k in reversed(range(self.N)):
            P_next      = P[k+1]
            S_next      = S[k+1]
            invHuu[k]   = LA.inv(Huu[k])
            GinvHuu     = G[k]@invHuu[k]
            HxuinvHuu   = Hxu[k]@invHuu[k]
            A[k]        = F[k]-GinvHuu@Hxu[k].T
            R[k]        = GinvHuu@G[k].T
            M_p[k]      = -GinvHuu@(Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k])
            Q_k         = Hxx[k]-HxuinvHuu@Hxu[k].T
            N_p_k       = Hxp[k]+ scxI_grad[k] - pix_dis*scxi_grad[k] - HxuinvHuu@(Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k])
            PinvIRP[k]  = P_next@LA.inv(I+R[k]@P_next)
            P_curr      = Q_k + A[k].T@PinvIRP[k]@A[k]
            S_curr      = A[k].T@PinvIRP[k]@(M_p[k] - R[k]@S_next) + A[k].T@S_next + N_p_k
            P[k]        = P_curr
            S[k]        = S_curr
        
        for k in range(self.N):
            ui_grad[k]  = -invHuu[k]@((Hxu[k].T+G[k].T@PinvIRP[k]@A[k])@xi_grad[k] + G[k].T@PinvIRP[k]@(M_p[k]- R[k]@ S[k+1]) + G[k].T@S[k+1] + (Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k]))
            xi_grad[k+1] = F[k]@xi_grad[k] + G[k]@ui_grad[k]

        grad_out ={"xi_grad":xi_grad,
                   "ui_grad":ui_grad
                }
        
        return grad_out
    
    

    def SubP2_Gradient(self,auxSys2,grad_outl,grad_outc,scxL_grad,scuL_grad,scxC_grad,scuC_grad,px,pu,gammax,gammau,pix,piu,gammaix,gammaiu,ADMM_max,i_admm, prepared_batches=None):
        I_hess2      = np.identity(self.nxl+self.nxi*int(self.nq))
        scxl_grad    = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        scul_grad    = self.N*[np.zeros((self.nul,self.n_Pauto))]
        scxc_grad    = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        scuc_grad    = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]

        if prepared_batches is None:
            prepared_batches = self._prepare_subp2_batches(
                auxSys2, grad_outl, grad_outc, scxL_grad, scuL_grad, scxC_grad, scuC_grad,
                px, pu, gammax, gammau, pix, piu, gammaix, gammaiu, ADMM_max, i_admm
            )

        loop_start_time = TM.time()
        grad_subp2_batch = self._unshard_array_batch(
            self._block_tree_ready(self._run_subp2_horizon_solves(prepared_batches)),
            self.N
        )
        Grad2Time = (TM.time() - loop_start_time)*1000

        if self.enable_subp2_la_diagnostics:
            grad_subp2_batch_la = self._batched_subp2_solve(
                prepared_batches["L_hessian_batch"],
                prepared_batches["L_trajp_batch"],
                prepared_batches["reg_batch"],
                prepared_batches["I_hessian_np"]
            )
            abs_diff = np.abs(np.asarray(grad_subp2_batch, dtype=np.float64) - np.asarray(grad_subp2_batch_la, dtype=np.float64))
            step_max = np.max(abs_diff.reshape(self.N, -1), axis=1)
            worst_step = int(np.argmax(step_max))
            print(
                'ADMM iteration:', i_admm + 1,
                'SubP2_jax_vs_LA_diag:',
                f"train_iter={self.debug_train_iter}",
                f"task_idx={self.debug_task_idx}",
                'max_abs_diff=%.6e' % float(np.max(abs_diff)),
                'mean_abs_diff=%.6e' % float(np.mean(abs_diff)),
                'worst_step=%d' % worst_step,
                'worst_step_max=%.6e' % float(step_max[worst_step]),
            )
        else:
            grad_subp2_batch_la = None

        for k in range(self.N):
            grad_subp2_k = grad_subp2_batch[k]
            scxl_grad[k] = grad_subp2_k[0:self.nxl,:]
            scul_grad[k] = grad_subp2_k[self.nxl:(self.nxl+self.nul),:]
            scxc_grad[k] = grad_subp2_k[(self.nxl+self.nul):(self.nxl+self.nul+self.nxi*int(self.nq)),:]
            scuc_grad[k] = grad_subp2_k[(self.nxl+self.nul+self.nxi*int(self.nq)):(self.nxl+self.nul+self.nxi*int(self.nq)+self.nui*int(self.nq)),:]

        if self.enable_subp2_la_diagnostics:
            scxl_grad_la = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
            scul_grad_la = self.N*[np.zeros((self.nul,self.n_Pauto))]
            scxc_grad_la = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
            scuc_grad_la = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
            for k in range(self.N):
                grad_subp2_k_la = grad_subp2_batch_la[k]
                scxl_grad_la[k] = grad_subp2_k_la[0:self.nxl,:]
                scul_grad_la[k] = grad_subp2_k_la[self.nxl:(self.nxl+self.nul),:]
                scxc_grad_la[k] = grad_subp2_k_la[(self.nxl+self.nul):(self.nxl+self.nul+self.nxi*int(self.nq)),:]
                scuc_grad_la[k] = grad_subp2_k_la[(self.nxl+self.nul+self.nxi*int(self.nq)):(self.nxl+self.nul+self.nxi*int(self.nq)+self.nui*int(self.nq)),:]

        start_time      = TM.time()
        L_hessian_N_sym = prepared_batches["terminal_hessian"] + prepared_batches["terminal_reg"]*I_hess2
        L, _jitter      = self.try_cholesky(L_hessian_N_sym, jitter0=0.0)
        grad_subp2_N    = self.chol_solve(L, -prepared_batches["terminal_trajp"])
        gradtimeSubP2_N = (TM.time() - start_time)*1000
        scxl_grad[self.N] = grad_subp2_N[0:self.nxl,:]
        scxc_grad[self.N] = grad_subp2_N[self.nxl:(self.nxl+self.nxi*int(self.nq)),:]
        if self.enable_subp2_la_diagnostics:
            scxl_grad_la[self.N] = grad_subp2_N[0:self.nxl,:]
            scxc_grad_la[self.N] = grad_subp2_N[self.nxl:(self.nxl+self.nxi*int(self.nq)),:]

        min_eigs = prepared_batches["min_eigs"] + [prepared_batches["terminal_min_eigval"]]
        eig_time = prepared_batches["eig_time"] + prepared_batches["terminal_eig_time"]
        subp2_axis_size = prepared_batches["subp2_axis_size"]
        subp2_effective_steps_per_device = self.N / max(subp2_axis_size, 1)
        print('min_eigen=',np.min(min_eigs)) 
        print('ADMM iteration:',i_admm+1,"Eig_time_terminal:--- %s ms ---" % format(eig_time,'.2f')) 
        print('ADMM iteration:',i_admm+1,"GSubP2_batch_time:--- %s ms ---" % format(Grad2Time,'.2f'))
        print('ADMM iteration:',i_admm+1,"GSubP2_terminal_solve_time:--- %s ms ---" % format(gradtimeSubP2_N,'.2f'))
        print('ADMM iteration:',i_admm+1,"SubP2_parallel_diag: selected_devices=%d available_devices=%d N=%d steps_per_device~=%.2f" % (subp2_axis_size, len(self.jax_subp2_devices), self.N, subp2_effective_steps_per_device))
        grad_out2 = {
                    "scxl_grad":scxl_grad,
                    "scul_grad":scul_grad,
                    "scxc_grad":scxc_grad,
                    "scuc_grad":scuc_grad
                    }
        if self.enable_subp2_la_diagnostics:
            grad_out2_la = {
                        "scxl_grad":scxl_grad_la,
                        "scul_grad":scul_grad_la,
                        "scxc_grad":scxc_grad_la,
                        "scuc_grad":scuc_grad_la
                        }
            grad_out2_la_diffs = {}
            for key in ("scxl_grad", "scul_grad", "scxc_grad", "scuc_grad"):
                diffs = [
                    np.max(
                        np.abs(
                            np.asarray(grad_out2[key][k], dtype=np.float64) -
                            np.asarray(grad_out2_la[key][k], dtype=np.float64)
                        )
                    )
                    for k in range(len(grad_out2[key]))
                ]
                grad_out2_la_diffs[key] = float(np.max(diffs)) if diffs else 0.0
            print(
                'ADMM iteration:', i_admm + 1,
                'SubP2_grad_out2_jax_vs_LA_diag:',
                f"train_iter={self.debug_train_iter}",
                f"task_idx={self.debug_task_idx}",
                'scxl_max=%.6e' % grad_out2_la_diffs["scxl_grad"],
                'scul_max=%.6e' % grad_out2_la_diffs["scul_grad"],
                'scxc_max=%.6e' % grad_out2_la_diffs["scxc_grad"],
                'scuc_max=%.6e' % grad_out2_la_diffs["scuc_grad"],
            )
        Grad2Time += gradtimeSubP2_N
        
        return grad_out2, Grad2Time

    def _prepare_subp2_batches(self, auxSys2, grad_outl, grad_outc, scxL_grad, scuL_grad, scxC_grad, scuC_grad,
                               px, pu, gammax, gammau, pix, piu, gammaix, gammaiu, ADMM_max, i_admm):
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
        px_dis       = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis       = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        pix_dis      = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis      = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        EigTime      = 0.0
        MIN_eigen    = []
        nq_int = int(self.nq)

        xl_grad_batch = [np.asarray(x, dtype=np.float64) for x in xl_grad[:self.N]]
        ul_grad_batch = [np.asarray(x, dtype=np.float64) for x in ul_grad[:self.N]]
        scxL_grad_batch = [np.asarray(x, dtype=np.float64) for x in scxL_grad[:self.N]]
        scuL_grad_batch = [np.asarray(x, dtype=np.float64) for x in scuL_grad]
        scxC_grad_batch = [np.asarray(x, dtype=np.float64) for x in scxC_grad[:self.N]]
        scuC_grad_batch = [np.asarray(x, dtype=np.float64) for x in scuC_grad]
        cable_state_grad_time_major = np.transpose(
            np.stack([np.asarray(grad_outc[i]['xi_grad'], dtype=np.float64) for i in range(nq_int)], axis=0),
            (1, 0, 2, 3)
        ).reshape(self.N + 1, self.nxi * nq_int, self.n_Pauto)
        cable_ctrl_grad_time_major = np.transpose(
            np.stack([np.asarray(grad_outc[i]['ui_grad'], dtype=np.float64) for i in range(nq_int)], axis=0),
            (1, 0, 2, 3)
        ).reshape(self.N, self.nui * nq_int, self.n_Pauto)

        if self.enable_subp2_prepare_diagnostics:
            xc_prepare_max = 0.0
            uc_prepare_max = 0.0
            trajp_prepare_max = 0.0

        for k in range(self.N):
            Lscxlscxl_k = np.asarray(Lscxlscxl[k], dtype=np.float64)
            Lscxlscul_k = np.asarray(Lscxlscul[k], dtype=np.float64)
            Lscxlscxc_k = np.asarray(Lscxlscxc[k], dtype=np.float64)
            Lscxlscuc_k = np.asarray(Lscxlscuc[k], dtype=np.float64)
            Lsculscul_k = np.asarray(Lsculscul[k], dtype=np.float64)
            Lsculscxc_k = np.asarray(Lsculscxc[k], dtype=np.float64)
            Lsculscuc_k = np.asarray(Lsculscuc[k], dtype=np.float64)
            Lscxcscxc_k = np.asarray(Lscxcscxc[k], dtype=np.float64)
            Lscxcscuc_k = np.asarray(Lscxcscuc[k], dtype=np.float64)
            Lscucscuc_k = np.asarray(Lscucscuc[k], dtype=np.float64)
            if self.enable_subp2_prepare_diagnostics:
                xc_grad_k_legacy = np.asarray(grad_outc[0]['xi_grad'][k], dtype=np.float64)
                uc_grad_k_legacy = np.asarray(grad_outc[0]['ui_grad'][k], dtype=np.float64)
                for i in range(1, nq_int):
                    xc_grad_k_legacy = np.vstack((xc_grad_k_legacy, np.asarray(grad_outc[i]['xi_grad'][k], dtype=np.float64)))
                    uc_grad_k_legacy = np.vstack((uc_grad_k_legacy, np.asarray(grad_outc[i]['ui_grad'][k], dtype=np.float64)))
                xc_prepare_max = max(
                    xc_prepare_max,
                    float(np.max(np.abs(cable_state_grad_time_major[k] - xc_grad_k_legacy)))
                )
                uc_prepare_max = max(
                    uc_prepare_max,
                    float(np.max(np.abs(cable_ctrl_grad_time_major[k] - uc_grad_k_legacy)))
                )
            L_hessian_k = np.block([
                [Lscxlscxl_k,   Lscxlscul_k,   Lscxlscxc_k,   Lscxlscuc_k],
                [Lscxlscul_k.T, Lsculscul_k,   Lsculscxc_k,   Lsculscuc_k],
                [Lscxlscxc_k.T, Lsculscxc_k.T, Lscxcscxc_k,   Lscxcscuc_k],
                [Lscxlscuc_k.T, Lsculscuc_k.T, Lscxcscuc_k.T, Lscucscuc_k],
            ])
            L_trajp_k = np.vstack([
                np.asarray(Lscxlp[k], dtype=np.float64) - px_dis*xl_grad_batch[k] - scxL_grad_batch[k],
                np.asarray(Lsculp[k], dtype=np.float64) - pu_dis*ul_grad_batch[k] - scuL_grad_batch[k],
                np.asarray(Lscxcp[k], dtype=np.float64) - pix_dis*cable_state_grad_time_major[k] - scxC_grad_batch[k],
                np.asarray(Lscucp[k], dtype=np.float64) - piu_dis*cable_ctrl_grad_time_major[k] - scuC_grad_batch[k],
            ])
            if self.enable_subp2_prepare_diagnostics:
                L_trajp_k_legacy = np.vstack([
                    np.asarray(Lscxlp[k], dtype=np.float64) - px_dis*xl_grad_batch[k] - scxL_grad_batch[k],
                    np.asarray(Lsculp[k], dtype=np.float64) - pu_dis*ul_grad_batch[k] - scuL_grad_batch[k],
                    np.asarray(Lscxcp[k], dtype=np.float64) - pix_dis*xc_grad_k_legacy - scxC_grad_batch[k],
                    np.asarray(Lscucp[k], dtype=np.float64) - piu_dis*uc_grad_k_legacy - scuC_grad_batch[k],
                ])
                trajp_prepare_max = max(
                    trajp_prepare_max,
                    float(np.max(np.abs(L_trajp_k - L_trajp_k_legacy)))
                )
            Lscxlscxl_ok = np.asarray(Lscxlscxl_o[k], dtype=np.float64)
            Lscxlscul_ok = np.asarray(Lscxlscul_o[k], dtype=np.float64)
            Lscxlscxc_ok = np.asarray(Lscxlscxc_o[k], dtype=np.float64)
            Lscxlscuc_ok = np.asarray(Lscxlscuc_o[k], dtype=np.float64)
            Lsculscul_ok = np.asarray(Lsculscul_o[k], dtype=np.float64)
            Lsculscxc_ok = np.asarray(Lsculscxc_o[k], dtype=np.float64)
            Lsculscuc_ok = np.asarray(Lsculscuc_o[k], dtype=np.float64)
            Lscxcscxc_ok = np.asarray(Lscxcscxc_o[k], dtype=np.float64)
            Lscxcscuc_ok = np.asarray(Lscxcscuc_o[k], dtype=np.float64)
            Lscucscuc_ok = np.asarray(Lscucscuc_o[k], dtype=np.float64)
            L_hessian_ko = np.block([
                [Lscxlscxl_ok,   Lscxlscul_ok,   Lscxlscxc_ok,   Lscxlscuc_ok],
                [Lscxlscul_ok.T, Lsculscul_ok,   Lsculscxc_ok,   Lsculscuc_ok],
                [Lscxlscxc_ok.T, Lsculscxc_ok.T, Lscxcscxc_ok,   Lscxcscuc_ok],
                [Lscxlscuc_ok.T, Lsculscuc_ok.T, Lscxcscuc_ok.T, Lscucscuc_ok],
            ])
            start_time = TM.time()
            min_eigval = float(np.min(LA.eigvalsh(L_hessian_ko)))
            EigTime += (TM.time() - start_time)*1000
            MIN_eigen.append(min_eigval)
            reg = -min_eigval + 1e-4 if min_eigval < 0 else 0.0
            if k == 0:
                L_hessian_batch = [L_hessian_k]
                L_trajp_batch = [L_trajp_k]
                reg_batch = [reg]
            else:
                L_hessian_batch.append(L_hessian_k)
                L_trajp_batch.append(L_trajp_k)
                reg_batch.append(reg)

        L_hessian_batch = np.stack(L_hessian_batch, axis=0)
        L_trajp_batch = np.stack(L_trajp_batch, axis=0)
        reg_batch = np.asarray(reg_batch, dtype=np.float64)

        subp2_axis_size = self._jax_subp2_device_count(self.N)
        subp2_devices = self.jax_subp2_devices[:subp2_axis_size]
        L_hessian_sharded, L_trajp_sharded, reg_sharded, _ = self._shard_subp2_solve_batches_fixed_axis(
            L_hessian_batch,
            L_trajp_batch,
            reg_batch,
            I_hessian,
            subp2_axis_size,
            devices=subp2_devices,
        )

        L_hessian_N = np.block([
                                [np.asarray(Lscxlscxl[self.N], dtype=np.float64),   np.asarray(Lscxlscxc[self.N], dtype=np.float64)],
                                [np.asarray(Lscxlscxc[self.N], dtype=np.float64).T, np.asarray(Lscxcscxc[self.N], dtype=np.float64)]
                                ])
        xc_grad_N   = grad_outc[0]['xi_grad'][self.N]
        for i in range(1,int(self.nq)):
            xc_grad_N = np.vstack((xc_grad_N,grad_outc[i]['xi_grad'][self.N]))
        L_trajp_N   = np.vstack([
                            np.asarray(Lscxlp[self.N], dtype=np.float64) - px_dis*np.asarray(xl_grad[self.N], dtype=np.float64) - np.asarray(scxL_grad[self.N], dtype=np.float64),
                            np.asarray(Lscxcp[self.N], dtype=np.float64) - pix_dis*np.asarray(xc_grad_N, dtype=np.float64) - np.asarray(scxC_grad[self.N], dtype=np.float64)
                            ])
        L_hessian_No = np.block([
                                [np.asarray(Lscxlscxl_o[self.N], dtype=np.float64),   np.asarray(Lscxlscxc_o[self.N], dtype=np.float64)],
                                [np.asarray(Lscxlscxc_o[self.N], dtype=np.float64).T, np.asarray(Lscxcscxc_o[self.N], dtype=np.float64)]
                                ])
        start_time = TM.time()
        terminal_min_eigval = float(np.min(LA.eigvalsh(L_hessian_No)))
        terminal_eig_time = (TM.time() - start_time)*1000
        terminal_reg = -terminal_min_eigval + 1e-4 if terminal_min_eigval < 0 else 0.0
        if self.enable_subp2_prepare_diagnostics:
            print(
                'SubP2_prepare_cable_grad_diag:',
                'train_iter=%s' % self.debug_train_iter,
                'task_idx=%s' % self.debug_task_idx,
                'xc_max=%.6e' % xc_prepare_max,
                'uc_max=%.6e' % uc_prepare_max,
                'trajp_max=%.6e' % trajp_prepare_max,
            )

        return {
            "L_hessian_batch": L_hessian_batch,
            "L_trajp_batch": L_trajp_batch,
            "reg_batch": reg_batch,
            "I_hessian_np": I_hessian,
            "L_hessian_sharded": L_hessian_sharded,
            "L_trajp_sharded": L_trajp_sharded,
            "reg_sharded": reg_sharded,
            "I_hessian_jax": jnp.asarray(I_hessian, dtype=jnp.float64),
            "subp2_axis_size": subp2_axis_size,
            "eig_time": EigTime,
            "min_eigs": MIN_eigen,
            "terminal_hessian": L_hessian_N,
            "terminal_trajp": L_trajp_N,
            "terminal_min_eigval": terminal_min_eigval,
            "terminal_eig_time": terminal_eig_time,
            "terminal_reg": terminal_reg,
        }
    

    def SubP3_Gradient(self,auxSys3,grad_outl,grad_outc,grad_out2,scxL_grad,scuL_grad,scxC_grad,scuC_grad,px,pu,gammax,gammau,pix,piu,gammaix,gammaiu,ADMM_max,i_admm, prepared_batches=None):
        xl_grad         = grad_outl['xl_grad']
        scxl_grad       = grad_out2['scxl_grad']
        dscxL_updatedp  = auxSys3['dscxL_updatedp']
        scxL_grad_new   = (self.N+1)*[np.zeros((self.nxl,self.n_Pauto))]
        scuL_grad_new   = self.N*[np.zeros((self.nul,self.n_Pauto))]
        scxC_grad_new   = (self.N+1)*[np.zeros((self.nxi*int(self.nq),self.n_Pauto))]
        scuC_grad_new   = self.N*[np.zeros((self.nui*int(self.nq),self.n_Pauto))]
        px_dis          = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis          = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        pix_dis         = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis         = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        nq = int(self.nq)

        if prepared_batches is None:
            prepared_batches = self._prepare_subp3_batches(
                grad_outl, grad_outc, grad_out2, scxL_grad, scuL_grad, scxC_grad, scuC_grad, auxSys3
            )
        if self.enable_subp3_parallel_diagnostics:
            if prepared_batches.get("subp3_batched_time_major", False):
                print('ADMM iteration:',i_admm+1,"SubP3_parallel_diag: kernel=batched_time_major N=%d nq=%d load_axis=horizon cable_axes=horizon,cable" % (self.N, nq))
            elif prepared_batches.get("subp3_combined_pmap", False):
                subp3_axis_size = int(prepared_batches["subp3_axis_size"])
                load_steps_per_device = self.N / max(subp3_axis_size, 1)
                cable_state_steps_per_device = (nq * (self.N + 1)) / max(subp3_axis_size, 1)
                cable_ctrl_steps_per_device = (nq * self.N) / max(subp3_axis_size, 1)
                print('ADMM iteration:',i_admm+1,"SubP3_parallel_diag: selected_devices=%d available_devices=%d N=%d nq=%d load_steps_per_device~=%.2f cable_state_steps_per_device~=%.2f cable_ctrl_steps_per_device~=%.2f" % (subp3_axis_size, len(self.jax_subp3_devices), self.N, nq, load_steps_per_device, cable_state_steps_per_device, cable_ctrl_steps_per_device))
            else:
                load_axis_size = int(prepared_batches["load_axis_size"])
                cable_axis_size = int(prepared_batches["cable_axis_size"])
                load_steps_per_device = self.N / max(load_axis_size, 1)
                if prepared_batches.get("cable_time_major", False):
                    cable_state_steps_per_device = (self.N + 1) / max(cable_axis_size, 1)
                    cable_ctrl_steps_per_device = self.N / max(cable_axis_size, 1)
                else:
                    cable_state_steps_per_device = (nq * (self.N + 1)) / max(cable_axis_size, 1)
                    cable_ctrl_steps_per_device = (nq * self.N) / max(cable_axis_size, 1)
                print('ADMM iteration:',i_admm+1,"SubP3_parallel_diag: load_devices=%d available_load_devices=%d cable_devices=%d available_cable_devices=%d N=%d nq=%d load_steps_per_device~=%.2f cable_state_steps_per_device~=%.2f cable_ctrl_steps_per_device~=%.2f" % (load_axis_size, len(self.jax_subp3_load_devices), cable_axis_size, len(self.jax_subp3_cable_devices), self.N, nq, load_steps_per_device, cable_state_steps_per_device, cable_ctrl_steps_per_device))

        if prepared_batches.get("subp3_combined_pmap", False):
            subp3_exec_start = TM.time()
            outputs = self._block_tree_ready(
                self._run_subp3_updates(
                    prepared_batches,
                    np.float64(px_dis),
                    np.float64(pu_dis),
                    np.float64(pix_dis),
                    np.float64(piu_dis)
                )
            )
            self._last_subp3_exec_time_ms = (TM.time() - subp3_exec_start) * 1000

            def _unshard_output(name, batch_size):
                output = outputs[name]
                return np.asarray(output, dtype=np.float64).reshape(
                    (output.shape[0] * output.shape[1],) + tuple(output.shape[2:])
                )[:batch_size]

            load_state_batch_np = _unshard_output("load_state", self.N)
            load_ctrl_batch_np = _unshard_output("load_ctrl", self.N)
            cable_state_batch_np = _unshard_output("cable_state", nq * (self.N + 1))
            cable_ctrl_batch_np = _unshard_output("cable_ctrl", nq * self.N)

            cable_state_time_major = cable_state_batch_np.reshape(
                self.N + 1, nq, self.nxi, self.n_Pauto
            ).reshape(self.N + 1, self.nxi * nq, self.n_Pauto)
            cable_ctrl_time_major = cable_ctrl_batch_np.reshape(
                self.N, nq, self.nui, self.n_Pauto
            ).reshape(self.N, self.nui * nq, self.n_Pauto)

            scxL_grad_new = [load_state_batch_np[k] for k in range(self.N)]
            scxL_grad_new += [scxL_grad[self.N] + px_dis * (xl_grad[self.N] - scxl_grad[self.N]) + dscxL_updatedp[self.N]]
            scuL_grad_new = [load_ctrl_batch_np[k] for k in range(self.N)]
            scxC_grad_new = [cable_state_time_major[k] for k in range(self.N + 1)]
            scuC_grad_new = [cable_ctrl_time_major[k] for k in range(self.N)]

            grad_out3 = {
                "scxL_grad":scxL_grad_new,
                "scuL_grad":scuL_grad_new,
                "scxC_grad":scxC_grad_new,
                "scuC_grad":scuC_grad_new
            }

            return grad_out3

        if prepared_batches.get("subp3_batched_time_major", False):
            outputs = self._block_tree_ready(
                self._batched_subp3_time_major_updates(
                    prepared_batches["load_state_pack_batch"],
                    prepared_batches["load_ctrl_pack_batch"],
                    prepared_batches["cable_state_pack_batch"],
                    prepared_batches["cable_ctrl_pack_batch"],
                    np.float64(px_dis),
                    np.float64(pu_dis),
                    np.float64(pix_dis),
                    np.float64(piu_dis)
                )
            )
            load_state_batch_np = np.asarray(outputs["load_state"], dtype=np.float64)
            load_ctrl_batch_np = np.asarray(outputs["load_ctrl"], dtype=np.float64)
            cable_state_time_major = np.asarray(outputs["cable_state"], dtype=np.float64).reshape(
                self.N + 1, nq, self.nxi, self.n_Pauto
            ).reshape(self.N + 1, self.nxi * nq, self.n_Pauto)
            cable_ctrl_time_major = np.asarray(outputs["cable_ctrl"], dtype=np.float64).reshape(
                self.N, nq, self.nui, self.n_Pauto
            ).reshape(self.N, self.nui * nq, self.n_Pauto)

            scxL_grad_new = [load_state_batch_np[k] for k in range(self.N)]
            scxL_grad_new += [scxL_grad[self.N] + px_dis * (xl_grad[self.N] - scxl_grad[self.N]) + dscxL_updatedp[self.N]]
            scuL_grad_new = [load_ctrl_batch_np[k] for k in range(self.N)]
            scxC_grad_new = [cable_state_time_major[k] for k in range(self.N + 1)]
            scuC_grad_new = [cable_ctrl_time_major[k] for k in range(self.N)]

            grad_out3 = {
                "scxL_grad":scxL_grad_new,
                "scuL_grad":scuL_grad_new,
                "scxC_grad":scxC_grad_new,
                "scuC_grad":scuC_grad_new
            }

            return grad_out3

        subp3_executor = self._subp3_executor if self._subp3_has_disjoint_device_groups() else None
        if subp3_executor is not None:
            load_future = subp3_executor.submit(
                self._run_subp3_load_updates,
                prepared_batches,
                np.float64(px_dis),
                np.float64(pu_dis)
            )
            cable_future = subp3_executor.submit(
                self._run_subp3_cable_updates,
                prepared_batches,
                np.float64(pix_dis),
                np.float64(piu_dis)
            )
            load_outputs_batch = load_future.result()
            cable_outputs_batch = cable_future.result()
        else:
            load_outputs_batch = self._run_subp3_load_updates(
                prepared_batches,
                np.float64(px_dis),
                np.float64(pu_dis)
            )
            cable_outputs_batch = self._run_subp3_cable_updates(
                prepared_batches,
                np.float64(pix_dis),
                np.float64(piu_dis)
            )

        cable_time_major = prepared_batches.get("cable_time_major", False)
        cable_state_batch_size = self.N + 1 if cable_time_major else nq * (self.N + 1)
        cable_ctrl_batch_size = self.N if cable_time_major else nq * self.N
        if subp3_executor is not None:
            load_unshard_future = subp3_executor.submit(
                self._unshard_subp3_flat_outputs, load_outputs_batch, self.N, self.N
            )
            cable_unshard_future = subp3_executor.submit(
                self._unshard_subp3_flat_outputs, cable_outputs_batch, cable_state_batch_size, cable_ctrl_batch_size
            )
            load_state_batch_np, load_ctrl_batch_np = load_unshard_future.result()
            cable_state_batch_np, cable_ctrl_batch_np = cable_unshard_future.result()
        else:
            load_state_batch_np, load_ctrl_batch_np = self._unshard_subp3_flat_outputs(
                load_outputs_batch, self.N, self.N
            )
            cable_state_batch_np, cable_ctrl_batch_np = self._unshard_subp3_flat_outputs(
                cable_outputs_batch, cable_state_batch_size, cable_ctrl_batch_size
            )

        if cable_time_major:
            cable_state_time_major = cable_state_batch_np.reshape(
                self.N + 1, nq, self.nxi, self.n_Pauto
            ).reshape(self.N + 1, self.nxi * nq, self.n_Pauto)
            cable_ctrl_time_major = cable_ctrl_batch_np.reshape(
                self.N, nq, self.nui, self.n_Pauto
            ).reshape(self.N, self.nui * nq, self.n_Pauto)
        else:
            cable_state_time_major = cable_state_batch_np.reshape(
                self.N + 1, nq, self.nxi, self.n_Pauto
            ).reshape(self.N + 1, self.nxi * nq, self.n_Pauto)
            cable_ctrl_time_major = cable_ctrl_batch_np.reshape(
                self.N, nq, self.nui, self.n_Pauto
            ).reshape(self.N, self.nui * nq, self.n_Pauto)

        scxL_grad_new = [load_state_batch_np[k] for k in range(self.N)]
        scxL_grad_new += [scxL_grad[self.N] + px_dis * (xl_grad[self.N] - scxl_grad[self.N]) + dscxL_updatedp[self.N]]
        scuL_grad_new = [load_ctrl_batch_np[k] for k in range(self.N)]
        scxC_grad_new = [cable_state_time_major[k] for k in range(self.N + 1)]
        scuC_grad_new = [cable_ctrl_time_major[k] for k in range(self.N)]

        grad_out3 = {
            "scxL_grad":scxL_grad_new,
            "scuL_grad":scuL_grad_new,
            "scxC_grad":scxC_grad_new,
            "scuC_grad":scuC_grad_new
        }

        return grad_out3
    




    def _stack_pytree_batch(self, pytree_list):
        return jax.tree_util.tree_map(
            lambda *xs: np.stack([np.asarray(x, dtype=np.float64) for x in xs], axis=0),
            *pytree_list
        )

    def _stack_float_batch(self, array_list):
        return np.stack([np.asarray(x, dtype=np.float64) for x in array_list], axis=0)

    def _pack_cable_optsol_batch(self, opt_solc_list):
        return {
            "Quu_inv": self._stack_float_batch([opt_sol["Quu_inv"] for opt_sol in opt_solc_list]),
            "Qxu": self._stack_float_batch([opt_sol["Qxu"] for opt_sol in opt_solc_list]),
            "K_FB": self._stack_float_batch([opt_sol["K_FB"] for opt_sol in opt_solc_list]),
            "Fx": self._stack_float_batch([opt_sol["Fx"] for opt_sol in opt_solc_list]),
            "Fu": self._stack_float_batch([opt_sol["Fu"] for opt_sol in opt_solc_list]),
            "Hxx": self._stack_float_batch([opt_sol["Hxx"] for opt_sol in opt_solc_list]),
            "Hxu": self._stack_float_batch([opt_sol["Hxu"] for opt_sol in opt_solc_list]),
            "Huu": self._stack_float_batch([opt_sol["Huu"] for opt_sol in opt_solc_list]),
        }

    def _pack_cable_auxsys_batch(self, auxsys_list):
        return {
            "HxxN": self._stack_float_batch([aux["HxxN"] for aux in auxsys_list]),
            "HxNp": self._stack_float_batch([aux["HxNp"] for aux in auxsys_list]),
            "Hxp": self._stack_float_batch([aux["Hxp"] for aux in auxsys_list]),
            "Hup": self._stack_float_batch([aux["Hup"] for aux in auxsys_list]),
        }

    def _jax_parallel_device_count(self, batch_size):
        return max(1, min(batch_size, jax.local_device_count()))

    def _jax_cable_device_count(self, batch_size):
        return max(1, min(batch_size, len(self.jax_cable_devices)))

    def _jax_subp1_agent_device_count(self, batch_size):
        batch_size = int(batch_size)
        if batch_size <= 0:
            return 1
        return max(1, min(len(self.jax_subp1_agent_devices), max(batch_size, len(self.jax_subp1_agent_devices))))

    def _jax_subp3_load_device_count(self, batch_size):
        return self._jax_subp3_device_count(batch_size)

    def _jax_subp3_cable_device_count(self, batch_size):
        return self._jax_subp3_device_count(batch_size)

    def _jax_subp3_device_count(self, batch_size):
        batch_size = int(batch_size)
        available_devices = len(self.jax_subp3_devices)
        if batch_size <= 0:
            return 1
        return max(1, min(self._subp3_max_devices, available_devices))

    def _jax_subp2_device_count(self, batch_size):
        batch_size = int(batch_size)
        available_devices = len(self.jax_subp2_devices)
        if batch_size <= 0:
            return 1
        return max(1, min(200, available_devices))

    def _subp3_has_disjoint_device_groups(self):
        return self._subp3_disjoint_device_groups

    def _shard_pytree_batch(self, batched_pytree, batch_size, axis_size=None, devices=None):
        axis_size = self._jax_parallel_device_count(batch_size) if axis_size is None else max(1, min(axis_size, batch_size))
        if devices is not None:
            devices = tuple(devices[:axis_size])
        per_device = math.ceil(batch_size / axis_size)
        padded_size = axis_size * per_device
        pad_amount = padded_size - batch_size

        def _pad_and_reshape(x):
            x = np.asarray(x)
            if pad_amount > 0:
                pad_width = [(0, pad_amount)] + [(0, 0)] * (x.ndim - 1)
                x = np.pad(x, pad_width)
            x = x.reshape((axis_size, per_device) + x.shape[1:])
            if devices is not None:
                return jax.device_put_sharded([x[i] for i in range(axis_size)], devices)
            return jax.device_put(x)

        return jax.tree_util.tree_map(_pad_and_reshape, batched_pytree), axis_size

    def _shard_pytree_batch_fixed_axis(self, batched_pytree, batch_size, axis_size, devices=None):
        axis_size = max(1, int(axis_size))
        if devices is not None:
            devices = tuple(devices[:axis_size])
        per_device = math.ceil(batch_size / axis_size)
        padded_size = axis_size * per_device
        pad_amount = padded_size - batch_size

        def _pad_and_reshape(x):
            x = np.asarray(x)
            if pad_amount > 0:
                pad_width = [(0, pad_amount)] + [(0, 0)] * (x.ndim - 1)
                x = np.pad(x, pad_width)
            x = x.reshape((axis_size, per_device) + x.shape[1:])
            if devices is not None:
                return jax.device_put_sharded([x[i] for i in range(axis_size)], devices)
            return jax.device_put(x)

        return jax.tree_util.tree_map(_pad_and_reshape, batched_pytree), axis_size

    def _unshard_pytree_batch(self, sharded_pytree, batch_size):
        flat_pytree = jax.tree_util.tree_map(
            lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]),
            sharded_pytree
        )
        return self._unstack_pytree_batch(flat_pytree, batch_size)

    def _unshard_pytree_array_batch(self, sharded_pytree, batch_size):
        return jax.tree_util.tree_map(
            lambda x: np.asarray(x, dtype=np.float64).reshape((x.shape[0] * x.shape[1],) + tuple(x.shape[2:]))[:batch_size],
            sharded_pytree
        )

    def _unshard_subp3_flat_outputs(self, sharded_outputs, state_batch_size, ctrl_batch_size):
        state_flat = np.asarray(sharded_outputs["state"], dtype=np.float64).reshape(
            (sharded_outputs["state"].shape[0] * sharded_outputs["state"].shape[1],)
            + tuple(sharded_outputs["state"].shape[2:])
        )[:state_batch_size]
        ctrl_flat = np.asarray(sharded_outputs["ctrl"], dtype=np.float64).reshape(
            (sharded_outputs["ctrl"].shape[0] * sharded_outputs["ctrl"].shape[1],)
            + tuple(sharded_outputs["ctrl"].shape[2:])
        )[:ctrl_batch_size]
        return state_flat, ctrl_flat

    def _shard_array_batch(self, batch_array, batch_size, axis_size=None, devices=None):
        axis_size = self._jax_parallel_device_count(batch_size) if axis_size is None else max(1, min(axis_size, batch_size))
        if devices is not None:
            devices = tuple(devices[:axis_size])
        per_device = math.ceil(batch_size / axis_size)
        padded_size = axis_size * per_device
        pad_amount = padded_size - batch_size
        batch_array = np.asarray(batch_array)
        if pad_amount > 0:
            pad_width = [(0, pad_amount)] + [(0, 0)] * (batch_array.ndim - 1)
            batch_array = np.pad(batch_array, pad_width)
        batch_array = batch_array.reshape((axis_size, per_device) + batch_array.shape[1:])
        if devices is not None:
            return jax.device_put_sharded([batch_array[i] for i in range(axis_size)], devices), axis_size
        return jax.device_put(batch_array), axis_size

    def _shard_array_batch_fixed_axis(self, batch_array, batch_size, axis_size, devices=None):
        axis_size = max(1, int(axis_size))
        if devices is not None:
            devices = tuple(devices[:axis_size])
        per_device = math.ceil(batch_size / axis_size)
        padded_size = axis_size * per_device
        pad_amount = padded_size - batch_size
        batch_array = np.asarray(batch_array)
        if pad_amount > 0:
            pad_width = [(0, pad_amount)] + [(0, 0)] * (batch_array.ndim - 1)
            batch_array = np.pad(batch_array, pad_width)
        batch_array = batch_array.reshape((axis_size, per_device) + batch_array.shape[1:])
        if devices is not None:
            return jax.device_put_sharded([batch_array[i] for i in range(axis_size)], devices), axis_size
        return jax.device_put(batch_array), axis_size

    def _shard_subp2_solve_batches_fixed_axis(self, L_hessian_batch, L_trajp_batch, reg_batch, I_hessian, axis_size, devices=None):
        axis_size = max(1, int(axis_size))
        if devices is not None:
            devices = tuple(devices[:axis_size])

        batch_size = int(L_hessian_batch.shape[0])
        per_device = math.ceil(batch_size / axis_size)
        padded_size = axis_size * per_device
        pad_amount = padded_size - batch_size

        L_hessian_batch = np.asarray(L_hessian_batch, dtype=np.float64)
        L_trajp_batch = np.asarray(L_trajp_batch, dtype=np.float64)
        reg_batch = np.asarray(reg_batch, dtype=np.float64)
        I_hessian = np.asarray(I_hessian, dtype=np.float64)

        if pad_amount > 0:
            dummy_hessian = np.repeat(I_hessian[None, :, :], pad_amount, axis=0)
            dummy_trajp = np.zeros((pad_amount,) + L_trajp_batch.shape[1:], dtype=np.float64)
            dummy_reg = np.zeros((pad_amount,), dtype=np.float64)
            L_hessian_batch = np.concatenate((L_hessian_batch, dummy_hessian), axis=0)
            L_trajp_batch = np.concatenate((L_trajp_batch, dummy_trajp), axis=0)
            reg_batch = np.concatenate((reg_batch, dummy_reg), axis=0)

        L_hessian_batch = L_hessian_batch.reshape((axis_size, per_device) + L_hessian_batch.shape[1:])
        L_trajp_batch = L_trajp_batch.reshape((axis_size, per_device) + L_trajp_batch.shape[1:])
        reg_batch = reg_batch.reshape((axis_size, per_device))

        if devices is not None:
            return (
                jax.device_put_sharded([L_hessian_batch[i] for i in range(axis_size)], devices),
                jax.device_put_sharded([L_trajp_batch[i] for i in range(axis_size)], devices),
                jax.device_put_sharded([reg_batch[i] for i in range(axis_size)], devices),
                axis_size,
            )

        return (
            jax.device_put(L_hessian_batch),
            jax.device_put(L_trajp_batch),
            jax.device_put(reg_batch),
            axis_size,
        )

    def _unshard_array_batch(self, sharded_array, batch_size):
        flat_array = np.asarray(sharded_array, dtype=np.float64).reshape((sharded_array.shape[0] * sharded_array.shape[1],) + tuple(sharded_array.shape[2:]))
        return flat_array[:batch_size]

    def _parallel_worker_count(self, task_count):
        if task_count <= 1:
            return 1
        return max(1, min(task_count, os.cpu_count() or 1))

    def _parallel_collect(self, executor, fn, args_list):
        futures = [executor.submit(fn, *args) for args in args_list]
        return [future.result() for future in futures]

    def _unstack_pytree_batch(self, batched_pytree, batch_size):
        batched_pytree_np = jax.tree_util.tree_map(
            lambda x: np.asarray(x, dtype=np.float64),
            batched_pytree
        )
        return [
            jax.tree_util.tree_map(
                lambda x, i=i: x[i],
                batched_pytree_np
            )
            for i in range(batch_size)
        ]

    def _to_jax_batch(self, array_list):
        return jax.device_put(np.stack([np.asarray(x, dtype=np.float64) for x in array_list], axis=0))

    def _block_tree_ready(self, pytree):
        return jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            pytree
        )

    def _subp3_update_vmappable(self, grad_curr, grad_primal, grad_subp2, dgrad_update, penalty):
        return grad_curr + penalty * (grad_primal - grad_subp2) + dgrad_update

    def _subp2_solve_single(self, L_hessian_k, L_trajp_k, reg, I_hessian):
        L_hessian_reg = np.asarray(L_hessian_k, dtype=np.float64) + float(reg) * np.asarray(I_hessian, dtype=np.float64)
        L, _jitter = self.try_cholesky(L_hessian_reg, jitter0=0.0)
        return self.chol_solve(L, -np.asarray(L_trajp_k, dtype=np.float64))

    def _subp2_solve_chunk(self, L_hessian_chunk, L_trajp_chunk, reg_chunk, I_hessian):
        return self._batched_subp2_solve(L_hessian_chunk, L_trajp_chunk, reg_chunk, I_hessian)

    def _subp2_time_worker_count(self, task_count):
        if task_count <= 1:
            return 1
        return max(1, min(task_count, self._subp2_max_workers))

    def _subp2_solve_shard_jax(self, L_hessian_shard, L_trajp_shard, reg_shard, I_hessian):
        L_hessian_shard = jnp.asarray(L_hessian_shard, dtype=jnp.float64)
        L_trajp_shard = jnp.asarray(L_trajp_shard, dtype=jnp.float64)
        reg_shard = jnp.asarray(reg_shard, dtype=jnp.float64)
        I_hessian = jnp.asarray(I_hessian, dtype=jnp.float64)

        def _solve_one(inputs):
            L_hessian_k, L_trajp_k, reg_k = inputs
            # Match the legacy LA path as closely as possible while staying
            # inside JAX tracing: float64 and the same unsymmetrized
            # Cholesky -> solve -> solve sequence.
            L_hessian_reg = L_hessian_k + reg_k * I_hessian
            L = jax.lax.linalg.cholesky(L_hessian_reg, symmetrize_input=False)
            Y = jnp.linalg.solve(L, -L_trajp_k)
            return jnp.linalg.solve(jnp.swapaxes(L, -1, -2), Y)

        return jax.lax.map(_solve_one, (L_hessian_shard, L_trajp_shard, reg_shard))

    def _get_sharded_subp2_solve_fn(self, axis_size):
        axis_size = self._jax_subp2_device_count(axis_size)
        if not hasattr(self, "_sharded_subp2_solve_fns"):
            self._sharded_subp2_solve_fns = {}
        if axis_size not in self._sharded_subp2_solve_fns:
            subp2_devices = self.jax_subp2_devices[:axis_size]
            self._sharded_subp2_solve_fns[axis_size] = pmap(
                lambda L_hessian_shard, L_trajp_shard, reg_shard, I_hessian:
                    self._subp2_solve_shard_jax(L_hessian_shard, L_trajp_shard, reg_shard, I_hessian),
                in_axes=(0, 0, 0, None),
                devices=subp2_devices
            )
        return self._sharded_subp2_solve_fns[axis_size]

    def _run_subp2_horizon_solves(self, prepared_batches):
        return self._get_sharded_subp2_solve_fn(prepared_batches["subp2_axis_size"])(
            prepared_batches["L_hessian_sharded"],
            prepared_batches["L_trajp_sharded"],
            prepared_batches["reg_sharded"],
            prepared_batches["I_hessian_jax"]
        )

    def _zero_ddp_cable_gradient_jax(self):
        return {
            "xi_grad": jnp.zeros((self.N + 1, self.nxi, self.n_Pauto), dtype=jnp.float64),
            "ui_grad": jnp.zeros((self.N, self.nui, self.n_Pauto), dtype=jnp.float64),
        }

    def _zero_p_cable_gradient_jax(self):
        return {
            "xi_grad": jnp.zeros((self.N + 1, self.nxi, self.n_Pauto), dtype=jnp.float64),
            "ui_grad": jnp.zeros((self.N, self.nui, self.n_Pauto), dtype=jnp.float64),
            "p_grad": jnp.zeros((self.N + 1, self.n_Pauto, self.n_Pauto), dtype=jnp.float64),
        }

    def _masked_ddp_cable_gradient_shard(self, opt_solc_shard, auxsysc_shard, scxi_grad_shard,
                                         scxI_grad_shard, scui_grad_shard, scuI_grad_shard,
                                         active_mask_shard, pix_dis, piu_dis):
        def _run_one(inputs):
            opt_solc, auxsysc, scxi_grad, scxI_grad, scui_grad, scuI_grad, active = inputs
            return jax.lax.cond(
                active,
                lambda data: self._ddp_cable_gradient_vmappable(
                    data[0], data[1], data[2], data[3], data[4], data[5], pix_dis, piu_dis
                ),
                lambda data: self._zero_ddp_cable_gradient_jax(),
                (opt_solc, auxsysc, scxi_grad, scxI_grad, scui_grad, scuI_grad)
            )
        return jax.lax.map(
            _run_one,
            (opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
             scui_grad_shard, scuI_grad_shard, active_mask_shard)
        )

    def _masked_caos_cable_gradient_shard(self, opt_solc_shard, auxsysc_shard, scxi_grad_shard,
                                          scxI_grad_shard, scui_grad_shard, scuI_grad_shard,
                                          active_mask_shard, pix_dis, piu_dis):
        def _run_one(inputs):
            opt_solc, auxsysc, scxi_grad, scxI_grad, scui_grad, scuI_grad, active = inputs
            return jax.lax.cond(
                active,
                lambda data: self._caos_cable_gradient_vmappable(
                    data[0], data[1], data[2], data[3], data[4], data[5], pix_dis, piu_dis
                ),
                lambda data: self._zero_p_cable_gradient_jax(),
                (opt_solc, auxsysc, scxi_grad, scxI_grad, scui_grad, scuI_grad)
            )
        return jax.lax.map(
            _run_one,
            (opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
             scui_grad_shard, scuI_grad_shard, active_mask_shard)
        )

    def _masked_cao_cable_gradient_shard(self, opt_solc_shard, auxsysc_shard, scxi_grad_shard,
                                         scxI_grad_shard, scui_grad_shard, scuI_grad_shard,
                                         active_mask_shard, pix_dis, piu_dis):
        def _run_one(inputs):
            opt_solc, auxsysc, scxi_grad, scxI_grad, scui_grad, scuI_grad, active = inputs
            return jax.lax.cond(
                active,
                lambda data: self._cao_cable_gradient_vmappable(
                    data[0], data[1], data[2], data[3], data[4], data[5], pix_dis, piu_dis
                ),
                lambda data: self._zero_p_cable_gradient_jax(),
                (opt_solc, auxsysc, scxi_grad, scxI_grad, scui_grad, scuI_grad)
            )
        return jax.lax.map(
            _run_one,
            (opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
             scui_grad_shard, scuI_grad_shard, active_mask_shard)
        )

    def _masked_pdp_cable_gradient_shard(self, opt_solc_shard, auxsysc_shard, scxi_grad_shard,
                                         scxI_grad_shard, scui_grad_shard, scuI_grad_shard,
                                         active_mask_shard, pix_dis, piu_dis):
        def _run_one(inputs):
            opt_solc, auxsysc, scxi_grad, scxI_grad, scui_grad, scuI_grad, active = inputs
            return jax.lax.cond(
                active,
                lambda data: self._pdp_cable_gradient_vmappable(
                    data[0], data[1], data[2], data[3], data[4], data[5], pix_dis, piu_dis
                ),
                lambda data: self._zero_ddp_cable_gradient_jax(),
                (opt_solc, auxsysc, scxi_grad, scxI_grad, scui_grad, scuI_grad)
            )
        return jax.lax.map(
            _run_one,
            (opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
             scui_grad_shard, scuI_grad_shard, active_mask_shard)
        )

    def _get_sharded_ddp_cable_gradient_fn(self, axis_size):
        axis_size = self._jax_subp1_agent_device_count(axis_size)
        if not hasattr(self, "_sharded_ddp_cable_gradient_fns"):
            self._sharded_ddp_cable_gradient_fns = {}
        if axis_size not in self._sharded_ddp_cable_gradient_fns:
            self._sharded_ddp_cable_gradient_fns[axis_size] = pmap(
                lambda opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
                       scui_grad_shard, scuI_grad_shard, active_mask_shard, pix_dis, piu_dis:
                    self._masked_ddp_cable_gradient_shard(
                        opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
                        scui_grad_shard, scuI_grad_shard, active_mask_shard, pix_dis, piu_dis
                    ),
                in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
                devices=self.jax_subp1_agent_devices[:axis_size]
            )
        return self._sharded_ddp_cable_gradient_fns[axis_size]

    def _get_sharded_caos_cable_gradient_fn(self, axis_size):
        axis_size = self._jax_subp1_agent_device_count(axis_size)
        if not hasattr(self, "_sharded_caos_cable_gradient_fns"):
            self._sharded_caos_cable_gradient_fns = {}
        if axis_size not in self._sharded_caos_cable_gradient_fns:
            self._sharded_caos_cable_gradient_fns[axis_size] = pmap(
                lambda opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
                       scui_grad_shard, scuI_grad_shard, active_mask_shard, pix_dis, piu_dis:
                    self._masked_caos_cable_gradient_shard(
                        opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
                        scui_grad_shard, scuI_grad_shard, active_mask_shard, pix_dis, piu_dis
                    ),
                in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
                devices=self.jax_subp1_agent_devices[:axis_size]
            )
        return self._sharded_caos_cable_gradient_fns[axis_size]

    def _get_sharded_cao_cable_gradient_fn(self, axis_size):
        axis_size = self._jax_subp1_agent_device_count(axis_size)
        if not hasattr(self, "_sharded_cao_cable_gradient_fns"):
            self._sharded_cao_cable_gradient_fns = {}
        if axis_size not in self._sharded_cao_cable_gradient_fns:
            self._sharded_cao_cable_gradient_fns[axis_size] = pmap(
                lambda opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
                       scui_grad_shard, scuI_grad_shard, active_mask_shard, pix_dis, piu_dis:
                    self._masked_cao_cable_gradient_shard(
                        opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
                        scui_grad_shard, scuI_grad_shard, active_mask_shard, pix_dis, piu_dis
                    ),
                in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
                devices=self.jax_subp1_agent_devices[:axis_size]
            )
        return self._sharded_cao_cable_gradient_fns[axis_size]

    def _get_sharded_pdp_cable_gradient_fn(self, axis_size):
        axis_size = self._jax_subp1_agent_device_count(axis_size)
        if not hasattr(self, "_sharded_pdp_cable_gradient_fns"):
            self._sharded_pdp_cable_gradient_fns = {}
        if axis_size not in self._sharded_pdp_cable_gradient_fns:
            self._sharded_pdp_cable_gradient_fns[axis_size] = pmap(
                lambda opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
                       scui_grad_shard, scuI_grad_shard, active_mask_shard, pix_dis, piu_dis:
                    self._masked_pdp_cable_gradient_shard(
                        opt_solc_shard, auxsysc_shard, scxi_grad_shard, scxI_grad_shard,
                        scui_grad_shard, scuI_grad_shard, active_mask_shard, pix_dis, piu_dis
                    ),
                in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
                devices=self.jax_subp1_agent_devices[:axis_size]
            )
        return self._sharded_pdp_cable_gradient_fns[axis_size]

    def _get_sharded_subp3_cable_update_fn(self):
        if not hasattr(self, "_sharded_subp3_cable_update_fn"):
            self._sharded_subp3_cable_update_fn = pmap(
                lambda grad_curr_shard, grad_primal_shard, grad_subp2_shard, dgrad_update_shard, penalty:
                    vmap(
                        lambda grad_curr, grad_primal, grad_subp2, dgrad_update:
                            self._subp3_update_vmappable(grad_curr, grad_primal, grad_subp2, dgrad_update, penalty),
                        in_axes=(0, 0, 0, 0)
                    )(grad_curr_shard, grad_primal_shard, grad_subp2_shard, dgrad_update_shard),
                in_axes=(0, 0, 0, 0, None)
            )
        return self._sharded_subp3_cable_update_fn

    def _get_sharded_subp3_cable_updates_fn(self, axis_size):
        axis_size = int(axis_size)
        if not hasattr(self, "_sharded_subp3_cable_updates_fns"):
            self._sharded_subp3_cable_updates_fns = {}
        if axis_size not in self._sharded_subp3_cable_updates_fns:
            self._sharded_subp3_cable_updates_fns[axis_size] = pmap(
                lambda state_pack_shard, ctrl_pack_shard, pix_dis, piu_dis: {
                    "state": state_pack_shard[:, 0] + pix_dis * (state_pack_shard[:, 1] - state_pack_shard[:, 2]) + state_pack_shard[:, 3],
                    "ctrl": ctrl_pack_shard[:, 0] + piu_dis * (ctrl_pack_shard[:, 1] - ctrl_pack_shard[:, 2]) + ctrl_pack_shard[:, 3],
                },
                in_axes=(0, 0, None, None),
                devices=self.jax_subp3_cable_devices[:axis_size]
            )
        return self._sharded_subp3_cable_updates_fns[axis_size]

    def _get_sharded_subp3_load_updates_fn(self, axis_size):
        axis_size = int(axis_size)
        if not hasattr(self, "_sharded_subp3_load_updates_fns"):
            self._sharded_subp3_load_updates_fns = {}
        if axis_size not in self._sharded_subp3_load_updates_fns:
            self._sharded_subp3_load_updates_fns[axis_size] = pmap(
                lambda state_pack_shard, ctrl_pack_shard, px_dis, pu_dis: {
                    "state": state_pack_shard[:, 0] + px_dis * (state_pack_shard[:, 1] - state_pack_shard[:, 2]) + state_pack_shard[:, 3],
                    "ctrl": ctrl_pack_shard[:, 0] + pu_dis * (ctrl_pack_shard[:, 1] - ctrl_pack_shard[:, 2]) + ctrl_pack_shard[:, 3],
                },
                in_axes=(0, 0, None, None),
                devices=self.jax_subp3_load_devices[:axis_size]
            )
        return self._sharded_subp3_load_updates_fns[axis_size]

    def _get_sharded_subp3_updates_fn(self, axis_size):
        axis_size = int(axis_size)
        if not hasattr(self, "_sharded_subp3_updates_fns"):
            self._sharded_subp3_updates_fns = {}
        if axis_size not in self._sharded_subp3_updates_fns:
            self._sharded_subp3_updates_fns[axis_size] = pmap(
                lambda load_state_pack_shard, load_ctrl_pack_shard,
                       cable_state_pack_shard, cable_ctrl_pack_shard,
                       px_dis, pu_dis, pix_dis, piu_dis: {
                    "load_state": load_state_pack_shard[:, 0] + px_dis * (load_state_pack_shard[:, 1] - load_state_pack_shard[:, 2]) + load_state_pack_shard[:, 3],
                    "load_ctrl": load_ctrl_pack_shard[:, 0] + pu_dis * (load_ctrl_pack_shard[:, 1] - load_ctrl_pack_shard[:, 2]) + load_ctrl_pack_shard[:, 3],
                    "cable_state": cable_state_pack_shard[:, 0] + pix_dis * (cable_state_pack_shard[:, 1] - cable_state_pack_shard[:, 2]) + cable_state_pack_shard[:, 3],
                    "cable_ctrl": cable_ctrl_pack_shard[:, 0] + piu_dis * (cable_ctrl_pack_shard[:, 1] - cable_ctrl_pack_shard[:, 2]) + cable_ctrl_pack_shard[:, 3],
                },
                in_axes=(0, 0, 0, 0, None, None, None, None),
                devices=self.jax_subp3_devices[:axis_size]
            )
        return self._sharded_subp3_updates_fns[axis_size]

    @partial(jax.jit, static_argnums=0)
    def _batched_subp3_load_updates(self, state_curr, state_primal, state_subp2, state_update,
                                    ctrl_curr, ctrl_primal, ctrl_subp2, ctrl_update,
                                    px_dis, pu_dis):
        return {
            "state": state_curr + px_dis * (state_primal - state_subp2) + state_update,
            "ctrl": ctrl_curr + pu_dis * (ctrl_primal - ctrl_subp2) + ctrl_update,
        }

    @partial(jax.jit, static_argnums=0)
    def _batched_subp3_cable_updates(self, state_curr, state_primal, state_subp2, state_update,
                                     ctrl_curr, ctrl_primal, ctrl_subp2, ctrl_update,
                                     pix_dis, piu_dis):
        return {
            "state": state_curr + pix_dis * (state_primal - state_subp2) + state_update,
            "ctrl": ctrl_curr + piu_dis * (ctrl_primal - ctrl_subp2) + ctrl_update,
        }

    @partial(jax.jit, static_argnums=0)
    def _batched_subp3_time_major_updates(self, load_state_pack, load_ctrl_pack,
                                          cable_state_pack, cable_ctrl_pack,
                                          px_dis, pu_dis, pix_dis, piu_dis):
        return {
            "load_state": load_state_pack[:, 0] + px_dis * (load_state_pack[:, 1] - load_state_pack[:, 2]) + load_state_pack[:, 3],
            "load_ctrl": load_ctrl_pack[:, 0] + pu_dis * (load_ctrl_pack[:, 1] - load_ctrl_pack[:, 2]) + load_ctrl_pack[:, 3],
            "cable_state": cable_state_pack[:, 0] + pix_dis * (cable_state_pack[:, 1] - cable_state_pack[:, 2]) + cable_state_pack[:, 3],
            "cable_ctrl": cable_ctrl_pack[:, 0] + piu_dis * (cable_ctrl_pack[:, 1] - cable_ctrl_pack[:, 2]) + cable_ctrl_pack[:, 3],
        }

    def _run_subp3_load_updates(self, prepared_batches, px_dis, pu_dis):
        return self._get_sharded_subp3_load_updates_fn(prepared_batches["load_axis_size"])(
            prepared_batches["load_state_pack_sharded"],
            prepared_batches["load_ctrl_pack_sharded"],
            px_dis,
            pu_dis
        )

    def _run_subp3_cable_updates(self, prepared_batches, pix_dis, piu_dis):
        return self._get_sharded_subp3_cable_updates_fn(prepared_batches["cable_axis_size"])(
            prepared_batches["cable_state_pack_sharded"],
            prepared_batches["cable_ctrl_pack_sharded"],
            pix_dis,
            piu_dis
        )

    def _run_subp3_updates(self, prepared_batches, px_dis, pu_dis, pix_dis, piu_dis):
        return self._get_sharded_subp3_updates_fn(prepared_batches["subp3_axis_size"])(
            prepared_batches["load_state_pack_sharded"],
            prepared_batches["load_ctrl_pack_sharded"],
            prepared_batches["cable_state_pack_sharded"],
            prepared_batches["cable_ctrl_pack_sharded"],
            px_dis,
            pu_dis,
            pix_dis,
            piu_dis
        )

    def _prepare_subp3_batches(self, grad_outl, grad_outc, grad_out2, scxL_grad, scuL_grad, scxC_grad, scuC_grad, auxSys3):
        nq = int(self.nq)
        xl_grad = grad_outl['xl_grad']
        ul_grad = grad_outl['ul_grad']
        scxl_grad = grad_out2['scxl_grad']
        scul_grad = grad_out2['scul_grad']
        scxc_grad = grad_out2['scxc_grad']
        scuc_grad = grad_out2['scuc_grad']
        dscxL_updatedp = auxSys3['dscxL_updatedp']
        dscuL_updatedp = auxSys3['dscuL_updatedp']
        dscxC_updatedp = auxSys3['dscxC_updatedp']
        dscuC_updatedp = auxSys3['dscuC_updatedp']
        subp3_batch_size = max(self.N, nq * (self.N + 1), nq * self.N)
        subp3_axis_size = self._jax_subp3_device_count(subp3_batch_size)
        subp3_devices = self.jax_subp3_devices[:subp3_axis_size]
        def _build_load_batches():
            load_state_pack_batch = np.empty((self.N, 4, self.nxl, self.n_Pauto), dtype=np.float64)
            load_state_pack_batch[:, 0] = np.stack([np.asarray(x, dtype=np.float64) for x in scxL_grad[:self.N]], axis=0)
            load_state_pack_batch[:, 1] = np.stack([np.asarray(x, dtype=np.float64) for x in xl_grad[:self.N]], axis=0)
            load_state_pack_batch[:, 2] = np.stack([np.asarray(x, dtype=np.float64) for x in scxl_grad[:self.N]], axis=0)
            load_state_pack_batch[:, 3] = np.stack([np.asarray(x, dtype=np.float64) for x in dscxL_updatedp[:self.N]], axis=0)
            load_ctrl_pack_batch = np.empty((self.N, 4, self.nul, self.n_Pauto), dtype=np.float64)
            load_ctrl_pack_batch[:, 0] = np.stack([np.asarray(x, dtype=np.float64) for x in scuL_grad], axis=0)
            load_ctrl_pack_batch[:, 1] = np.stack([np.asarray(x, dtype=np.float64) for x in ul_grad], axis=0)
            load_ctrl_pack_batch[:, 2] = np.stack([np.asarray(x, dtype=np.float64) for x in scul_grad], axis=0)
            load_ctrl_pack_batch[:, 3] = np.stack([np.asarray(x, dtype=np.float64) for x in dscuL_updatedp], axis=0)
            load_axis_size = subp3_axis_size
            load_state_pack_sharded, _ = self._shard_array_batch_fixed_axis(
                load_state_pack_batch, self.N, load_axis_size, devices=subp3_devices
            )
            load_ctrl_pack_sharded, _ = self._shard_array_batch_fixed_axis(
                load_ctrl_pack_batch, self.N, load_axis_size, devices=subp3_devices
            )
            return {
                "load_axis_size": load_axis_size,
                "load_state_pack_sharded": load_state_pack_sharded,
                "load_ctrl_pack_sharded": load_ctrl_pack_sharded,
            }

        def _build_cable_batches():
            cable_state_pack_time_major = np.empty((self.N + 1, 4, nq, self.nxi, self.n_Pauto), dtype=np.float64)
            cable_state_pack_time_major[:, 0] = np.stack(
                [np.asarray(x, dtype=np.float64) for x in scxC_grad],
                axis=0
            ).reshape((self.N + 1), nq, self.nxi, self.n_Pauto)
            cable_state_pack_time_major[:, 1] = np.transpose(
                np.stack([np.asarray(grad_outc[i]['xi_grad'], dtype=np.float64) for i in range(nq)], axis=0),
                (1, 0, 2, 3)
            )
            cable_state_pack_time_major[:, 2] = np.stack(
                [np.asarray(x, dtype=np.float64) for x in scxc_grad],
                axis=0
            ).reshape((self.N + 1), nq, self.nxi, self.n_Pauto)
            cable_state_pack_time_major[:, 3] = np.stack(
                [np.asarray(x, dtype=np.float64) for x in dscxC_updatedp],
                axis=0
            ).reshape((self.N + 1), nq, self.nxi, self.n_Pauto)
            cable_ctrl_pack_time_major = np.empty((self.N, 4, nq, self.nui, self.n_Pauto), dtype=np.float64)
            cable_ctrl_pack_time_major[:, 0] = np.stack(
                [np.asarray(scuC_grad[k], dtype=np.float64).reshape(nq, self.nui, self.n_Pauto) for k in range(self.N)],
                axis=0
            )
            cable_ctrl_pack_time_major[:, 1] = np.transpose(
                np.stack([np.asarray(grad_outc[i]['ui_grad'], dtype=np.float64) for i in range(nq)], axis=0),
                (1, 0, 2, 3)
            )
            cable_ctrl_pack_time_major[:, 2] = np.stack(
                [np.asarray(scuc_grad[k], dtype=np.float64).reshape(nq, self.nui, self.n_Pauto) for k in range(self.N)],
                axis=0
            )
            cable_ctrl_pack_time_major[:, 3] = np.stack(
                [np.asarray(dscuC_updatedp[k], dtype=np.float64).reshape(nq, self.nui, self.n_Pauto) for k in range(self.N)],
                axis=0
            )
            cable_state_pack_flat = np.transpose(
                cable_state_pack_time_major, (0, 2, 1, 3, 4)
            ).reshape(
                (self.N + 1) * nq, 4, self.nxi, self.n_Pauto
            )
            cable_ctrl_pack_flat = np.transpose(
                cable_ctrl_pack_time_major, (0, 2, 1, 3, 4)
            ).reshape(
                self.N * nq, 4, self.nui, self.n_Pauto
            )
            cable_axis_size = subp3_axis_size
            cable_state_pack_sharded, _ = self._shard_array_batch_fixed_axis(
                cable_state_pack_flat, nq * (self.N + 1), cable_axis_size, devices=subp3_devices
            )
            cable_ctrl_pack_sharded, _ = self._shard_array_batch_fixed_axis(
                cable_ctrl_pack_flat, nq * self.N, cable_axis_size, devices=subp3_devices
            )
            return {
                "cable_axis_size": cable_axis_size,
                "cable_time_major": False,
                "cable_state_pack_sharded": cable_state_pack_sharded,
                "cable_ctrl_pack_sharded": cable_ctrl_pack_sharded,
            }

        if self._subp3_executor is not None:
            load_future = self._subp3_executor.submit(_build_load_batches)
            cable_future = self._subp3_executor.submit(_build_cable_batches)
            load_batches = load_future.result()
            cable_batches = cable_future.result()
        else:
            load_batches = _build_load_batches()
            cable_batches = _build_cable_batches()

        return {
            "subp3_combined_pmap": True,
            "subp3_axis_size": subp3_axis_size,
            **load_batches,
            **cable_batches,
        }

    def _ddp_cable_gradient_vmappable(self, opt_sol, auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix_dis, piu_dis):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        S           = [jnp.zeros((self.nxi,self.n_Pauto)) for _ in range(self.N+1)]
        S[self.N]   = HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]
        v_FF        = [jnp.zeros((self.nui,self.n_Pauto)) for _ in range(self.N)]
        xi_grad     = [jnp.zeros((self.nxi,self.n_Pauto)) for _ in range(self.N+1)]
        ui_grad     = [jnp.zeros((self.nui,self.n_Pauto)) for _ in range(self.N)]
        for k in reversed(range(self.N)):
            Hxp_k    = Hxp[k] + scxI_grad[k] - pix_dis*scxi_grad[k]
            Hup_k    = Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            S[k]     = Hxp_k + F[k].T@S[k+1] + K_fb[k].T@(Hup_k + G[k].T@S[k+1])
        for k in range(self.N):
            ui_grad[k]  = K_fb[k]@xi_grad[k]+v_FF[k]
            xi_grad[k+1]= F[k]@xi_grad[k]+G[k]@ui_grad[k]
        return {
            "xi_grad": jnp.stack(xi_grad, axis=0),
            "ui_grad": jnp.stack(ui_grad, axis=0),
        }

    def _caos_cable_gradient_vmappable(self, opt_sol, auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix_dis, piu_dis):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        S           = [jnp.zeros((self.nxi,self.n_Pauto)) for _ in range(self.N+1)]
        Vpp         = [jnp.zeros((self.n_Pauto,self.n_Pauto)) for _ in range(self.N+1)]
        S[self.N]   = HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]
        Vpp[self.N] = jnp.zeros((self.n_Pauto,self.n_Pauto))
        v_FF        = [jnp.zeros((self.nui,self.n_Pauto)) for _ in range(self.N)]
        xi_grad     = [jnp.zeros((self.nxi,self.n_Pauto)) for _ in range(self.N+1)]
        p_grad      = [jnp.identity(self.n_Pauto) for _ in range(self.N+1)]
        ui_grad     = [jnp.zeros((self.nui,self.n_Pauto)) for _ in range(self.N)]
        for k in reversed(range(self.N)):
            Hpp_k    = jnp.zeros((self.n_Pauto,self.n_Pauto))
            Hxp_k    = Hxp[k] + scxI_grad[k] - pix_dis*scxi_grad[k]
            Hup_k    = Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            Vpp[k]   = Hpp_k + Vpp[k+1] + (Hup_k + G[k].T@S[k+1]).T@v_FF[k]
            S[k]     = Hxp_k + F[k].T@S[k+1] + K_fb[k].T@(Hup_k + G[k].T@S[k+1])
        for k in range(self.N):
            ui_grad[k]  = K_fb[k]@xi_grad[k]+v_FF[k]@p_grad[k]
            xi_grad[k+1]= F[k]@xi_grad[k]+G[k]@ui_grad[k]
            p_grad[k+1] = p_grad[k]
        return {
            "xi_grad": jnp.stack(xi_grad, axis=0),
            "ui_grad": jnp.stack(ui_grad, axis=0),
            "p_grad": jnp.stack(p_grad, axis=0),
        }

    def _cao_cable_gradient_vmappable(self, opt_sol, auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix_dis, piu_dis):
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysi['HxxN'], auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        Vpp         = [jnp.zeros((self.n_Pauto,self.n_Pauto)) for _ in range(self.N+1)]
        Vpx         = [jnp.zeros((self.n_Pauto,self.nxi)) for _ in range(self.N+1)]
        Vxp         = [jnp.zeros((self.nxi,self.n_Pauto)) for _ in range(self.N+1)]
        Vxx         = [jnp.zeros((self.nxi,self.nxi)) for _ in range(self.N+1)]
        Kfb_p       = [jnp.zeros((self.nui,self.n_Pauto)) for _ in range(self.N)]
        Kfb_x       = [jnp.zeros((self.nui,self.nxi)) for _ in range(self.N)]
        p_grad      = [jnp.identity(self.n_Pauto) for _ in range(self.N+1)]
        xi_grad     = [jnp.zeros((self.nxi,self.n_Pauto)) for _ in range(self.N+1)]
        ui_grad     = [jnp.zeros((self.nui,self.n_Pauto)) for _ in range(self.N)]
        Vpp[self.N] = jnp.zeros((self.n_Pauto,self.n_Pauto))
        Vpx[self.N] = (HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]).T
        Vxp[self.N] = HxNp + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]
        Vxx[self.N] = HxxN
        for k in reversed(range(self.N)):
            Hpx_k    = (Hxp[k]+ scxI_grad[k] - pix_dis*scxi_grad[k]).T
            Hxx_k    = Hxx[k]
            Hup_k    = Hup[k]+ scuI_grad[k] - piu_dis*scui_grad[k]
            Quu_k    = Huu[k]+G[k].T@Vxx[k+1]@G[k]
            invQuu_k = jnp.linalg.inv(Quu_k)
            Kfb_p[k] = -invQuu_k@(Hup_k+G[k].T@Vxp[k+1])
            Kfb_x[k] = -invQuu_k@(Hxu[k].T+G[k].T@Vxx[k+1]@F[k])
            Vpp[k]   = Vpp[k+1] + (Hup_k.T+Vpx[k+1]@G[k])@Kfb_p[k]
            Vpx[k]   = Hpx_k + Vpx[k+1]@F[k] + Kfb_p[k].T@(Hxu[k]+F[k].T@Vxx[k+1]@G[k]).T
            Vxp[k]   = Vpx[k].T
            Vxx[k]   = Hxx_k + F[k].T@Vxx[k+1]@F[k] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_x[k]
        for k in range(self.N):
            ui_grad[k]   = Kfb_p[k]@p_grad[k] + Kfb_x[k]@xi_grad[k]
            xi_grad[k+1] = F[k]@xi_grad[k]+G[k]@ui_grad[k]
            p_grad[k+1]  = p_grad[k]
        return {
            "xi_grad": jnp.stack(xi_grad, axis=0),
            "ui_grad": jnp.stack(ui_grad, axis=0),
            "p_grad": jnp.stack(p_grad, axis=0),
        }

    def _pdp_cable_gradient_vmappable(self, opt_sol, auxSysi, scxi_grad, scxI_grad, scui_grad, scuI_grad, pix_dis, piu_dis):
        Hxx, Hxu, Huu, F, G  = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSysi['HxxN'], auxSysi['HxNp'], auxSysi['Hxp'], auxSysi['Hup']
        P           = [jnp.zeros((self.nxi,self.nxi)) for _ in range(self.N+1)]
        S           = [jnp.zeros((self.nxi,self.n_Pauto)) for _ in range(self.N+1)]
        A           = [jnp.zeros((self.nxi,self.nxi)) for _ in range(self.N)]
        R           = [jnp.zeros((self.nxi,self.nxi)) for _ in range(self.N)]
        M_p         = [jnp.zeros((self.nxi,self.n_Pauto)) for _ in range(self.N)]
        invHuu      = [jnp.zeros((self.nui,self.nui)) for _ in range(self.N)]
        PinvIRP     = [jnp.zeros((self.nxi,self.nxi)) for _ in range(self.N)]
        P[self.N]   = HxxN
        S[self.N]   = HxNp  + scxI_grad[self.N] - pix_dis*scxi_grad[self.N]
        xi_grad     = [jnp.zeros((self.nxi,self.n_Pauto)) for _ in range(self.N+1)]
        ui_grad     = [jnp.zeros((self.nui,self.n_Pauto)) for _ in range(self.N)]
        I           = jnp.identity(self.nxi)
        for k in reversed(range(self.N)):
            P_next      = P[k+1]
            S_next      = S[k+1]
            invHuu[k]   = jnp.linalg.inv(Huu[k])
            GinvHuu     = G[k]@invHuu[k]
            HxuinvHuu   = Hxu[k]@invHuu[k]
            A[k]        = F[k]-GinvHuu@Hxu[k].T
            R[k]        = GinvHuu@G[k].T
            M_p[k]      = -GinvHuu@(Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k])
            Q_k         = Hxx[k]-HxuinvHuu@Hxu[k].T
            N_p_k       = Hxp[k]+ scxI_grad[k] - pix_dis*scxi_grad[k] - HxuinvHuu@(Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k])
            PinvIRP[k]  = P_next@jnp.linalg.inv(I+R[k]@P_next)
            P_curr      = Q_k + A[k].T@PinvIRP[k]@A[k]
            S_curr      = A[k].T@PinvIRP[k]@(M_p[k] - R[k]@S_next) + A[k].T@S_next + N_p_k
            P[k]        = P_curr
            S[k]        = S_curr
        for k in range(self.N):
            ui_grad[k]  = -invHuu[k]@((Hxu[k].T+G[k].T@PinvIRP[k]@A[k])@xi_grad[k] + G[k].T@PinvIRP[k]@(M_p[k]- R[k]@ S[k+1]) + G[k].T@S[k+1] + (Hup[k] + scuI_grad[k] - piu_dis*scui_grad[k]))
            xi_grad[k+1] = F[k]@xi_grad[k] + G[k]@ui_grad[k]
        return {
            "xi_grad": jnp.stack(xi_grad, axis=0),
            "ui_grad": jnp.stack(ui_grad, axis=0),
        }

    @partial(jax.jit, static_argnums=0)
    def _batched_ddp_cable_gradient(self, opt_solc_batch, auxsysc_batch, scxi_grad_batch, scxI_grad_batch, scui_grad_batch, scuI_grad_batch, pix_dis, piu_dis):
        return vmap(
            lambda opt_solc, auxsysi, scxi_grad, scxI_grad, scui_grad, scuI_grad:
                self._ddp_cable_gradient_vmappable(
                    opt_solc, auxsysi, scxi_grad, scxI_grad, scui_grad, scuI_grad,
                    pix_dis, piu_dis
                ),
            in_axes=(0, 0, 0, 0, 0, 0)
        )(opt_solc_batch, auxsysc_batch, scxi_grad_batch, scxI_grad_batch, scui_grad_batch, scuI_grad_batch)

    @partial(jax.jit, static_argnums=0)
    def _batched_caos_cable_gradient(self, opt_solc_batch, auxsysc_batch, scxi_grad_batch, scxI_grad_batch, scui_grad_batch, scuI_grad_batch, pix_dis, piu_dis):
        return vmap(
            lambda opt_solc, auxsysi, scxi_grad, scxI_grad, scui_grad, scuI_grad:
                self._caos_cable_gradient_vmappable(
                    opt_solc, auxsysi, scxi_grad, scxI_grad, scui_grad, scuI_grad,
                    pix_dis, piu_dis
                ),
            in_axes=(0, 0, 0, 0, 0, 0)
        )(opt_solc_batch, auxsysc_batch, scxi_grad_batch, scxI_grad_batch, scui_grad_batch, scuI_grad_batch)

    @partial(jax.jit, static_argnums=0)
    def _batched_cao_cable_gradient(self, opt_solc_batch, auxsysc_batch, scxi_grad_batch, scxI_grad_batch, scui_grad_batch, scuI_grad_batch, pix_dis, piu_dis):
        return vmap(
            lambda opt_solc, auxsysi, scxi_grad, scxI_grad, scui_grad, scuI_grad:
                self._cao_cable_gradient_vmappable(
                    opt_solc, auxsysi, scxi_grad, scxI_grad, scui_grad, scuI_grad,
                    pix_dis, piu_dis
                ),
            in_axes=(0, 0, 0, 0, 0, 0)
        )(opt_solc_batch, auxsysc_batch, scxi_grad_batch, scxI_grad_batch, scui_grad_batch, scuI_grad_batch)

    @partial(jax.jit, static_argnums=0)
    def _batched_pdp_cable_gradient(self, opt_solc_batch, auxsysc_batch, scxi_grad_batch, scxI_grad_batch, scui_grad_batch, scuI_grad_batch, pix_dis, piu_dis):
        return vmap(
            lambda opt_solc, auxsysi, scxi_grad, scxI_grad, scui_grad, scuI_grad:
                self._pdp_cable_gradient_vmappable(
                    opt_solc, auxsysi, scxi_grad, scxI_grad, scui_grad, scuI_grad,
                    pix_dis, piu_dis
                ),
            in_axes=(0, 0, 0, 0, 0, 0)
        )(opt_solc_batch, auxsysc_batch, scxi_grad_batch, scxI_grad_batch, scui_grad_batch, scuI_grad_batch)

    def _batched_subp2_solve(self, L_hessian_batch, L_trajp_batch, reg_batch, I_hessian):
        L_hessian_reg_batch = np.asarray(L_hessian_batch, dtype=np.float64) + np.asarray(reg_batch, dtype=np.float64)[:, None, None] * np.asarray(I_hessian, dtype=np.float64)[None, :, :]
        L_batch = LA.cholesky(L_hessian_reg_batch)
        Y_batch = LA.solve(L_batch, -np.asarray(L_trajp_batch, dtype=np.float64))
        return LA.solve(np.swapaxes(L_batch, -1, -2), Y_batch)

    def ADMM_Gradient_Solver(self,Opt_Sol1_l, Opt_Sol1_cddp, Opt_Sol1_c, Opt_Sol2, Opt_Sol3, Ref_xl, Ref_ul, ref_xq, ref_uq, weight1, weight2):
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
        gMeanerror_l   = [] # error between two load gradient trajecotries at two successive ADMM iterations
        gMeanerror_c   = [] # error between two cable gradient trajecotries at two successive ADMM iterations
        backward_time  = 0
        Gradtime_1     = 0
        Gradtime_2     = 0
        Gradtime_3     = 0
        for i_admm in range(self.max_iter_ADMM):
            ADMM_time      = 0
            # gradients of Subproblem1
            opt_sol        = Opt_Sol1_l[i_admm]
            auxSysl        = self.Get_AuxSys_DDP_Load(opt_sol,Ref_xl,Ref_ul,scxl,scul,scxL,scuL,weight1,i_admm)
            start_time     = TM.time()
            grad_outl      = self.DDP_Load_Gradient(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-4],weight1[-3],weight1[-2],weight1[-1],int(self.max_iter_ADMM),i_admm)
            gradtimeOur    = (TM.time() - start_time)*1000
            Gradtime_1 += gradtimeOur
            start_time     = TM.time()
            grad_outl_Caos = self.Cao_Load_Gradient_s(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-4],weight1[-3],weight1[-2],weight1[-1],int(self.max_iter_ADMM), i_admm)
            gradtimeCaos   = (TM.time() - start_time)*1000
            start_time     = TM.time()
            grad_outl_Cao  = self.Cao_Load_Gradient(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-4],weight1[-3],weight1[-2],weight1[-1],int(self.max_iter_ADMM),i_admm)
            gradtimeCao    = (TM.time() - start_time)*1000
            start_time     = TM.time()
            grad_outl_PDP  = self.PDP_Load_Gradient(opt_sol,auxSysl, scxl_grad, scxL_grad, scul_grad, scuL_grad, weight1[-4],weight1[-3],weight1[-2],weight1[-1],int(self.max_iter_ADMM), i_admm)
            gradtimePDP    = (TM.time() - start_time)*1000
            grad_outc         = []
            grad_outcCao      = []
            grad_outcPDP      = []
            gradtimeOur_sum   = 0
            gradtimeCaos_sum  = 0
            gradtimeCao_sum   = 0
            gradtimePDP_sum   = 0
            AuxSysc       = []

            # Parallelize cable gradient computation with vmap
            nq = int(self.nq)
            opt_solc_list = [Opt_Sol1_cddp[i_admm][i] for i in range(nq)]
            scxi_grad_list = []
            scxI_grad_list = []
            scui_grad_list = []
            scuI_grad_list = []

            for i in range(nq):
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
                scxi_grad_list.append(scxi_grad)
                scxI_grad_list.append(scxI_grad)
                scui_grad_list.append(scui_grad)
                scuI_grad_list.append(scuI_grad)

            # Build per-cable auxiliary systems before launching the sharded JAX computation.
            auxsys_start_time = TM.time()
            for i in range(nq):
                auxSysi   = self.Get_AuxSys_DDP_Cable(
                    opt_solc_list[i],
                    ref_xq[i],
                    ref_uq[i*self.nui:(i+1)*self.nui],
                    scxc[i],
                    scuc[i],
                    scxC[i],
                    scuC[i],
                    weight2,
                    i_admm
                )
                AuxSysc  += [auxSysi]
            auxsys_build_time = (TM.time() - auxsys_start_time)*1000

            opt_solc_batch = self._pack_cable_optsol_batch(opt_solc_list)
            auxsysc_batch = self._pack_cable_auxsys_batch(AuxSysc)
            scxi_grad_batch = self._stack_float_batch(scxi_grad_list)
            scxI_grad_batch = self._stack_float_batch(scxI_grad_list)
            scui_grad_batch = self._stack_float_batch(scui_grad_list)
            scuI_grad_batch = self._stack_float_batch(scuI_grad_list)
            cable_axis_size = self._jax_subp1_agent_device_count(nq)
            cable_devices = self.jax_subp1_agent_devices[:cable_axis_size]
            opt_solc_sharded, _ = self._shard_pytree_batch_fixed_axis(opt_solc_batch, nq, cable_axis_size, devices=cable_devices)
            auxsysc_sharded, _ = self._shard_pytree_batch_fixed_axis(auxsysc_batch, nq, cable_axis_size, devices=cable_devices)
            scxi_grad_sharded, _ = self._shard_pytree_batch_fixed_axis(scxi_grad_batch, nq, cable_axis_size, devices=cable_devices)
            scxI_grad_sharded, _ = self._shard_pytree_batch_fixed_axis(scxI_grad_batch, nq, cable_axis_size, devices=cable_devices)
            scui_grad_sharded, _ = self._shard_pytree_batch_fixed_axis(scui_grad_batch, nq, cable_axis_size, devices=cable_devices)
            scuI_grad_sharded, _ = self._shard_pytree_batch_fixed_axis(scuI_grad_batch, nq, cable_axis_size, devices=cable_devices)
            active_mask_sharded, _ = self._shard_array_batch_fixed_axis(np.ones((nq,), dtype=np.bool_), nq, cable_axis_size, devices=cable_devices)
            pix_dis_cable = np.float64(self.open_loop_penalty(weight2[-4], weight2[-2], i_admm, int(self.max_iter_ADMM)))
            piu_dis_cable = np.float64(self.open_loop_penalty(weight2[-3], weight2[-1], i_admm, int(self.max_iter_ADMM)))

            seq_single_cable_time = None
            seq_all_cable_time = None
            if self.enable_cable_parallel_diagnostics and nq > 0:
                seq_start_time = TM.time()
                _ = self.DDP_Cable_Gradient(
                    opt_solc_list[0], AuxSysc[0],
                    scxi_grad_list[0], scxI_grad_list[0], scui_grad_list[0], scuI_grad_list[0],
                    weight2[-4], weight2[-3], weight2[-2], weight2[-1],
                    int(self.max_iter_ADMM), i_admm
                )
                seq_single_cable_time = (TM.time() - seq_start_time) * 1000

                seq_start_time = TM.time()
                for i in range(nq):
                    _ = self.DDP_Cable_Gradient(
                        opt_solc_list[i], AuxSysc[i],
                        scxi_grad_list[i], scxI_grad_list[i], scui_grad_list[i], scuI_grad_list[i],
                        weight2[-4], weight2[-3], weight2[-2], weight2[-1],
                        int(self.max_iter_ADMM), i_admm
                    )
                seq_all_cable_time = (TM.time() - seq_start_time) * 1000

            if not hasattr(self, "_subp1_pmap_warmed_axes"):
                self._subp1_pmap_warmed_axes = set()
            if cable_axis_size not in self._subp1_pmap_warmed_axes:
                _ = self._block_tree_ready(self._get_sharded_ddp_cable_gradient_fn(cable_axis_size)(
                    opt_solc_sharded, auxsysc_sharded, scxi_grad_sharded, scxI_grad_sharded, scui_grad_sharded, scuI_grad_sharded,
                    active_mask_sharded, pix_dis_cable, piu_dis_cable
                ))
                _ = self._block_tree_ready(self._get_sharded_caos_cable_gradient_fn(cable_axis_size)(
                    opt_solc_sharded, auxsysc_sharded, scxi_grad_sharded, scxI_grad_sharded, scui_grad_sharded, scuI_grad_sharded,
                    active_mask_sharded, pix_dis_cable, piu_dis_cable
                ))
                _ = self._block_tree_ready(self._get_sharded_cao_cable_gradient_fn(cable_axis_size)(
                    opt_solc_sharded, auxsysc_sharded, scxi_grad_sharded, scxI_grad_sharded, scui_grad_sharded, scuI_grad_sharded,
                    active_mask_sharded, pix_dis_cable, piu_dis_cable
                ))
                _ = self._block_tree_ready(self._get_sharded_pdp_cable_gradient_fn(cable_axis_size)(
                    opt_solc_sharded, auxsysc_sharded, scxi_grad_sharded, scxI_grad_sharded, scui_grad_sharded, scuI_grad_sharded,
                    active_mask_sharded, pix_dis_cable, piu_dis_cable
                ))
                self._subp1_pmap_warmed_axes.add(cable_axis_size)

            vmap_start_time = TM.time()
            grad_DDP_batch = self._get_sharded_ddp_cable_gradient_fn(cable_axis_size)(
                opt_solc_sharded, auxsysc_sharded, scxi_grad_sharded, scxI_grad_sharded, scui_grad_sharded, scuI_grad_sharded,
                active_mask_sharded, pix_dis_cable, piu_dis_cable
            )
            grad_DDP_batch = self._block_tree_ready(grad_DDP_batch)
            gradtimeOur_sum = (TM.time() - vmap_start_time)*1000

            vmap_start_time = TM.time()
            grad_Caos_batch = self._get_sharded_caos_cable_gradient_fn(cable_axis_size)(
                opt_solc_sharded, auxsysc_sharded, scxi_grad_sharded, scxI_grad_sharded, scui_grad_sharded, scuI_grad_sharded,
                active_mask_sharded, pix_dis_cable, piu_dis_cable
            )
            grad_Caos_batch = self._block_tree_ready(grad_Caos_batch)
            gradtimeCaos_sum = (TM.time() - vmap_start_time)*1000

            vmap_start_time = TM.time()
            grad_Cao_batch = self._get_sharded_cao_cable_gradient_fn(cable_axis_size)(
                opt_solc_sharded, auxsysc_sharded, scxi_grad_sharded, scxI_grad_sharded, scui_grad_sharded, scuI_grad_sharded,
                active_mask_sharded, pix_dis_cable, piu_dis_cable
            )
            grad_Cao_batch = self._block_tree_ready(grad_Cao_batch)
            gradtimeCao_sum = (TM.time() - vmap_start_time)*1000

            vmap_start_time = TM.time()
            grad_PDP_batch = self._get_sharded_pdp_cable_gradient_fn(cable_axis_size)(
                opt_solc_sharded, auxsysc_sharded, scxi_grad_sharded, scxI_grad_sharded, scui_grad_sharded, scuI_grad_sharded,
                active_mask_sharded, pix_dis_cable, piu_dis_cable
            )
            grad_PDP_batch = self._block_tree_ready(grad_PDP_batch)
            gradtimePDP_sum = (TM.time() - vmap_start_time)*1000

            unpack_start_time = TM.time()
            grad_outc = self._unshard_pytree_batch(grad_DDP_batch, nq)
            grad_outcCao = self._unshard_pytree_batch(grad_Cao_batch, nq)
            grad_outcPDP = self._unshard_pytree_batch(grad_PDP_batch, nq)
            unpack_time = (TM.time() - unpack_start_time)*1000

            gradtimeOur_avgcable = gradtimeOur_sum / nq if nq > 0 else 0
            gradtimeCaos_avgcable = gradtimeCaos_sum / nq if nq > 0 else 0
            gradtimeCao_avgcable = gradtimeCao_sum / nq if nq > 0 else 0
            gradtimePDP_avgcable = gradtimePDP_sum / nq if nq > 0 else 0
            Gradtime_1 += gradtimeOur_sum
            # gradients of Subproblem2
            opt_sol1_c = Opt_Sol1_c[i_admm]
            opt_sol2   = Opt_Sol2[i_admm]
            auxSys2    = self.Get_AuxSys_SubP2(opt_sol,opt_sol1_c,opt_sol2,scxL,scuL,scxC,scuC,Pauto,i_admm)
            subp2_batches = self._prepare_subp2_batches(
                auxSys2, grad_outl, grad_outc, scxL_grad, scuL_grad, scxC_grad, scuC_grad,
                weight1[-4], weight1[-3], weight1[-2], weight1[-1],
                weight2[-4], weight2[-3], weight2[-2], weight2[-1],
                int(self.max_iter_ADMM), i_admm
            )
            if not hasattr(self, "_subp2_pmap_warmed_axes"):
                self._subp2_pmap_warmed_axes = set()
            if subp2_batches["subp2_axis_size"] not in self._subp2_pmap_warmed_axes:
                _ = self._block_tree_ready(self._run_subp2_horizon_solves(subp2_batches))
                self._subp2_pmap_warmed_axes.add(subp2_batches["subp2_axis_size"])
            grad_out2, Grad2Time  = self.SubP2_Gradient(auxSys2,grad_outl,grad_outc,scxL_grad,scuL_grad,scxC_grad,scuC_grad,weight1[-4],weight1[-3],weight1[-2],weight1[-1],weight2[-4],weight2[-3],weight2[-2],weight2[-1],int(self.max_iter_ADMM),i_admm, prepared_batches=subp2_batches) 
            Gradtime_2 += Grad2Time
            # gradients of Subproblem3
            auxSys3    = self.Get_AuxSys_SubP3(opt_sol,opt_sol1_c,opt_sol2,weight1,weight2,i_admm)
            subp3_batches = self._prepare_subp3_batches(grad_outl, grad_outc, grad_out2, scxL_grad, scuL_grad, scxC_grad, scuC_grad, auxSys3)
            if not hasattr(self, "_subp3_pmap_warmed_axes"):
                self._subp3_pmap_warmed_axes = set()
            subp3_warm_axis = int(subp3_batches.get("subp3_axis_size", subp3_batches.get("load_axis_size", 1)))
            subp3_warm_key = (
                subp3_warm_axis,
                tuple(subp3_batches["load_state_pack_sharded"].shape),
                tuple(subp3_batches["load_ctrl_pack_sharded"].shape),
                tuple(subp3_batches["cable_state_pack_sharded"].shape),
                tuple(subp3_batches["cable_ctrl_pack_sharded"].shape),
            )
            if subp3_warm_key not in self._subp3_pmap_warmed_axes:
                _ = self.SubP3_Gradient(auxSys3,grad_outl,grad_outc,grad_out2,scxL_grad,scuL_grad,scxC_grad,scuC_grad,weight1[-4],weight1[-3],weight1[-2],weight1[-1],weight2[-4],weight2[-3],weight2[-2],weight2[-1],int(self.max_iter_ADMM),i_admm, prepared_batches=subp3_batches)
                self._subp3_pmap_warmed_axes.add(subp3_warm_key)
            start_time           = TM.time()
            grad_out3  = self.SubP3_Gradient(auxSys3,grad_outl,grad_outc,grad_out2,scxL_grad,scuL_grad,scxC_grad,scuC_grad,weight1[-4],weight1[-3],weight1[-2],weight1[-1],weight2[-4],weight2[-3],weight2[-2],weight2[-1],int(self.max_iter_ADMM),i_admm, prepared_batches=subp3_batches)
            gradtime_3_total = (TM.time() - start_time)*1000
            gradtime_3_exec = getattr(self, "_last_subp3_exec_time_ms", None)
            gradtime_3 = gradtime_3_exec if gradtime_3_exec is not None else gradtime_3_total
            Gradtime_3 += gradtime_3
            print("Grad3time:--- %s ms ---" % format(gradtime_3,'.2f'))
            if self.enable_subp3_parallel_diagnostics:
                print("SubP3_total_time:--- %s ms ---" % format(gradtime_3_total,'.2f'))
            # backward time at each ADMM iteration
            ADMM_time  = gradtimeOur + gradtimeOur_sum + Grad2Time + gradtime_3
            backward_time += ADMM_time
            # update
            scxl       = opt_sol2['scxl_traj']
            scul       = opt_sol2['scul_traj']
            scxc       = opt_sol2['scxc_traj']
            scuc       = opt_sol2['scuc_traj']
            opt_sol3   = Opt_Sol3[i_admm]
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
            ul_grad    = grad_outl['ul_grad']
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
            
           
            gError1     = 0
            gErroru1    = 0
            gError1_c   = 0
            gErroru1_c  = 0
            for k in range(self.N):
                gerror1 = xl_grad[k+1] - scxl_grad[k+1] 
                gError1 += LA.norm(gerror1,ord='fro')
                gerroru1 = ul_grad[k] - scul_grad[k]
                gErroru1 += LA.norm(gerroru1,ord='fro')
                for i in range(int(self.nq)):
                    gerror1_c = grad_outc[i]['xi_grad'][k+1] - np.reshape(scxc_grad[k+1][i*self.nxi:(i+1)*self.nxi,:],(self.nxi,self.n_Pauto))
                    gError1_c += LA.norm(gerror1_c,ord='fro')
                    gerroru1_c = grad_outc[i]['ui_grad'][k] - np.reshape(scuc_grad[k][i*self.nui:(i+1)*self.nui,:],(self.nui,self.n_Pauto))
                    gErroru1_c += LA.norm(gerroru1_c,ord='fro')
            gmeanerror1 = np.sqrt(gError1**2+gErroru1**2)/self.N
            gmeanerror1_c = np.sqrt(gError1_c**2+gErroru1_c**2)/(self.N*self.nq) 
            gMeanerror_l += [gmeanerror1]
            gMeanerror_c += [gmeanerror1_c]       

            meanerror1 = Error1/self.N
            meanerror2 = Error2/self.N    
            MeanerrorCao += [meanerror1]
            MeanerrorPDP += [meanerror2]
            meanerror1_c = Error1_c/(self.N*self.nq)
            meanerror2_c = Error2_c/(self.N*self.nq)
            MeanerrorCao_c += [meanerror1_c]
            MeanerrorPDP_c += [meanerror2_c]

            if i_admm >= 0: #== self.max_iter_ADMM-1:
            
                print("g_Our:--- %s ms ---" % format(gradtimeOur,'.2f'))
                print("g_Cao_s:--- %s ms ---" % format(gradtimeCaos,'.2f'))
                print("g_Cao:--- %s ms ---" % format(gradtimeCao,'.2f'))
                print("g_PDP:--- %s ms ---" % format(gradtimePDP,'.2f'))
                print("g_Our_cable:--- %s ms ---" % format(gradtimeOur_avgcable,'.2f'))
                print("g_Cao_s_cable:--- %s ms ---" % format(gradtimeCaos_avgcable,'.2f'))
                print("g_Cao_cable:--- %s ms ---" % format(gradtimeCao_avgcable,'.2f'))
                print("g_PDP_cable:--- %s ms ---" % format(gradtimePDP_avgcable,'.2f'))
                print("cable_pmap_exec_DDPour:--- %s ms ---" % format(gradtimeOur_sum,'.2f'))
                if seq_single_cable_time is not None and seq_all_cable_time is not None:
                    print("cable_seq_single_DDPour:--- %s ms ---" % format(seq_single_cable_time, '.2f'))
                    print("cable_seq_all_DDPour:--- %s ms ---" % format(seq_all_cable_time, '.2f'))
                    print("cable_pmap_speedup_DDPour:--- %s x ---" % format(seq_all_cable_time / max(gradtimeOur_sum, 1e-12), '.2f'))
                print("cable_pmap_exec_Cao_s:--- %s ms ---" % format(gradtimeCaos_sum,'.2f'))
                print("cable_pmap_exec_Cao:--- %s ms ---" % format(gradtimeCao_sum,'.2f'))
                print("cable_pmap_exec_PDP:--- %s ms ---" % format(gradtimePDP_sum,'.2f'))
                # print("cable_unpack:--- %s ms ---" % format(unpack_time,'.2f'))
                # print("cable_total_wall:--- %s ms ---" % format(parallel_time,'.2f'))
                print('meanerrorCao=',meanerror1,'meanerrorPDP=',meanerror2)
                print('meanerrorCao_c=',meanerror1_c,'meanerrorPDP_c=',meanerror2_c)
                gMeanerror_l = [float(x) for x in gMeanerror_l]
                gMeanerror_c = [float(x) for x in gMeanerror_c]
                print('gMeanerror_l=',gMeanerror_l,'gMeanerror_c=',gMeanerror_c)

        # averaged gradient computation time across ADMM iterations
        Gradtime_1_avg = Gradtime_1 / int(self.max_iter_ADMM)
        Gradtime_2_avg = Gradtime_2 / int(self.max_iter_ADMM)
        Gradtime_3_avg = Gradtime_3 / int(self.max_iter_ADMM)
        print("Average Grad1 time:--- %s ms ---" % format(Gradtime_1_avg,'.2f'))
        print("Average Grad2 time:--- %s ms ---" % format(Gradtime_2_avg,'.2f'))
        print("Average Grad3 time:--- %s ms ---" % format(Gradtime_3_avg,'.2f'))
        output = {"Grad_Out1l": Grad_Out1l, "Grad_Out1c": Grad_Out1c, "Grad_Out2": Grad_Out2, "Grad_Out3": Grad_Out3,
                  "GradTime": GradTime, "GradTimeCaos": GradTimeCaos, "GradTimeCao": GradTimeCao, "GradTimePDP": GradTimePDP,
                  "GradTime_c": GradTime_c, "GradTimeCaos_c": GradTimeCaos_c, "GradTimeCao_c": GradTimeCao_c, "GradTimePDP_c": GradTimePDP_c,
                  "MeanerrorCao": MeanerrorCao, "MeanerrorPDP": MeanerrorPDP, "MeanerrorCao_c": MeanerrorCao_c, "MeanerrorPDP_c": MeanerrorPDP_c,
                  "gMeanerror_l": gMeanerror_l, "gMeanerror_c": gMeanerror_c, "backward_time": backward_time,
                  "Gradtime_1_avg": Gradtime_1_avg, "Gradtime_2_avg": Gradtime_2_avg, "Gradtime_3_avg": Gradtime_3_avg
                  }
        
        return output
    
    

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
        self.gamma_min = -3
        self.gamma_max = 3
        #------------- loss definition -------------#
        # tracking loss
        track_error_l = self.xl - self.xl_ref
        track_error_i = self.xi - self.xi_ref
        self.loss_track_l = track_error_l.T@track_error_l
        self.weight_i     = np.diag(np.array([1,1,1,0,0,0,0,0,0,0,0,0,1,0])) 
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
    

    # def adaptive_meta_loss_weights(self,loss_t,loss_rp,wt): # using ideas from heuristic adaptive ADMM penalty parameters
    #     if loss_t > 1.25*loss_rp:
    #         wt_new = np.clip(1.5*wt,0.2,5)
    #     elif loss_rp > 1.25*loss_t:
    #         wt_new = np.clip(wt/1.5,0.2,5)
    #     else:
    #         wt_new = wt
    #     return wt_new
    
   
    def adaptive_meta_loss_weights(self, loss_t, loss_rp, g_t, g_rp, wt, alpha=7, beta_w=0.8, eps=1e-8, k_min=0.2, k_max=200.0):
        def _as_scalar(x, name):
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.size != 1:
                raise ValueError(f"{name} must be scalar-like, got shape {x_arr.shape}")
            return float(x_arr.reshape(-1)[0])

        loss_t = _as_scalar(loss_t, "loss_t")
        loss_rp = _as_scalar(loss_rp, "loss_rp")
        g_t = _as_scalar(g_t, "g_t")
        g_rp = _as_scalar(g_rp, "g_rp")
        wt = _as_scalar(wt, "wt")

        # -------- safer auto-K (clip + exponent) --------
        k_auto = (loss_t / (loss_rp + eps)) ** alpha
        k_auto = float(np.clip(k_auto, k_min, k_max))
        wt_target = 2.0 * k_auto * g_rp / (g_t + k_auto * g_rp + eps)
        # -------- slow weight update --------
        wt_new = (1 - beta_w) * wt + beta_w * wt_target
        wt_new = float(np.clip(wt_new, 0.01, 1.99))
        wrp_new = 2.0 - wt_new

        return wt_new, wrp_new, k_auto


    # def adaptive_meta_loss_weights(self, loss_t, loss_rp, wt,alpha=5, beta_w=0.6, eps=1e-8,k_min=0.25, k_max=10.0):

    #     # -------- initialize reference losses once --------
    #     if not hasattr(self, "Lt0"):
    #         self.Lt0  = float(loss_t)  + eps
    #         self.Lrp0 = float(loss_rp) + eps

    #     # -------- relative progress (dimensionless) --------
    #     r_t  = loss_t  / self.Lt0
    #     r_rp = loss_rp / self.Lrp0

    #     # -------- loss-ratio based auto-K --------
    #     k_auto = (r_t / (r_rp + eps)) ** alpha
    #     k_auto = float(np.clip(k_auto, k_min, k_max))

    #     # -------- target weights from loss ratio only --------
    #     wt_target = 2.0 * k_auto / (1.0 + k_auto)

    #     # -------- slow weight update --------
    #     wt_new = (1 - beta_w) * wt + beta_w * wt_target
    #     wt_new = float(np.clip(wt_new, 0.01, 1.99))
    #     wrp_new = 2.0 - wt_new

    #     return wt_new, wrp_new, k_auto



     


    def Set_Parameters(self,tunable_para):
        weight       = np.zeros(self.n_Pauto)
        for k in range(self.n_Pauto):
            weight[k]= self.p_min + (self.p_max - self.p_min) * 1/(1+np.exp(-tunable_para[k])) # sigmoid boundedness
            if k == self.npl-2:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1+np.exp(-tunable_para[k]))
            elif k == self.npl-1:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1+np.exp(-tunable_para[k]))
            elif k == self.n_Pauto-2:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1+np.exp(-tunable_para[k]))
            elif k == self.n_Pauto-1:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1+np.exp(-tunable_para[k]))

        return weight
    

    def Set_Parameters_nn_l(self,tunable_para):
        weight       = np.zeros(self.npl)
        for k in range(self.npl):
            weight[k]= self.p_min + (self.p_max - self.p_min) * tunable_para[0,k] # sigmoid boundedness
            if k == self.npl-2:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * tunable_para[0,k]
            elif k == self.npl-1:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * tunable_para[0,k]
        return weight
    
    def Set_Parameters_nn_i(self,tunable_para):
        weight       = np.zeros(self.npi)
        for k in range(self.npi):
            weight[k]= self.p_min + (self.p_max - self.p_min) * tunable_para[0,k] # sigmoid boundedness
            if k == self.npi-2:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * tunable_para[0,k]
            elif k == self.npi-1:
                weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * tunable_para[0,k]
        return weight
    

    def ChainRule_Gradient(self,tunable_para):
        Tunable      = SX.sym('Tp',1,self.n_Pauto)
        Weight       = SX.sym('wp',1,self.n_Pauto)
        for k in range(self.n_Pauto):
            Weight[k]= self.p_min + (self.p_max - self.p_min) * 1/(1 + exp(-Tunable[k])) # sigmoid boundedness
            if k == self.npl-2:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1 + exp(-Tunable[k]))
            elif k == self.npl-1:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1 + exp(-Tunable[k]))
            elif k == self.n_Pauto-2:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1 + exp(-Tunable[k]))
            elif k == self.n_Pauto-1:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * 1/(1 + exp(-Tunable[k]))
        dWdT         = jacobian(Weight,Tunable)
        dWdT_fn      = Function('dWdT',[Tunable],[dWdT],['Tp0'],['dWdT_f'])
        weight_grad  = dWdT_fn(Tp0=tunable_para)['dWdT_f'].full()

        return weight_grad
    
    def ChainRule_Gradient_nn_l(self,tunable_para):
        Tunable      = SX.sym('Tp',1,self.npl)
        Weight       = SX.sym('wp',1,self.npl)
        for k in range(self.npl):
            Weight[k]= self.p_min + (self.p_max - self.p_min) * Tunable[k] # sigmoid boundedness
            if k == self.npl-2:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * Tunable[k] 
            elif k == self.npl-1:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * Tunable[k] 
        dWdT         = jacobian(Weight,Tunable)
        dWdT_fn      = Function('dWdT',[Tunable],[dWdT],['Tp0'],['dWdT_f'])
        weight_grad  = dWdT_fn(Tp0=tunable_para)['dWdT_f'].full()
        return weight_grad
    
    def ChainRule_Gradient_nn_i(self,tunable_para):
        Tunable      = SX.sym('Tp',1,self.npi)
        Weight       = SX.sym('wp',1,self.npi)
        for k in range(self.npi):
            Weight[k]= self.p_min + (self.p_max - self.p_min) * Tunable[k] # sigmoid boundedness
            if k == self.npi-2:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * Tunable[k]
            elif k == self.npi-1:
                Weight[k]= self.gamma_min + (self.gamma_max - self.gamma_min) * Tunable[k]
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
            loss_track += error_iN.T@self.weight_i@error_iN # zero weight_i
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
        k_admm          = -1 # the last, the most recent trajectories and gradients
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
        gloss_t     = LA.norm(dltdw)
        gloss_rp    = LA.norm(dlrpdw)

        return dldw, loss, loss_track, loss_resid, gloss_t,gloss_rp
  





















    





        

    

                


                    







    


        



    


    

    

    

        
        
            





    
