import os
import sys
import io
import contextlib
import threading
import multiprocessing as MP
import atexit
from concurrent.futures import ThreadPoolExecutor

_RUNNING_UNDER_VSCODE_OR_DEBUGGER = (
    sys.gettrace() is not None
    or "debugpy" in sys.modules
    or os.environ.get("VSCODE_PID") is not None
    or os.environ.get("PYDEVD_LOAD_VALUES_ASYNC") is not None
)

_DEFAULT_JAX_CPU_DEVICE_COUNT = "256"
_force_host_flag = f"--xla_force_host_platform_device_count={_DEFAULT_JAX_CPU_DEVICE_COUNT}"
_existing_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count=" in _existing_xla_flags and (
    _RUNNING_UNDER_VSCODE_OR_DEBUGGER or "DIFFCOORD_JAX_CPU_DEVICE_COUNT" in os.environ
):
    _xla_flags = [
        flag for flag in _existing_xla_flags.split()
        if not flag.startswith("--xla_force_host_platform_device_count=")
    ]
    os.environ["XLA_FLAGS"] = " ".join(_xla_flags + [_force_host_flag]).strip()
elif "--xla_force_host_platform_device_count=" not in _existing_xla_flags:
    os.environ["XLA_FLAGS"] = f"{_existing_xla_flags} {_force_host_flag}".strip()

for _env_name, _env_value in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ[_env_name] = _env_value
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("DIFFCOORD_SUBP1_PRINTS", "0")

from casadi import *
import numpy as np
from numpy import linalg as LA
import math
from scipy.spatial.transform import Rotation as Rot
from scipy import linalg as sLA
from scipy.linalg import null_space
import time as TM

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except Exception:
    jax = None
    jnp = None
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import ArpackNoConvergence

_KINODYN_SUBP2_PROCESS_STATE = {}
_KINODYN_CABLE_DDP_PROCESS_STATE = {}


def _disable_debugger_trace_in_worker():
    if os.environ.get("DIFFCOORD_DISABLE_DEBUGGER_IN_WORKERS", "1") != "1":
        return
    for _env_name in ("PYDEVD_LOAD_VALUES_ASYNC", "PYDEVD_USE_FRAME_EVAL", "DEBUGPY_RUNNING"):
        os.environ.pop(_env_name, None)
    try:
        with open(os.devnull, "w") as _devnull, contextlib.redirect_stderr(_devnull):
            sys.settrace(None)
    except Exception:
        pass
    try:
        threading.settrace(None)
    except Exception:
        pass


def _block_jax_tree_ready(tree):
    if jax is None:
        return tree
    for leaf in jax.tree_util.tree_leaves(tree):
        block_until_ready = getattr(leaf, "block_until_ready", None)
        if block_until_ready is not None:
            block_until_ready()
    return tree


@contextlib.contextmanager
def _no_debugger_trace_during_fork():
    if os.environ.get("DIFFCOORD_DISABLE_DEBUGGER_IN_WORKERS", "1") != "1":
        yield
        return
    parent_trace = sys.gettrace()
    parent_thread_trace = threading.gettrace()
    try:
        with open(os.devnull, "w") as _devnull, contextlib.redirect_stderr(_devnull):
            sys.settrace(None)
            threading.settrace(None)
        yield
    finally:
        try:
            with open(os.devnull, "w") as _devnull, contextlib.redirect_stderr(_devnull):
                sys.settrace(parent_trace)
        except Exception:
            pass
        try:
            threading.settrace(parent_thread_trace)
        except Exception:
            pass


def _kinodyn_subp2_process_solve(args):
    kind, k, x0_k, p_k = args
    st = _KINODYN_SUBP2_PROCESS_STATE
    if kind == "N":
        sol = st["solver2N"](
            x0=x0_k,
            lbx=st["lbw2N"],
            ubx=st["ubw2N"],
            p=p_k,
            lbg=st["lbg2N"],
            ubg=st["ubg2N"],
        )
    else:
        sol = st["solver2"](
            x0=x0_k,
            lbx=st["lbw2"],
            ubx=st["ubw2"],
            p=p_k,
            lbg=st["lbg2"],
            ubg=st["ubg2"],
        )
    return kind, k, np.asarray(sol["x"].full(), dtype=np.float64).reshape(-1)


def _kinodyn_cable_ddp_process_solve(item):
    i, para_i = item
    planner = _KINODYN_CABLE_DDP_PROCESS_STATE["planner"]
    para_i = np.asarray(para_i, dtype=np.float64).reshape(-1)
    xi_fb = para_i[0:planner.nxi]
    Ref_xi = para_i[planner.nxi:planner.nxi * (planner.N + 2)]
    Ref_ui = para_i[
        planner.nxi + planner.nxi * (planner.N + 1):
        planner.nxi + planner.nxi * (planner.N + 1) + planner.nui
    ]
    n_scxi_start = planner.nxi * (planner.N + 2) + planner.nui
    scxi = para_i[n_scxi_start:n_scxi_start + planner.nxi * (planner.N + 1)]
    n_scxI_start = n_scxi_start + planner.nxi * (planner.N + 1)
    scxI = para_i[n_scxI_start:n_scxI_start + planner.nxi * (planner.N + 1)]
    n_scui_start = n_scxI_start + planner.nxi * (planner.N + 1)
    scui = para_i[n_scui_start:n_scui_start + planner.nui * planner.N]
    n_scuI_start = n_scui_start + planner.nui * planner.N
    scuI = para_i[n_scuI_start:n_scuI_start + planner.nui * planner.N]
    n_weig_start = n_scuI_start + planner.nui * planner.N
    weight2 = para_i[n_weig_start:n_weig_start + planner.npi]
    i_admm = para_i[-1]
    with contextlib.redirect_stdout(io.StringIO()):
        opt_sol_i = planner.DDP_Cable_ADMM_Subp1(
            xi_fb, Ref_xi, Ref_ui, weight2, scxi, scui, scxI, scuI, 10, 1e-2, i_admm
        )
    return i, opt_sol_i


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
        self.Jl     = self.Jlcom.copy()
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
        self.rov    = sysm_para[12] # the radius of horizontal obstacle [m]
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
        self._solver2_map_cache = {}
        self._solver2_map_bound_cache = {}
        self._solver2_process_pool = None
        self._solver2_process_pool_config = None
        self._cable_ddp_process_pool = None
        self._cable_ddp_process_pool_config = None
        if jax is not None:
            jax_devices = jax.devices()
            load_device = jax_devices[0] if jax_devices else None
            self._load_solver_single_jit = jax.jit(MPC_Planner._ddp_load_exact_single, device=load_device)
            self._load_solver_jit = jax.jit(MPC_Planner._ddp_load_exact_batched, device=load_device)
            self._load_policy_solver_single_jit = jax.jit(MPC_Planner._ddp_load_policy_single, device=load_device)
            self._load_traj_solver_single_jit = jax.jit(MPC_Planner._ddp_load_traj_single, device=load_device)
            self._cable_solver_single_jit = jax.jit(MPC_Planner._ddp_cable_exact_single)
            self._cable_solver_jit = jax.jit(MPC_Planner._ddp_cable_exact_batched)
            self._cable_traj_solver_single_jit = jax.jit(MPC_Planner._ddp_cable_traj_single)
            self._cable_traj_solver_jit = jax.jit(MPC_Planner._ddp_cable_traj_batched)
            if os.environ.get("DIFFCOORD_KINODYN_SUBP1_CONCURRENT", "1") == "1" and len(jax_devices) > int(self.nq):
                cable_devices = jax_devices[1:int(self.nq)+1]
            else:
                cable_devices = jax_devices[:int(self.nq)]
            if (
                os.environ.get("DIFFCOORD_KINODYN_CABLE_PMAP", "1") == "1"
                and len(cable_devices) == int(self.nq)
                and int(self.nq) > 1
            ):
                self._cable_solver_pmap = jax.pmap(MPC_Planner._ddp_cable_exact_single, devices=cable_devices)
                self._cable_traj_solver_pmap = jax.pmap(MPC_Planner._ddp_cable_traj_single, devices=cable_devices)
                self._cable_solver_pmap_size = int(self.nq)
            else:
                self._cable_solver_pmap = None
                self._cable_traj_solver_pmap = None
                self._cable_solver_pmap_size = 0
        else:
            self._load_solver_single_jit = None
            self._load_solver_jit = None
            self._load_policy_solver_single_jit = None
            self._load_traj_solver_single_jit = None
            self._cable_solver_single_jit = None
            self._cable_solver_jit = None
            self._cable_traj_solver_single_jit = None
            self._cable_traj_solver_jit = None
            self._cable_solver_pmap = None
            self._cable_traj_solver_pmap = None
            self._cable_solver_pmap_size = 0
        self._subp3_update_jit = None
        atexit.register(self.close_solver2_process_pool)
        atexit.register(self.close_cable_ddp_process_pool)
    
    def Rotational_Inertia(self,rp):
        # rp=(x,y,0), a column vector, is the coordinate of the point-mass added on the uniform circular plate in its body frame 
        ratio_m    = self.m1*self.m2/self.ml
        self.Jl    = self.Jlcom + ratio_m*(rp.T@rp*np.identity(3)-rp@rp.T)
        self.Jl_inv = LA.inv(self.Jl)

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
        self.ui_bound = 1e3
        self.ui_lb = self.nui*[-self.ui_bound]
        self.ui_ub = self.nui*[self.ui_bound]

    def SetDyns(self, model_l, model_i):
        self.model_l = self.xl + self.dt*model_l # 4th-order Runge-Kutta discrete-time load dynamics model
        self.model_i = self.xi + self.dt*model_i # 4th-order Runge-Kutta discrete-time cable dynamics model
        self.model_l_fn = Function('mdynl',[self.xl, self.ul],[self.model_l],['xl0','ul0'],['mdynlf'])
        self.model_i_fn = Function('mdyni',[self.xi, self.ui],[self.model_i],['xi0','ui0'],['mdynif'])

    def SetWeightPara(self):
        # self.nwsl    = self.nxl
        self.para_l  = SX.sym('paral',1,(2*self.nxl+self.nul+4)) # including the ADMM penalty parameter
        self.npl     = self.para_l.numel()
        self.para_i  = SX.sym('parai',1,(2*self.nxi+self.nui+4)) # including the ADMM penalty parameter
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
        self.pix_dis  = self.open_loop_penalty(self.para_i[0,-4],self.para_i[0,-2],self.a,ADMM_max)
        self.piu_dis  = self.open_loop_penalty(self.para_i[0,-3],self.para_i[0,-1],self.a,ADMM_max)
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

   
    def DDP_Load_ADMM_Subp1(self,xl_0,Ref_xl,Ref_ul,weight1,scxl,scul,scxL,scuL,max_iter,e_tol,i_admm,need_derivs=True):
        load_backend = os.environ.get("DIFFCOORD_KINODYN_LOAD_DDP_BACKEND", "jax").strip().lower()
        if (
            load_backend in ("jax", "jit", "batched")
            and jax is not None
            and os.environ.get("DIFFCOORD_SUBP1_EXACT_JAX", "1") == "1"
            and getattr(self, "_load_solver_jit", None) is not None
        ):
            if not hasattr(self, "max_iter_ADMM"):
                self.max_iter_ADMM = max_iter
            para_l = np.concatenate((
                np.asarray(xl_0, dtype=np.float64).reshape(-1),
                np.asarray(Ref_xl, dtype=np.float64).reshape(-1),
                np.asarray(Ref_ul, dtype=np.float64).reshape(-1),
                np.asarray(scxl, dtype=np.float64).reshape(-1),
                np.asarray(scxL, dtype=np.float64).reshape(-1),
                np.asarray(scul, dtype=np.float64).reshape(-1),
                np.asarray(scuL, dtype=np.float64).reshape(-1),
                np.asarray(weight1, dtype=np.float64).reshape(-1),
                np.asarray([float(i_admm)], dtype=np.float64),
            ))
            _opt_soll, opt_sol_l = self.jax_MPC_Load_DDP_Planning_SubP1([para_l], need_derivs=need_derivs)
            return opt_sol_l[0]
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
            X_nominal[:,k:k+1] = self.model_i_fn(xi0=X_nominal[:,k],ui0=u_k)['mdynif'].full() # start from a bad state
            U_nominal[:,k:k+1]   = u_k
            cost_prev     += self.Ji_k_fn(xi0=X_nominal[:,k],ui0=u_k,scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],
                                        scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],
                                        refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,parai0=weight2,a0=i_admm)['Ji_kf'].full()
        cost_prev += self.Ji_N_fn(xi0=X_nominal[:,-1],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
                                 scxI0=scxI[self.N*self.nxi:(self.N+1)*self.nxi],parai0=weight2,a0=i_admm)['Ji_Nf'].full()
        # else:
        #     for k in range(self.N):
        #         X_nominal[:,k+1:k+2] = np.reshape(scxi[k*self.nxi:(k+1)*self.nxi],(self.nxi,1))
        #         U_nominal[:,k:k+1]   = np.reshape(scui[k*self.nui:(k+1)*self.nui],(self.nui,1))
        #         cost_prev     += self.Ji_k_fn(xi0=X_nominal[:,k],ui0=U_nominal[:,k],scxi0=scxi[k*self.nxi:(k+1)*self.nxi],scxI0=scxI[k*self.nxi:(k+1)*self.nxi],
        #                                 scui0=scui[k*self.nui:(k+1)*self.nui],scuI0=scuI[k*self.nui:(k+1)*self.nui],
        #                                 refxi0=Ref_xi[k*self.nxi:(k+1)*self.nxi],refui0=Ref_ui,parai0=weight2,a0=i_admm)['Ji_kf'].full()
        #     cost_prev += self.Ji_N_fn(xi0=X_nominal[:,-1],refxi0=Ref_xi[self.N*self.nxi:(self.N+1)*self.nxi],scxi0=scxi[self.N*self.nxi:(self.N+1)*self.nxi],
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


    @staticmethod
    def _jax_skew_times(w, v):
        return jnp.array(
            [
                -w[2] * v[1] + w[1] * v[2],
                w[2] * v[0] - w[0] * v[2],
                -w[1] * v[0] + w[0] * v[1],
            ],
            dtype=w.dtype,
        )

    @staticmethod
    def _jax_load_continuous_dynamics(x, u, params):
        vl = x[3:6]
        ql = x[6:10]
        wl = x[10:13]
        Fl = u[0:3]
        Ml = u[3:6]
        dpl = vl
        dvl = -9.81 * jnp.array([0.0, 0.0, 1.0], dtype=x.dtype) + Fl / params["ml"]
        w1, w2, w3 = wl[0], wl[1], wl[2]
        Omega = jnp.array(
            [[0.0, -w1, -w2, -w3], [w1, 0.0, w3, -w2], [w2, -w3, 0.0, w1], [w3, w2, -w1, 0.0]],
            dtype=x.dtype,
        )
        dql = 0.5 * Omega @ ql
        dwl = params["Jl_inv"] @ (Ml - MPC_Planner._jax_skew_times(wl, params["Jl"] @ wl))
        return jnp.concatenate([dpl, dvl, dql, dwl])

    @staticmethod
    def jax_load_dynamics(x, u, params):
        dt = params["dt"]
        f = MPC_Planner._jax_load_continuous_dynamics
        k1 = f(x, u, params)
        k2 = f(x + 0.5 * dt * k1, u, params)
        k3 = f(x + 0.5 * dt * k2, u, params)
        k4 = f(x + dt * k3, u, params)
        return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    @staticmethod
    def _jax_cable_continuous_dynamics(x, u, params):
        di = x[0:3]
        wi = x[3:6]
        ai = x[6:9]
        ji = x[9:12]
        vti = x[13]
        si = u[0:3]
        ati = u[3]
        return jnp.concatenate(
            [
                MPC_Planner._jax_skew_times(wi, di),
                ai,
                ji,
                si,
                jnp.array([vti, ati], dtype=x.dtype),
            ]
        )

    @staticmethod
    def jax_cable_dynamics_single(x, u, params):
        dt = params["dt"]
        f = MPC_Planner._jax_cable_continuous_dynamics
        k1 = f(x, u, params)
        k2 = f(x + 0.5 * dt * k1, u, params)
        k3 = f(x + 0.5 * dt * k2, u, params)
        k4 = f(x + dt * k3, u, params)
        return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    @staticmethod
    def _build_init_load_traj(ref_ul_seq, params, nxl):
        zero_xl = jnp.zeros((nxl,), dtype=ref_ul_seq.dtype)
        scxl_body = jax.vmap(lambda u: MPC_Planner.jax_load_dynamics(zero_xl, u, params))(ref_ul_seq)
        return jnp.concatenate([scxl_body, jnp.zeros((1, nxl), dtype=ref_ul_seq.dtype)], axis=0)

    @staticmethod
    def _build_init_cable_traj(ref_uq_mat, params, nxi, horizon):
        nq = ref_uq_mat.shape[0]
        nui = ref_uq_mat.shape[1]
        zero_xc = jnp.zeros((nq, nxi), dtype=ref_uq_mat.dtype)
        scxc_body = jax.vmap(lambda x0, ui: MPC_Planner.jax_cable_dynamics_single(x0, ui, params))(zero_xc, ref_uq_mat)
        scxc_traj = jnp.concatenate(
            [jnp.broadcast_to(scxc_body[:, None, :], (nq, horizon, nxi)), jnp.zeros((nq, 1, nxi), dtype=ref_uq_mat.dtype)],
            axis=1,
        )
        scuc_traj = jnp.broadcast_to(ref_uq_mat[:, None, :], (nq, horizon, nui))
        return scxc_traj, scuc_traj

    @staticmethod
    def jax_load_stage_cost(x, u, params):
        diff_x_ref = x - params["ref_x"]
        diff_u_ref = u - params["ref_u"]
        resid_x = x - params["scx"] + params["y_x"] / params["rho_lx"]
        resid_u = u - params["scu"] + params["y_u"] / params["rho_lu"]
        cost = 0.5 * (jnp.sum(diff_x_ref**2 * params["Q_weight"]) + jnp.sum(diff_u_ref**2 * params["R_weight"]))
        cost += 0.5 * params["rho_lx"] * jnp.sum(resid_x**2)
        cost += 0.5 * params["rho_lu"] * jnp.sum(resid_u**2)
        return cost

    @staticmethod
    def jax_load_terminal_cost(x, params):
        diff_x_ref = x - params["ref_x"]
        resid_x = x - params["scx"] + params["y_x"] / params["rho_lx"]
        return 0.5 * jnp.sum(diff_x_ref**2 * params["Q_terminal_weight"]) + 0.5 * params["rho_lx"] * jnp.sum(resid_x**2)

    @staticmethod
    def jax_cable_stage_cost(x, u, params):
        diff_x_ref = x - params["ref_x_i"]
        diff_u_ref = u - params["ref_u_i"]
        resid_x = x - params["scx_i"] + params["y_x_i"] / params["rho_ix"]
        resid_u = u - params["scu_i"] + params["y_u_i"] / params["rho_iu"]
        cost = 0.5 * (jnp.sum(diff_x_ref**2 * params["Qi_weight"]) + jnp.sum(diff_u_ref**2 * params["Ri_weight"]))
        cost += 0.5 * params["rho_ix"] * jnp.sum(resid_x**2)
        cost += 0.5 * params["rho_iu"] * jnp.sum(resid_u**2)
        return cost

    @staticmethod
    def jax_cable_terminal_cost(x, params):
        diff_x_ref = x - params["ref_x_i"]
        resid_x = x - params["scx_i"] + params["y_x_i"] / params["rho_ix"]
        return 0.5 * jnp.sum(diff_x_ref**2 * params["Qi_terminal_weight"]) + 0.5 * params["rho_ix"] * jnp.sum(resid_x**2)

    @staticmethod
    def _ddp_stage_params(params, t):
        merged = dict(params["static"])
        merged.update(jax.tree_util.tree_map(lambda a: a[t], params["stage"]))
        return merged

    @staticmethod
    def _ddp_terminal_params(params):
        merged = dict(params["static"])
        merged.update(params["terminal"])
        return merged

    @staticmethod
    def _ddp_chol_solve(L, B):
        y = jnp.linalg.solve(L, B)
        return jnp.linalg.solve(L.T, y)

    @staticmethod
    def _ddp_bad_initial_rollout(dynamics_fn, x0, us, params):
        zero_x = jnp.zeros_like(x0)

        def one_step(t, u):
            seed_x = jnp.where(t == 0, x0, zero_x)
            return dynamics_fn(seed_x, u, MPC_Planner._ddp_stage_params(params, t))

        xs_body = jax.vmap(one_step)(jnp.arange(us.shape[0]), us)
        return jnp.concatenate([xs_body, zero_x[None, :]], axis=0)

    @staticmethod
    def _ddp_total_cost(cost_fn, term_cost_fn, xs, us, params):
        stage_costs = jax.vmap(
            lambda t, x, u: cost_fn(x, u, MPC_Planner._ddp_stage_params(params, t))
        )(jnp.arange(us.shape[0]), xs[:-1], us)
        terminal_cost = term_cost_fn(xs[-1], MPC_Planner._ddp_terminal_params(params))
        return jnp.sum(stage_costs) + terminal_cost

    @staticmethod
    def _ddp_forward_policy(dynamics_fn, cost_fn, term_cost_fn, x0, xs_nominal, us_nominal, K_fb, k_ff, alpha, params):
        def step(x_k, k):
            delta_x = x_k - xs_nominal[k]
            u_k = us_nominal[k] + K_fb[k] @ delta_x + alpha * k_ff[k]
            x_next = dynamics_fn(x_k, u_k, MPC_Planner._ddp_stage_params(params, k))
            return x_next, (x_k, u_k)

        x_terminal, hist = jax.lax.scan(step, x0, jnp.arange(us_nominal.shape[0]))
        xs_new = jnp.concatenate([hist[0], x_terminal[None, :]], axis=0)
        us_new = hist[1]
        cost_new = MPC_Planner._ddp_total_cost(cost_fn, term_cost_fn, xs_new, us_new, params)
        return xs_new, us_new, cost_new

    @staticmethod
    def _ddp_derivatives(dynamics_fn, cost_fn, term_cost_fn, xs, us, params):
        def one_stage(k, x, u):
            p_k = MPC_Planner._ddp_stage_params(params, k)
            Fx = jax.jacfwd(dynamics_fn, 0)(x, u, p_k)
            Fu = jax.jacfwd(dynamics_fn, 1)(x, u, p_k)
            Qx_bar = jax.grad(cost_fn, 0)(x, u, p_k)
            Qu_bar = jax.grad(cost_fn, 1)(x, u, p_k)
            Qxx_bar = jax.hessian(cost_fn, 0)(x, u, p_k)
            Quu_bar = jax.hessian(cost_fn, 1)(x, u, p_k)
            Qxu_bar = jax.jacfwd(jax.grad(cost_fn, 0), 1)(x, u, p_k)
            return Fx, Fu, Qx_bar, Qu_bar, Qxx_bar, Qxu_bar, Quu_bar

        Fx, Fu, Qx_bar, Qu_bar, Qxx_bar, Qxu_bar, Quu_bar = jax.vmap(one_stage)(
            jnp.arange(us.shape[0]), xs[:-1], us
        )
        p_N = MPC_Planner._ddp_terminal_params(params)
        Vx_N = jax.grad(term_cost_fn, 0)(xs[-1], p_N)
        Vxx_N = jax.hessian(term_cost_fn, 0)(xs[-1], p_N)
        return Fx, Fu, Qx_bar, Qu_bar, Qxx_bar, Qxu_bar, Quu_bar, Vx_N, Vxx_N

    @staticmethod
    def _ddp_backward_pass(dynamics_fn, cost_fn, term_cost_fn, xs, us, params, reg):
        Fx, Fu, Qx_bar, Qu_bar, Qxx_bar, Qxu_bar, Quu_bar, Vx_N, Vxx_N = MPC_Planner._ddp_derivatives(
            dynamics_fn,
            cost_fn,
            term_cost_fn,
            xs,
            us,
            params,
        )
        nu = us.shape[-1]
        eye_u = jnp.eye(nu, dtype=xs.dtype)

        def backward_step(carry, inputs):
            Vx_next, Vxx_next = carry
            Fx_k, Fu_k, Qx_bar_k, Qu_bar_k, Qxx_bar_k, Qxu_bar_k, Quu_bar_k = inputs
            Qx_k = Qx_bar_k + Fx_k.T @ Vx_next
            Qu_k = Qu_bar_k + Fu_k.T @ Vx_next
            Qxx_hat_k = Fx_k.T @ Vxx_next @ Fx_k
            Qxu_hat_k = Fx_k.T @ Vxx_next @ Fu_k
            Quu_hat_k = Fu_k.T @ Vxx_next @ Fu_k
            Qxx_k = Qxx_bar_k + Qxx_hat_k
            Qxu_k = Qxu_bar_k + Qxu_hat_k
            Quu_k = Quu_bar_k + Quu_hat_k
            Quu_reg_k = Quu_k + reg * eye_u
            L = jax.lax.linalg.cholesky(Quu_reg_k, symmetrize_input=False)
            chol_ok = jnp.all(jnp.isfinite(L))
            Quu_inv_k = MPC_Planner._ddp_chol_solve(L, eye_u)
            K_fb_k = MPC_Planner._ddp_chol_solve(L, -Qxu_k.T)
            k_ff_k = MPC_Planner._ddp_chol_solve(L, -Qu_k)
            Vx_k = Qx_k + Qxu_k @ k_ff_k
            Vxx_raw_k = Qxx_k + Qxu_k @ K_fb_k
            Vxx_k = 0.5 * (Vxx_raw_k + Vxx_raw_k.T)
            Qu_norm_k = jnp.linalg.norm(Qu_k)
            outputs = (
                Quu_inv_k, Qxu_k, K_fb_k, k_ff_k, Fx_k, Fu_k, Qxx_bar_k,
                Qxu_bar_k, Quu_bar_k, Vx_k, Vxx_k, Qu_norm_k, chol_ok
            )
            return (Vx_k, Vxx_k), outputs

        inputs_rev = (
            Fx[::-1],
            Fu[::-1],
            Qx_bar[::-1],
            Qu_bar[::-1],
            Qxx_bar[::-1],
            Qxu_bar[::-1],
            Quu_bar[::-1],
        )
        (_, _), outs_rev = jax.lax.scan(backward_step, (Vx_N, Vxx_N), inputs_rev)
        outs = jax.tree_util.tree_map(lambda a: a[::-1], outs_rev)
        (Quu_inv, Qxu, K_fb, k_ff, Fx_out, Fu_out, Hxx, Hxu, Huu,
         Vx_body, Vxx_body, Qu_norms, chol_ok_each) = outs
        Vx = jnp.concatenate([Vx_body, Vx_N[None, :]], axis=0)
        Vxx = jnp.concatenate([Vxx_body, Vxx_N[None, :, :]], axis=0)
        Qu_2 = jnp.max(Qu_norms)
        chol_ok = jnp.all(chol_ok_each)
        return Quu_inv, Qxu, K_fb, k_ff, Fx_out, Fu_out, Hxx, Hxu, Huu, Vx, Vxx, Qu_2, chol_ok

    @staticmethod
    def _ddp_backward_pass_policy(dynamics_fn, cost_fn, term_cost_fn, xs, us, params, reg):
        Fx, Fu, Qx_bar, Qu_bar, Qxx_bar, Qxu_bar, Quu_bar, Vx_N, Vxx_N = MPC_Planner._ddp_derivatives(
            dynamics_fn,
            cost_fn,
            term_cost_fn,
            xs,
            us,
            params,
        )
        nu = us.shape[-1]
        eye_u = jnp.eye(nu, dtype=xs.dtype)

        def backward_step(carry, inputs):
            Vx_next, Vxx_next = carry
            Fx_k, Fu_k, Qx_bar_k, Qu_bar_k, Qxx_bar_k, Qxu_bar_k, Quu_bar_k = inputs
            Qx_k = Qx_bar_k + Fx_k.T @ Vx_next
            Qu_k = Qu_bar_k + Fu_k.T @ Vx_next
            Qxx_k = Qxx_bar_k + Fx_k.T @ Vxx_next @ Fx_k
            Qxu_k = Qxu_bar_k + Fx_k.T @ Vxx_next @ Fu_k
            Quu_k = Quu_bar_k + Fu_k.T @ Vxx_next @ Fu_k
            Quu_reg_k = Quu_k + reg * eye_u
            L = jax.lax.linalg.cholesky(Quu_reg_k, symmetrize_input=False)
            chol_ok = jnp.all(jnp.isfinite(L))
            K_fb_k = MPC_Planner._ddp_chol_solve(L, -Qxu_k.T)
            k_ff_k = MPC_Planner._ddp_chol_solve(L, -Qu_k)
            Vx_k = Qx_k + Qxu_k @ k_ff_k
            Vxx_raw_k = Qxx_k + Qxu_k @ K_fb_k
            Vxx_k = 0.5 * (Vxx_raw_k + Vxx_raw_k.T)
            return (Vx_k, Vxx_k), (K_fb_k, k_ff_k, jnp.linalg.norm(Qu_k), chol_ok)

        inputs_rev = (
            Fx[::-1],
            Fu[::-1],
            Qx_bar[::-1],
            Qu_bar[::-1],
            Qxx_bar[::-1],
            Qxu_bar[::-1],
            Quu_bar[::-1],
        )
        (_, _), outs_rev = jax.lax.scan(backward_step, (Vx_N, Vxx_N), inputs_rev)
        K_fb, k_ff, Qu_norms, chol_ok_each = jax.tree_util.tree_map(lambda a: a[::-1], outs_rev)
        return K_fb, k_ff, jnp.max(Qu_norms), jnp.all(chol_ok_each)

    @staticmethod
    def _ddp_exact_single(dynamics_fn, cost_fn, term_cost_fn, x0, u_initial, params):
        max_iter = 10
        e_tol = 1e-2
        reg_init = 1e-6
        reg_max = 1.0
        reg_up = 10.0
        alpha_init = 1.0
        alpha_factor = 0.5
        T, nu = u_initial.shape
        nx = x0.shape[0]
        xs0 = MPC_Planner._ddp_bad_initial_rollout(dynamics_fn, x0, u_initial, params)
        us0 = u_initial
        cost0 = MPC_Planner._ddp_total_cost(cost_fn, term_cost_fn, xs0, us0, params)
        zeros_T_nu_nx = jnp.zeros((T, nu, nx), dtype=x0.dtype)
        zeros_T_nu = jnp.zeros((T, nu), dtype=x0.dtype)
        zeros_T_nu_nu = jnp.zeros((T, nu, nu), dtype=x0.dtype)
        zeros_T_nx_nu = jnp.zeros((T, nx, nu), dtype=x0.dtype)
        zeros_T_nx_nx = jnp.zeros((T, nx, nx), dtype=x0.dtype)
        zeros_Tp1_nx = jnp.zeros((T + 1, nx), dtype=x0.dtype)
        zeros_Tp1_nx_nx = jnp.zeros((T + 1, nx, nx), dtype=x0.dtype)
        state0 = (
            jnp.array(1, dtype=jnp.int32),
            jnp.asarray(reg_init, dtype=x0.dtype),
            xs0,
            us0,
            cost0,
            xs0,
            us0,
            zeros_T_nu_nx,
            zeros_T_nu,
            zeros_T_nu_nu,
            zeros_T_nx_nu,
            zeros_T_nx_nx,
            zeros_T_nx_nu,
            zeros_T_nu_nu,
            zeros_Tp1_nx,
            zeros_Tp1_nx_nx,
            zeros_T_nx_nx,
            zeros_T_nx_nu,
            jnp.asarray(False),
        )

        alphas = jnp.asarray(
            [alpha_init, alpha_init*alpha_factor, alpha_init*alpha_factor**2,
             alpha_init*alpha_factor**3, alpha_init*alpha_factor**4],
            dtype=x0.dtype,
        )

        def body(state, _):
            (iteration, reg, xs, us, cost_prev, payload_xs_old, payload_us_old,
             K_fb_old, k_ff_old, Quu_inv_old,
             Qxu_old, Hxx_old, Hxu_old, Huu_old, Vx_old, Vxx_old, Fx_old, Fu_old,
             converged) = state

            def do_iteration(s):
                (iteration, reg, xs, us, cost_prev, payload_xs_old, payload_us_old,
                 K_fb_old, k_ff_old, Quu_inv_old,
                 Qxu_old, Hxx_old, Hxu_old, Huu_old, Vx_old, Vxx_old, Fx_old, Fu_old,
                 converged) = s
                (Quu_inv, Qxu, K_fb, k_ff, Fx, Fu, Hxx, Hxu, Huu, Vx, Vxx,
                 Qu_2, chol_ok) = MPC_Planner._ddp_backward_pass(
                    dynamics_fn, cost_fn, term_cost_fn, xs, us, params, reg
                )

                def try_alpha(carry, alpha):
                    accepted, xs_acc, us_acc, cost_acc, cost_last = carry

                    def eval_alpha(_):
                        xs_try, us_try, cost_try = MPC_Planner._ddp_forward_policy(
                            dynamics_fn, cost_fn, term_cost_fn, x0, xs, us, K_fb, k_ff, alpha, params
                        )
                        improved = cost_try < cost_prev
                        take = jnp.logical_and(jnp.logical_not(accepted), improved)
                        xs_next = jnp.where(take, xs_try, xs_acc)
                        us_next = jnp.where(take, us_try, us_acc)
                        cost_next = jnp.where(take, cost_try, cost_acc)
                        accepted_next = jnp.logical_or(accepted, improved)
                        return accepted_next, xs_next, us_next, cost_next, cost_try

                    return jax.lax.cond(
                        accepted,
                        lambda _: (accepted, xs_acc, us_acc, cost_acc, cost_last),
                        eval_alpha,
                        operand=None,
                    ), None

                init_line = (jnp.asarray(False), xs, us, cost_prev, cost_prev)
                accepted, xs_line, us_line, cost_line, cost_last = jax.lax.scan(try_alpha, init_line, alphas)[0]
                xs_next = jnp.where(accepted, xs_line, xs)
                us_next = jnp.where(accepted, us_line, us)
                cost_next = jnp.where(accepted, cost_line, cost_last)
                reg_after_line = jnp.where(accepted, reg, jnp.minimum(reg_max, reg * reg_up))
                reg_next = jnp.where(chol_ok, reg_after_line, jnp.minimum(reg_max, reg * reg_up))
                iteration_next = jnp.where(chol_ok, iteration + 1, iteration)
                converged_next = jnp.logical_and(chol_ok, Qu_2 <= e_tol)
                new_state = (
                    iteration_next,
                    reg_next,
                    jnp.where(chol_ok, xs_next, xs),
                    jnp.where(chol_ok, us_next, us),
                    jnp.where(chol_ok, cost_next, cost_prev),
                    jnp.where(chol_ok, xs, payload_xs_old),
                    jnp.where(chol_ok, us, payload_us_old),
                    jnp.where(chol_ok, K_fb, K_fb_old),
                    jnp.where(chol_ok, k_ff, k_ff_old),
                    jnp.where(chol_ok, Quu_inv, Quu_inv_old),
                    jnp.where(chol_ok, Qxu, Qxu_old),
                    jnp.where(chol_ok, Hxx, Hxx_old),
                    jnp.where(chol_ok, Hxu, Hxu_old),
                    jnp.where(chol_ok, Huu, Huu_old),
                    jnp.where(chol_ok, Vx, Vx_old),
                    jnp.where(chol_ok, Vxx, Vxx_old),
                    jnp.where(chol_ok, Fx, Fx_old),
                    jnp.where(chol_ok, Fu, Fu_old),
                    converged_next,
                )
                ratio = jnp.abs(cost_next - cost_prev) / jnp.abs(cost_prev)
                hist = (chol_ok, ratio, Qu_2)
                return new_state, hist

            should_skip = jnp.logical_or(converged, iteration > max_iter)
            return jax.lax.cond(
                should_skip,
                lambda s: (s, (jnp.asarray(False), jnp.asarray(0.0, dtype=x0.dtype), jnp.asarray(0.0, dtype=x0.dtype))),
                do_iteration,
                state,
            )

        final_state, hist = jax.lax.scan(body, state0, None, length=max_iter)
        (_, _, xs, us, _, payload_xs, payload_us,
         K_fb, _, Quu_inv, Qxu, Hxx, Hxu, Huu, Vx, Vxx, Fx, Fu, _) = final_state
        hist_active, hist_ratio, hist_qu = hist
        return xs, us, payload_xs, payload_us, Vxx, Vx, K_fb, Hxx, Qxu, Hxu, Huu, Quu_inv, Fx, Fu, hist_active, hist_ratio, hist_qu

    @staticmethod
    def _ddp_load_exact_single(x0, u_initial, params):
        return MPC_Planner._ddp_exact_single(
            MPC_Planner.jax_load_dynamics,
            MPC_Planner.jax_load_stage_cost,
            MPC_Planner.jax_load_terminal_cost,
            x0,
            u_initial,
            params,
        )

    @staticmethod
    def _ddp_load_exact_batched(x0_b, u_initial_b, params_b):
        return jax.vmap(MPC_Planner._ddp_load_exact_single)(x0_b, u_initial_b, params_b)

    @staticmethod
    def _ddp_cable_exact_single(x0, u_initial, params):
        return MPC_Planner._ddp_exact_single(
            MPC_Planner.jax_cable_dynamics_single,
            MPC_Planner.jax_cable_stage_cost,
            MPC_Planner.jax_cable_terminal_cost,
            x0,
            u_initial,
            params,
        )

    @staticmethod
    def _ddp_cable_exact_batched(x0_b, u_initial_b, params_b):
        return jax.vmap(MPC_Planner._ddp_cable_exact_single)(x0_b, u_initial_b, params_b)

    @staticmethod
    def _ddp_exact_single_traj(dynamics_fn, cost_fn, term_cost_fn, x0, u_initial, params):
        max_iter = 10
        e_tol = 1e-2
        reg_init = 1e-6
        reg_max = 1.0
        reg_up = 10.0
        alpha_init = 1.0
        alpha_factor = 0.5
        xs0 = MPC_Planner._ddp_bad_initial_rollout(dynamics_fn, x0, u_initial, params)
        us0 = u_initial
        cost0 = MPC_Planner._ddp_total_cost(cost_fn, term_cost_fn, xs0, us0, params)
        state0 = (
            jnp.array(1, dtype=jnp.int32),
            jnp.asarray(reg_init, dtype=x0.dtype),
            xs0,
            us0,
            cost0,
            jnp.asarray(False),
        )
        alphas = jnp.asarray(
            [alpha_init, alpha_init*alpha_factor, alpha_init*alpha_factor**2,
             alpha_init*alpha_factor**3, alpha_init*alpha_factor**4],
            dtype=x0.dtype,
        )

        def body(state, _):
            iteration, reg, xs, us, cost_prev, converged = state

            def do_iteration(s):
                iteration, reg, xs, us, cost_prev, converged = s
                K_fb, k_ff, Qu_2, chol_ok = MPC_Planner._ddp_backward_pass_policy(
                    dynamics_fn, cost_fn, term_cost_fn, xs, us, params, reg
                )

                def try_alpha(carry, alpha):
                    accepted, xs_acc, us_acc, cost_acc, cost_last = carry

                    def eval_alpha(_):
                        xs_try, us_try, cost_try = MPC_Planner._ddp_forward_policy(
                            dynamics_fn, cost_fn, term_cost_fn, x0, xs, us, K_fb, k_ff, alpha, params
                        )
                        improved = cost_try < cost_prev
                        take = jnp.logical_and(jnp.logical_not(accepted), improved)
                        xs_next = jnp.where(take, xs_try, xs_acc)
                        us_next = jnp.where(take, us_try, us_acc)
                        cost_next = jnp.where(take, cost_try, cost_acc)
                        accepted_next = jnp.logical_or(accepted, improved)
                        return accepted_next, xs_next, us_next, cost_next, cost_try

                    return jax.lax.cond(
                        accepted,
                        lambda _: (accepted, xs_acc, us_acc, cost_acc, cost_last),
                        eval_alpha,
                        operand=None,
                    ), None

                init_line = (jnp.asarray(False), xs, us, cost_prev, cost_prev)
                accepted, xs_line, us_line, cost_line, cost_last = jax.lax.scan(try_alpha, init_line, alphas)[0]
                xs_next = jnp.where(accepted, xs_line, xs)
                us_next = jnp.where(accepted, us_line, us)
                cost_next = jnp.where(accepted, cost_line, cost_last)
                reg_after_line = jnp.where(accepted, reg, jnp.minimum(reg_max, reg * reg_up))
                reg_next = jnp.where(chol_ok, reg_after_line, jnp.minimum(reg_max, reg * reg_up))
                iteration_next = jnp.where(chol_ok, iteration + 1, iteration)
                converged_next = jnp.logical_and(chol_ok, Qu_2 <= e_tol)
                new_state = (
                    iteration_next,
                    reg_next,
                    jnp.where(chol_ok, xs_next, xs),
                    jnp.where(chol_ok, us_next, us),
                    jnp.where(chol_ok, cost_next, cost_prev),
                    converged_next,
                )
                return new_state, None

            should_skip = jnp.logical_or(converged, iteration > max_iter)
            return jax.lax.cond(
                should_skip,
                lambda s: (s, None),
                do_iteration,
                state,
            )

        final_state, _ = jax.lax.scan(body, state0, None, length=max_iter)
        _, _, xs, us, _, _ = final_state
        return xs, us

    @staticmethod
    def _ddp_exact_single_policy(dynamics_fn, cost_fn, term_cost_fn, x0, u_initial, params):
        max_iter = 10
        e_tol = 1e-2
        reg_init = 1e-6
        reg_max = 1.0
        reg_up = 10.0
        alpha_init = 1.0
        alpha_factor = 0.5
        T, nu = u_initial.shape
        nx = x0.shape[0]
        xs0 = MPC_Planner._ddp_bad_initial_rollout(dynamics_fn, x0, u_initial, params)
        us0 = u_initial
        cost0 = MPC_Planner._ddp_total_cost(cost_fn, term_cost_fn, xs0, us0, params)
        zeros_K = jnp.zeros((T, nu, nx), dtype=x0.dtype)
        state0 = (
            jnp.array(1, dtype=jnp.int32),
            jnp.asarray(reg_init, dtype=x0.dtype),
            xs0,
            us0,
            cost0,
            zeros_K,
            jnp.asarray(False),
        )
        alphas = jnp.asarray(
            [alpha_init, alpha_init*alpha_factor, alpha_init*alpha_factor**2,
             alpha_init*alpha_factor**3, alpha_init*alpha_factor**4],
            dtype=x0.dtype,
        )

        def body(state, _):
            iteration, reg, xs, us, cost_prev, K_old, converged = state

            def do_iteration(s):
                iteration, reg, xs, us, cost_prev, K_old, converged = s
                K_fb, k_ff, Qu_2, chol_ok = MPC_Planner._ddp_backward_pass_policy(
                    dynamics_fn, cost_fn, term_cost_fn, xs, us, params, reg
                )

                def try_alpha(carry, alpha):
                    accepted, xs_acc, us_acc, cost_acc, cost_last = carry

                    def eval_alpha(_):
                        xs_try, us_try, cost_try = MPC_Planner._ddp_forward_policy(
                            dynamics_fn, cost_fn, term_cost_fn, x0, xs, us, K_fb, k_ff, alpha, params
                        )
                        improved = cost_try < cost_prev
                        take = jnp.logical_and(jnp.logical_not(accepted), improved)
                        xs_next = jnp.where(take, xs_try, xs_acc)
                        us_next = jnp.where(take, us_try, us_acc)
                        cost_next = jnp.where(take, cost_try, cost_acc)
                        accepted_next = jnp.logical_or(accepted, improved)
                        return accepted_next, xs_next, us_next, cost_next, cost_try

                    return jax.lax.cond(
                        accepted,
                        lambda _: (accepted, xs_acc, us_acc, cost_acc, cost_last),
                        eval_alpha,
                        operand=None,
                    ), None

                init_line = (jnp.asarray(False), xs, us, cost_prev, cost_prev)
                accepted, xs_line, us_line, cost_line, cost_last = jax.lax.scan(try_alpha, init_line, alphas)[0]
                xs_next = jnp.where(accepted, xs_line, xs)
                us_next = jnp.where(accepted, us_line, us)
                cost_next = jnp.where(accepted, cost_line, cost_last)
                reg_after_line = jnp.where(accepted, reg, jnp.minimum(reg_max, reg * reg_up))
                reg_next = jnp.where(chol_ok, reg_after_line, jnp.minimum(reg_max, reg * reg_up))
                iteration_next = jnp.where(chol_ok, iteration + 1, iteration)
                converged_next = jnp.logical_and(chol_ok, Qu_2 <= e_tol)
                new_state = (
                    iteration_next,
                    reg_next,
                    jnp.where(chol_ok, xs_next, xs),
                    jnp.where(chol_ok, us_next, us),
                    jnp.where(chol_ok, cost_next, cost_prev),
                    jnp.where(chol_ok, K_fb, K_old),
                    converged_next,
                )
                return new_state, None

            should_skip = jnp.logical_or(converged, iteration > max_iter)
            return jax.lax.cond(
                should_skip,
                lambda s: (s, None),
                do_iteration,
                state,
            )

        final_state, _ = jax.lax.scan(body, state0, None, length=max_iter)
        _, _, xs, us, _, K_fb, _ = final_state
        return xs, us, K_fb

    @staticmethod
    def _ddp_load_policy_single(x0, u_initial, params):
        return MPC_Planner._ddp_exact_single_policy(
            MPC_Planner.jax_load_dynamics,
            MPC_Planner.jax_load_stage_cost,
            MPC_Planner.jax_load_terminal_cost,
            x0,
            u_initial,
            params,
        )

    @staticmethod
    def _ddp_load_fast_bad_initial_rollout(x0, us, params):
        zero_x = jnp.zeros_like(x0)
        static = params["static"]

        def one_step(t, u):
            seed_x = jnp.where(t == 0, x0, zero_x)
            return MPC_Planner.jax_load_dynamics(seed_x, u, static)

        xs_body = jax.vmap(one_step)(jnp.arange(us.shape[0]), us)
        return jnp.concatenate([xs_body, zero_x[None, :]], axis=0)

    @staticmethod
    def _ddp_load_fast_total_cost(xs, us, params):
        static = params["static"]
        stage = params["stage"]
        terminal = params["terminal"]
        dx = xs[:-1] - stage["ref_x"]
        du = us - stage["ref_u"]
        rx = xs[:-1] - stage["scx"] + stage["y_x"] / static["rho_lx"]
        ru = us - stage["scu"] + stage["y_u"] / static["rho_lu"]
        cost = 0.5 * jnp.sum(dx * dx * static["Q_weight"][None, :])
        cost += 0.5 * jnp.sum(du * du * static["R_weight"][None, :])
        cost += 0.5 * static["rho_lx"] * jnp.sum(rx * rx)
        cost += 0.5 * static["rho_lu"] * jnp.sum(ru * ru)
        dx_T = xs[-1] - terminal["ref_x"]
        rx_T = xs[-1] - terminal["scx"] + terminal["y_x"] / static["rho_lx"]
        cost += 0.5 * jnp.sum(dx_T * dx_T * static["Q_terminal_weight"])
        cost += 0.5 * static["rho_lx"] * jnp.sum(rx_T * rx_T)
        return cost

    @staticmethod
    def _ddp_load_fast_backward_policy(xs, us, params, reg):
        static = params["static"]
        stage = params["stage"]
        terminal = params["terminal"]
        dyn_x = jax.jacfwd(MPC_Planner.jax_load_dynamics, 0)
        dyn_u = jax.jacfwd(MPC_Planner.jax_load_dynamics, 1)
        q = static["Q_weight"]
        r = static["R_weight"]
        qn = static["Q_terminal_weight"]
        rho_x = static["rho_lx"]
        rho_u = static["rho_lu"]
        nx = xs.shape[-1]
        nu = us.shape[-1]

        def one_stage(x, u, ref_x, ref_u, scx, scu, y_x, y_u):
            Fx = dyn_x(x, u, static)
            Fu = dyn_u(x, u, static)
            lx = q * (x - ref_x) + rho_x * (x - scx + y_x / rho_x)
            lu = r * (u - ref_u) + rho_u * (u - scu + y_u / rho_u)
            Hxx = jnp.diag(q) + rho_x * jnp.eye(nx, dtype=x.dtype)
            Huu = jnp.diag(r) + rho_u * jnp.eye(nu, dtype=x.dtype)
            Hxu = jnp.zeros((nx, nu), dtype=x.dtype)
            return Fx, Fu, lx, lu, Hxx, Hxu, Huu

        Fx, Fu, lx, lu, Hxx, Hxu, Huu = jax.vmap(one_stage)(
            xs[:-1], us, stage["ref_x"], stage["ref_u"], stage["scx"], stage["scu"],
            stage["y_x"], stage["y_u"]
        )
        Vx_N = qn * (xs[-1] - terminal["ref_x"]) + rho_x * (xs[-1] - terminal["scx"] + terminal["y_x"] / rho_x)
        Vxx_N = jnp.diag(qn) + rho_x * jnp.eye(nx, dtype=xs.dtype)
        eye_u = jnp.eye(nu, dtype=xs.dtype)

        def backward_step(carry, inputs):
            Vx_next, Vxx_next = carry
            Fx_k, Fu_k, lx_k, lu_k, Hxx_k, Hxu_k, Huu_k = inputs
            Qx_k = lx_k + Fx_k.T @ Vx_next
            Qu_k = lu_k + Fu_k.T @ Vx_next
            Qxx_k = Hxx_k + Fx_k.T @ Vxx_next @ Fx_k
            Qxu_k = Hxu_k + Fx_k.T @ Vxx_next @ Fu_k
            Quu_k = Huu_k + Fu_k.T @ Vxx_next @ Fu_k
            L = jax.lax.linalg.cholesky(Quu_k + reg * eye_u, symmetrize_input=False)
            chol_ok = jnp.all(jnp.isfinite(L))
            K_fb_k = MPC_Planner._ddp_chol_solve(L, -Qxu_k.T)
            k_ff_k = MPC_Planner._ddp_chol_solve(L, -Qu_k)
            Vx_k = Qx_k + Qxu_k @ k_ff_k
            Vxx_raw_k = Qxx_k + Qxu_k @ K_fb_k
            Vxx_k = 0.5 * (Vxx_raw_k + Vxx_raw_k.T)
            return (Vx_k, Vxx_k), (K_fb_k, k_ff_k, jnp.linalg.norm(Qu_k), chol_ok)

        inputs_rev = (Fx[::-1], Fu[::-1], lx[::-1], lu[::-1], Hxx[::-1], Hxu[::-1], Huu[::-1])
        (_, _), outs_rev = jax.lax.scan(backward_step, (Vx_N, Vxx_N), inputs_rev)
        K_fb, k_ff, Qu_norms, chol_ok_each = jax.tree_util.tree_map(lambda a: a[::-1], outs_rev)
        return K_fb, k_ff, jnp.max(Qu_norms), jnp.all(chol_ok_each)

    @staticmethod
    def _ddp_load_fast_forward_policy(x0, xs_nominal, us_nominal, K_fb, k_ff, alpha, params):
        static = params["static"]

        def step(x_k, k):
            delta_x = x_k - xs_nominal[k]
            u_k = us_nominal[k] + K_fb[k] @ delta_x + alpha * k_ff[k]
            x_next = MPC_Planner.jax_load_dynamics(x_k, u_k, static)
            return x_next, (x_k, u_k)

        x_terminal, hist = jax.lax.scan(step, x0, jnp.arange(us_nominal.shape[0]))
        xs_new = jnp.concatenate([hist[0], x_terminal[None, :]], axis=0)
        us_new = hist[1]
        cost_new = MPC_Planner._ddp_load_fast_total_cost(xs_new, us_new, params)
        return xs_new, us_new, cost_new

    @staticmethod
    def _ddp_load_traj_single(x0, u_initial, params):
        max_iter = 10
        e_tol = 1e-2
        reg_init = 1e-6
        reg_max = 1.0
        reg_up = 10.0
        alphas = jnp.asarray([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=x0.dtype)
        xs0 = MPC_Planner._ddp_load_fast_bad_initial_rollout(x0, u_initial, params)
        cost0 = MPC_Planner._ddp_load_fast_total_cost(xs0, u_initial, params)
        state0 = (
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(reg_init, dtype=x0.dtype),
            xs0,
            u_initial,
            cost0,
            jnp.asarray(False),
        )

        def body(state, _):
            iteration, reg, xs, us, cost_prev, converged = state

            def do_iteration(s):
                iteration, reg, xs, us, cost_prev, converged = s
                K_fb, k_ff, Qu_2, chol_ok = MPC_Planner._ddp_load_fast_backward_policy(xs, us, params, reg)

                def try_alpha(carry, alpha):
                    accepted, xs_acc, us_acc, cost_acc, cost_last = carry

                    def eval_alpha(_):
                        xs_try, us_try, cost_try = MPC_Planner._ddp_load_fast_forward_policy(
                            x0, xs, us, K_fb, k_ff, alpha, params
                        )
                        improved = cost_try < cost_prev
                        take = jnp.logical_and(jnp.logical_not(accepted), improved)
                        return (
                            jnp.logical_or(accepted, improved),
                            jnp.where(take, xs_try, xs_acc),
                            jnp.where(take, us_try, us_acc),
                            jnp.where(take, cost_try, cost_acc),
                            cost_try,
                        )

                    return jax.lax.cond(
                        accepted,
                        lambda _: (accepted, xs_acc, us_acc, cost_acc, cost_last),
                        eval_alpha,
                        operand=None,
                    ), None

                accepted, xs_line, us_line, cost_line, cost_last = jax.lax.scan(
                    try_alpha, (jnp.asarray(False), xs, us, cost_prev, cost_prev), alphas
                )[0]
                xs_next = jnp.where(accepted, xs_line, xs)
                us_next = jnp.where(accepted, us_line, us)
                cost_next = jnp.where(accepted, cost_line, cost_last)
                reg_after_line = jnp.where(accepted, reg, jnp.minimum(reg_max, reg * reg_up))
                reg_next = jnp.where(chol_ok, reg_after_line, jnp.minimum(reg_max, reg * reg_up))
                iteration_next = jnp.where(chol_ok, iteration + 1, iteration)
                converged_next = jnp.logical_and(chol_ok, Qu_2 <= e_tol)
                new_state = (
                    iteration_next,
                    reg_next,
                    jnp.where(chol_ok, xs_next, xs),
                    jnp.where(chol_ok, us_next, us),
                    jnp.where(chol_ok, cost_next, cost_prev),
                    converged_next,
                )
                return new_state, None

            should_skip = jnp.logical_or(converged, iteration > max_iter)
            return jax.lax.cond(
                should_skip,
                lambda s: (s, None),
                do_iteration,
                state,
            )

        final_state, _ = jax.lax.scan(body, state0, None, length=max_iter)
        _, _, xs, us, _, _ = final_state
        return xs, us

    @staticmethod
    def _ddp_cable_fast_bad_initial_rollout(x0, us, params):
        zero_x = jnp.zeros_like(x0)
        static = params["static"]

        def one_step(t, u):
            seed_x = jnp.where(t == 0, x0, zero_x)
            return MPC_Planner.jax_cable_dynamics_single(seed_x, u, static)

        xs_body = jax.vmap(one_step)(jnp.arange(us.shape[0]), us)
        return jnp.concatenate([xs_body, zero_x[None, :]], axis=0)

    @staticmethod
    def _ddp_cable_fast_total_cost(xs, us, params):
        static = params["static"]
        stage = params["stage"]
        terminal = params["terminal"]
        dx = xs[:-1] - stage["ref_x_i"]
        du = us - stage["ref_u_i"]
        rx = xs[:-1] - stage["scx_i"] + stage["y_x_i"] / static["rho_ix"]
        ru = us - stage["scu_i"] + stage["y_u_i"] / static["rho_iu"]
        cost = 0.5 * jnp.sum(dx * dx * static["Qi_weight"][None, :])
        cost += 0.5 * jnp.sum(du * du * static["Ri_weight"][None, :])
        cost += 0.5 * static["rho_ix"] * jnp.sum(rx * rx)
        cost += 0.5 * static["rho_iu"] * jnp.sum(ru * ru)
        dx_T = xs[-1] - terminal["ref_x_i"]
        rx_T = xs[-1] - terminal["scx_i"] + terminal["y_x_i"] / static["rho_ix"]
        cost += 0.5 * jnp.sum(dx_T * dx_T * static["Qi_terminal_weight"])
        cost += 0.5 * static["rho_ix"] * jnp.sum(rx_T * rx_T)
        return cost

    @staticmethod
    def _ddp_cable_fast_backward_policy(xs, us, params, reg):
        static = params["static"]
        stage = params["stage"]
        terminal = params["terminal"]
        dyn_x = jax.jacfwd(MPC_Planner.jax_cable_dynamics_single, 0)
        dyn_u = jax.jacfwd(MPC_Planner.jax_cable_dynamics_single, 1)
        q = static["Qi_weight"]
        r = static["Ri_weight"]
        qn = static["Qi_terminal_weight"]
        rho_x = static["rho_ix"]
        rho_u = static["rho_iu"]
        nx = xs.shape[-1]
        nu = us.shape[-1]

        def one_stage(x, u, ref_x, ref_u, scx, scu, y_x, y_u):
            Fx = dyn_x(x, u, static)
            Fu = dyn_u(x, u, static)
            lx = q * (x - ref_x) + rho_x * (x - scx + y_x / rho_x)
            lu = r * (u - ref_u) + rho_u * (u - scu + y_u / rho_u)
            Hxx = jnp.diag(q) + rho_x * jnp.eye(nx, dtype=x.dtype)
            Huu = jnp.diag(r) + rho_u * jnp.eye(nu, dtype=x.dtype)
            Hxu = jnp.zeros((nx, nu), dtype=x.dtype)
            return Fx, Fu, lx, lu, Hxx, Hxu, Huu

        Fx, Fu, lx, lu, Hxx, Hxu, Huu = jax.vmap(one_stage)(
            xs[:-1], us, stage["ref_x_i"], stage["ref_u_i"], stage["scx_i"], stage["scu_i"],
            stage["y_x_i"], stage["y_u_i"]
        )
        Vx_N = qn * (xs[-1] - terminal["ref_x_i"]) + rho_x * (xs[-1] - terminal["scx_i"] + terminal["y_x_i"] / rho_x)
        Vxx_N = jnp.diag(qn) + rho_x * jnp.eye(nx, dtype=xs.dtype)
        eye_u = jnp.eye(nu, dtype=xs.dtype)

        def backward_step(carry, inputs):
            Vx_next, Vxx_next = carry
            Fx_k, Fu_k, lx_k, lu_k, Hxx_k, Hxu_k, Huu_k = inputs
            Qx_k = lx_k + Fx_k.T @ Vx_next
            Qu_k = lu_k + Fu_k.T @ Vx_next
            Qxx_k = Hxx_k + Fx_k.T @ Vxx_next @ Fx_k
            Qxu_k = Hxu_k + Fx_k.T @ Vxx_next @ Fu_k
            Quu_k = Huu_k + Fu_k.T @ Vxx_next @ Fu_k
            L = jax.lax.linalg.cholesky(Quu_k + reg * eye_u, symmetrize_input=False)
            chol_ok = jnp.all(jnp.isfinite(L))
            K_fb_k = MPC_Planner._ddp_chol_solve(L, -Qxu_k.T)
            k_ff_k = MPC_Planner._ddp_chol_solve(L, -Qu_k)
            Vx_k = Qx_k + Qxu_k @ k_ff_k
            Vxx_raw_k = Qxx_k + Qxu_k @ K_fb_k
            Vxx_k = 0.5 * (Vxx_raw_k + Vxx_raw_k.T)
            return (Vx_k, Vxx_k), (K_fb_k, k_ff_k, jnp.linalg.norm(Qu_k), chol_ok)

        inputs_rev = (Fx[::-1], Fu[::-1], lx[::-1], lu[::-1], Hxx[::-1], Hxu[::-1], Huu[::-1])
        (_, _), outs_rev = jax.lax.scan(backward_step, (Vx_N, Vxx_N), inputs_rev)
        K_fb, k_ff, Qu_norms, chol_ok_each = jax.tree_util.tree_map(lambda a: a[::-1], outs_rev)
        return K_fb, k_ff, jnp.max(Qu_norms), jnp.all(chol_ok_each)

    @staticmethod
    def _ddp_cable_fast_forward_policy(x0, xs_nominal, us_nominal, K_fb, k_ff, alpha, params):
        static = params["static"]

        def step(x_k, k):
            delta_x = x_k - xs_nominal[k]
            u_k = us_nominal[k] + K_fb[k] @ delta_x + alpha * k_ff[k]
            x_next = MPC_Planner.jax_cable_dynamics_single(x_k, u_k, static)
            return x_next, (x_k, u_k)

        x_terminal, hist = jax.lax.scan(step, x0, jnp.arange(us_nominal.shape[0]))
        xs_new = jnp.concatenate([hist[0], x_terminal[None, :]], axis=0)
        us_new = hist[1]
        cost_new = MPC_Planner._ddp_cable_fast_total_cost(xs_new, us_new, params)
        return xs_new, us_new, cost_new

    @staticmethod
    def _ddp_cable_traj_single(x0, u_initial, params):
        max_iter = 10
        e_tol = 1e-2
        reg_init = 1e-6
        reg_max = 1.0
        reg_up = 10.0
        alphas = jnp.asarray([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=x0.dtype)
        xs0 = MPC_Planner._ddp_cable_fast_bad_initial_rollout(x0, u_initial, params)
        cost0 = MPC_Planner._ddp_cable_fast_total_cost(xs0, u_initial, params)
        state0 = (
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(reg_init, dtype=x0.dtype),
            xs0,
            u_initial,
            cost0,
            jnp.asarray(False),
        )

        def body(state, _):
            iteration, reg, xs, us, cost_prev, converged = state

            def do_iteration(s):
                iteration, reg, xs, us, cost_prev, converged = s
                K_fb, k_ff, Qu_2, chol_ok = MPC_Planner._ddp_cable_fast_backward_policy(xs, us, params, reg)

                def try_alpha(carry, alpha):
                    accepted, xs_acc, us_acc, cost_acc, cost_last = carry

                    def eval_alpha(_):
                        xs_try, us_try, cost_try = MPC_Planner._ddp_cable_fast_forward_policy(
                            x0, xs, us, K_fb, k_ff, alpha, params
                        )
                        improved = cost_try < cost_prev
                        take = jnp.logical_and(jnp.logical_not(accepted), improved)
                        return (
                            jnp.logical_or(accepted, improved),
                            jnp.where(take, xs_try, xs_acc),
                            jnp.where(take, us_try, us_acc),
                            jnp.where(take, cost_try, cost_acc),
                            cost_try,
                        )

                    return jax.lax.cond(
                        accepted,
                        lambda _: (accepted, xs_acc, us_acc, cost_acc, cost_last),
                        eval_alpha,
                        operand=None,
                    ), None

                accepted, xs_line, us_line, cost_line, cost_last = jax.lax.scan(
                    try_alpha, (jnp.asarray(False), xs, us, cost_prev, cost_prev), alphas
                )[0]
                xs_next = jnp.where(accepted, xs_line, xs)
                us_next = jnp.where(accepted, us_line, us)
                cost_next = jnp.where(accepted, cost_line, cost_last)
                reg_after_line = jnp.where(accepted, reg, jnp.minimum(reg_max, reg * reg_up))
                reg_next = jnp.where(chol_ok, reg_after_line, jnp.minimum(reg_max, reg * reg_up))
                iteration_next = jnp.where(chol_ok, iteration + 1, iteration)
                converged_next = jnp.logical_and(chol_ok, Qu_2 <= e_tol)
                new_state = (
                    iteration_next,
                    reg_next,
                    jnp.where(chol_ok, xs_next, xs),
                    jnp.where(chol_ok, us_next, us),
                    jnp.where(chol_ok, cost_next, cost_prev),
                    converged_next,
                )
                return new_state, None

            should_skip = jnp.logical_or(converged, iteration > max_iter)
            return jax.lax.cond(
                should_skip,
                lambda s: (s, None),
                do_iteration,
                state,
            )

        final_state, _ = jax.lax.scan(body, state0, None, length=max_iter)
        _, _, xs, us, _, _ = final_state
        return xs, us

    @staticmethod
    def _ddp_cable_traj_batched(x0_b, u_initial_b, params_b):
        return jax.vmap(MPC_Planner._ddp_cable_traj_single)(x0_b, u_initial_b, params_b)

    @staticmethod
    def _legacy_derivs_batch(dynamics_fn, cost_fn, term_cost_fn, xs, us, params_b, reg_eps=1e-6):
        def _stage_params(params, t):
            merged = dict(params["static"])
            merged.update(jax.tree_util.tree_map(lambda a: a[t], params["stage"]))
            return merged

        def _terminal_params(params):
            merged = dict(params["static"])
            merged.update(params["terminal"])
            return merged

        def _single(xs_seq, us_seq, p_single):
            T = us_seq.shape[0]
            nu = us_seq.shape[1]

            def _step_derivs(t, x, u):
                p_t = _stage_params(p_single, t)
                fx = jax.jacfwd(dynamics_fn, 0)(x, u, p_t)
                fu = jax.jacfwd(dynamics_fn, 1)(x, u, p_t)
                lx = jax.grad(cost_fn, 0)(x, u, p_t)
                lu = jax.grad(cost_fn, 1)(x, u, p_t)
                hxx = jax.hessian(cost_fn, 0)(x, u, p_t)
                huu = jax.hessian(cost_fn, 1)(x, u, p_t)
                hxu = jax.jacfwd(jax.grad(cost_fn, 0), 1)(x, u, p_t)
                return fx, fu, lx, lu, hxx, hxu, huu

            fx, fu, lx, lu, hxx, hxu, huu = jax.vmap(_step_derivs)(jnp.arange(T), xs_seq[:-1], us_seq)
            p_T = _terminal_params(p_single)
            Vx_T = jax.grad(term_cost_fn, 0)(xs_seq[-1], p_T)
            Vxx_T = jax.hessian(term_cost_fn, 0)(xs_seq[-1], p_T)

            def _backward(carry, inputs):
                Vx_next, Vxx_next = carry
                fx_t, fu_t, lx_t, lu_t, hxx_t, hxu_t, huu_t = inputs
                Qx = lx_t + fx_t.T @ Vx_next
                Qu = lu_t + fu_t.T @ Vx_next
                Qxu = hxu_t + fx_t.T @ Vxx_next @ fu_t
                Quu = huu_t + fu_t.T @ Vxx_next @ fu_t
                Quu_reg = Quu + reg_eps * jnp.eye(nu, dtype=Quu.dtype)
                Quuinv = jnp.linalg.inv(Quu_reg)
                K = -Quuinv @ Qxu.T
                kff = -Quuinv @ Qu
                Qxx = hxx_t + fx_t.T @ Vxx_next @ fx_t
                Vx = Qx + Qxu @ kff
                Vxx = 0.5 * (Qxx + Qxu @ K + (Qxx + Qxu @ K).T)
                return (Vx, Vxx), (Quuinv, Qxu, K, fx_t, fu_t, hxx_t, hxu_t, huu_t, Vx, Vxx)

            inputs = (fx[::-1], fu[::-1], lx[::-1], lu[::-1], hxx[::-1], hxu[::-1], huu[::-1])
            (_, _), outs_rev = jax.lax.scan(_backward, (Vx_T, Vxx_T), inputs)
            Quuinv, Qxu, K, Fx, Fu, Hxx, Hxu, Huu, Vx_body, Vxx_body = jax.tree_util.tree_map(lambda a: a[::-1], outs_rev)
            Vx_full = jnp.concatenate([Vx_body, Vx_T[None, :]], axis=0)
            Vxx_full = jnp.concatenate([Vxx_body, Vxx_T[None, :, :]], axis=0)
            return Quuinv, Qxu, K, Fx, Fu, Hxx, Hxu, Huu, Vx_full, Vxx_full

        return jax.vmap(_single, in_axes=(0, 0, 0))(xs, us, params_b)

    @staticmethod
    def _static_load_derivs(xs, us, params_b):
        return MPC_Planner._legacy_derivs_batch(MPC_Planner.jax_load_dynamics, MPC_Planner.jax_load_stage_cost, MPC_Planner.jax_load_terminal_cost, xs, us, params_b)

    @staticmethod
    def _static_cable_derivs(xs, us, params_b):
        return MPC_Planner._legacy_derivs_batch(MPC_Planner.jax_cable_dynamics_single, MPC_Planner.jax_cable_stage_cost, MPC_Planner.jax_cable_terminal_cost, xs, us, params_b)

    def jax_MPC_Load_DDP_Planning_SubP1(self, ParaL, need_derivs=False):
        if os.environ.get("DIFFCOORD_SUBP1_EXACT_JAX", "1") == "1":
            N, nx, nu = int(self.N), int(self.nxl), int(self.nul)
            x0_list, u_guess_list, params_list = [], [], []
            for i in range(len(ParaL)):
                Parai = np.asarray(ParaL[i]).reshape(-1)
                idx = 0
                xl_fb = Parai[idx:idx+nx]; idx += nx
                Ref_xl_raw = Parai[idx:idx+nx*(N+1)]; idx += nx*(N+1)
                Ref_ul_raw = Parai[idx:idx+nu*N]; idx += nu*N
                scxl_raw = Parai[idx:idx+nx*(N+1)]; idx += nx*(N+1)
                scxL_raw = Parai[idx:idx+nx*(N+1)]; idx += nx*(N+1)
                scul_raw = Parai[idx:idx+nu*N]; idx += nu*N
                scuL_raw = Parai[idx:idx+nu*N]; idx += nu*N
                weight_para = Parai[idx:-1]
                i_admm = Parai[-1]
                rho_lx = float(self.open_loop_penalty(float(weight_para[-4]), float(weight_para[-2]), float(i_admm), self.max_iter_ADMM))
                rho_lu = float(self.open_loop_penalty(float(weight_para[-3]), float(weight_para[-1]), float(i_admm), self.max_iter_ADMM))
                x0_list.append(jnp.asarray(xl_fb, dtype=jnp.float64))
                u_guess_list.append(jnp.asarray(Ref_ul_raw, dtype=jnp.float64).reshape(N, nu))
                ref_x_traj = jnp.asarray(Ref_xl_raw, dtype=jnp.float64).reshape(N + 1, nx)
                ref_u_traj = jnp.asarray(Ref_ul_raw, dtype=jnp.float64).reshape(N, nu)
                scx_traj = jnp.asarray(scxl_raw, dtype=jnp.float64).reshape(N + 1, nx)
                yx_traj = jnp.asarray(scxL_raw, dtype=jnp.float64).reshape(N + 1, nx)
                scu_traj = jnp.asarray(scul_raw, dtype=jnp.float64).reshape(N, nu)
                yu_traj = jnp.asarray(scuL_raw, dtype=jnp.float64).reshape(N, nu)
                weight_j = jnp.asarray(weight_para, dtype=jnp.float64)
                params_list.append({
                    "static": {
                        "ml": float(self.ml),
                        "Jl": jnp.asarray(self.Jl, dtype=jnp.float64),
                        "Jl_inv": jnp.asarray(self.Jl_inv, dtype=jnp.float64),
                        "dt": float(self.dt),
                        "rho_lx": rho_lx,
                        "rho_lu": rho_lu,
                        "Q_weight": weight_j[0:nx],
                        "R_weight": weight_j[2*nx:2*nx+nu],
                        "Q_terminal_weight": weight_j[nx:2*nx],
                    },
                    "stage": {
                        "ref_x": ref_x_traj[:-1],
                        "ref_u": ref_u_traj,
                        "scx": scx_traj[:-1],
                        "scu": scu_traj,
                        "y_x": yx_traj[:-1],
                        "y_u": yu_traj,
                    },
                    "terminal": {
                        "ref_x": ref_x_traj[-1],
                        "scx": scx_traj[-1],
                        "y_x": yx_traj[-1],
                    },
                })
            params_b = jax.tree_util.tree_map(lambda *args: jnp.stack(args).astype(jnp.float64), *params_list)
            if (
                not need_derivs
                and len(x0_list) == 1
                and os.environ.get("DIFFCOORD_KINODYN_FORWARD_LOAD_KFB", "0") == "1"
                and getattr(self, "_load_policy_solver_single_jit", None) is not None
            ):
                params_single = jax.tree_util.tree_map(lambda a: a[0], params_b)
                xs_j, us_j, K_j = self._load_policy_solver_single_jit(
                    x0_list[0].astype(jnp.float64),
                    u_guess_list[0].astype(jnp.float64),
                    params_single,
                )
                xs_np = np.asarray(xs_j, dtype=np.float64)
                us_np = np.asarray(us_j, dtype=np.float64)
                K_np = np.asarray(K_j, dtype=np.float64)
                opt_sol_l = {
                    "xl_traj": xs_np,
                    "ul_traj": us_np,
                    "K_FB": [K_np[k] for k in range(N)],
                }
                opt_soll = {"xl_traj": [xs_np], "ul_traj": [us_np]}
                return opt_soll, [opt_sol_l]
            if (
                not need_derivs
                and len(x0_list) == 1
                and getattr(self, "_load_traj_solver_single_jit", None) is not None
            ):
                params_single = jax.tree_util.tree_map(lambda a: a[0], params_b)
                xs_j, us_j = self._load_traj_solver_single_jit(
                    x0_list[0].astype(jnp.float64),
                    u_guess_list[0].astype(jnp.float64),
                    params_single,
                )
                xs_np = np.asarray(xs_j, dtype=np.float64)
                us_np = np.asarray(us_j, dtype=np.float64)
                opt_sol_l = {
                    "xl_traj": xs_np,
                    "ul_traj": us_np,
                    "K_FB": [np.zeros((nu, nx), dtype=np.float64) for _ in range(N)],
                }
                opt_soll = {"xl_traj": [xs_np], "ul_traj": [us_np]}
                return opt_soll, [opt_sol_l]
            if len(x0_list) == 1 and getattr(self, "_load_solver_single_jit", None) is not None:
                params_single = jax.tree_util.tree_map(lambda a: a[0], params_b)
                result = self._load_solver_single_jit(
                    x0_list[0].astype(jnp.float64),
                    u_guess_list[0].astype(jnp.float64),
                    params_single,
                )
                result = tuple(jnp.expand_dims(v, axis=0) for v in result)
            else:
                result = self._load_solver_jit(
                    jnp.stack(x0_list).astype(jnp.float64),
                    jnp.stack(u_guess_list).astype(jnp.float64),
                    params_b,
                )
            (xs_np, us_np, payload_xs_np, payload_us_np, Vxx_np, Vx_np, K_np, Hxx_np, Qxu_np, Hxu_np, Huu_np,
             Quu_inv_np, Fx_np, Fu_np, hist_active_np, hist_ratio_np, hist_qu_np) = [
                np.asarray(v, dtype=np.float64) for v in result
            ]
            hist_active_np = hist_active_np.astype(bool)
            if os.environ.get("DIFFCOORD_SUBP1_PRINTS", "0") == "1":
                for i in range(xs_np.shape[0]):
                    iteration = 1
                    for active, ratio, qu_2 in zip(hist_active_np[i], hist_ratio_np[i], hist_qu_np[i]):
                        if active:
                            print('iteration:', iteration, 'ratio=', np.array([[ratio]]), 'Qu_2=', float(qu_2))
                            iteration += 1
            OPt_sol_l = []
            for i in range(xs_np.shape[0]):
                OPt_sol_l.append({
                    "xl_traj": xs_np[i],
                    "ul_traj": us_np[i],
                    "_payload_x_traj": payload_xs_np[i],
                    "_payload_u_traj": payload_us_np[i],
                    "Vxx": [Vxx_np[i, k] for k in range(N + 1)],
                    "Vx": [Vx_np[i, k].reshape(nx, 1) for k in range(N + 1)],
                    "K_FB": [K_np[i, k] for k in range(N)],
                    "Hxx": [Hxx_np[i, k] for k in range(N)],
                    "Qxu": [Qxu_np[i, k] for k in range(N)],
                    "Hxu": [Hxu_np[i, k] for k in range(N)],
                    "Huu": [Huu_np[i, k] for k in range(N)],
                    "Quu_inv": [Quu_inv_np[i, k] for k in range(N)],
                    "Fx": [Fx_np[i, k] for k in range(N)],
                    "Fu": [Fu_np[i, k] for k in range(N)],
                })
            opt_soll = {"xl_traj": [xs_np[i] for i in range(xs_np.shape[0])],
                        "ul_traj": [us_np[i] for i in range(us_np.shape[0])]}
            return opt_soll, OPt_sol_l if need_derivs else None

        if True:
            OPt_sol_l = []
            xl_traj = []
            ul_traj = []
            max_iter = 10
            e_tol = 1e-2
            for Parai in ParaL:
                Parai = np.asarray(Parai).reshape(-1)
                idx = 0
                xl_fb = Parai[idx:idx+self.nxl]; idx += self.nxl
                Ref_xl = Parai[idx:idx+self.nxl*(self.N+1)]; idx += self.nxl*(self.N+1)
                Ref_ul = Parai[idx:idx+self.nul*self.N]; idx += self.nul*self.N
                scxl = Parai[idx:idx+self.nxl*(self.N+1)]; idx += self.nxl*(self.N+1)
                scxL = Parai[idx:idx+self.nxl*(self.N+1)]; idx += self.nxl*(self.N+1)
                scul = Parai[idx:idx+self.nul*self.N]; idx += self.nul*self.N
                scuL = Parai[idx:idx+self.nul*self.N]; idx += self.nul*self.N
                weight1 = Parai[idx:-1]
                i_admm = Parai[-1]
                opt_sol_i = self.DDP_Load_ADMM_Subp1(
                    xl_fb, Ref_xl, Ref_ul, weight1, scxl, scul, scxL, scuL, max_iter, e_tol, i_admm
                )
                OPt_sol_l.append(opt_sol_i)
                xl_traj.append(opt_sol_i["xl_traj"])
                ul_traj.append(opt_sol_i["ul_traj"])
            opt_soll = {"xl_traj": xl_traj, "ul_traj": ul_traj}
            return opt_soll, OPt_sol_l if need_derivs else None

    def jax_MPC_Load_DDP_Planning_SubP1_arrays(
        self, xl_fb, Ref_xl, Ref_ul, scxl_traj_tp, scxL_traj_tp,
        scul_traj_tp, scuL_traj_tp, weight1, i_admm, return_timing=False
    ):
        N, nx, nu = int(self.N), int(self.nxl), int(self.nul)
        weight_arr = np.asarray(weight1, dtype=np.float64).reshape(-1)
        rho_lx = float(self.open_loop_penalty(float(weight_arr[-4]), float(weight_arr[-2]), float(i_admm), self.max_iter_ADMM))
        rho_lu = float(self.open_loop_penalty(float(weight_arr[-3]), float(weight_arr[-1]), float(i_admm), self.max_iter_ADMM))
        x0 = jnp.asarray(np.asarray(xl_fb, dtype=np.float64).reshape(nx), dtype=jnp.float64)
        u_guess = jnp.asarray(np.asarray(Ref_ul, dtype=np.float64).reshape(N, nu), dtype=jnp.float64)
        ref_x_traj = jnp.asarray(np.asarray(Ref_xl, dtype=np.float64).reshape(N + 1, nx), dtype=jnp.float64)
        ref_u_traj = jnp.asarray(np.asarray(Ref_ul, dtype=np.float64).reshape(N, nu), dtype=jnp.float64)
        scx_traj = jnp.asarray(np.asarray(scxl_traj_tp, dtype=np.float64).reshape(N + 1, nx), dtype=jnp.float64)
        yx_traj = jnp.asarray(np.asarray(scxL_traj_tp, dtype=np.float64).reshape(N + 1, nx), dtype=jnp.float64)
        scu_traj = jnp.asarray(np.asarray(scul_traj_tp, dtype=np.float64).reshape(N, nu), dtype=jnp.float64)
        yu_traj = jnp.asarray(np.asarray(scuL_traj_tp, dtype=np.float64).reshape(N, nu), dtype=jnp.float64)
        weight_j = jnp.asarray(weight_arr, dtype=jnp.float64)
        params = {
            "static": {
                "ml": float(self.ml),
                "Jl": jnp.asarray(self.Jl, dtype=jnp.float64),
                "Jl_inv": jnp.asarray(self.Jl_inv, dtype=jnp.float64),
                "dt": float(self.dt),
                "rho_lx": rho_lx,
                "rho_lu": rho_lu,
                "Q_weight": weight_j[0:nx],
                "R_weight": weight_j[2*nx:2*nx+nu],
                "Q_terminal_weight": weight_j[nx:2*nx],
            },
            "stage": {
                "ref_x": ref_x_traj[:-1],
                "ref_u": ref_u_traj,
                "scx": scx_traj[:-1],
                "scu": scu_traj,
                "y_x": yx_traj[:-1],
                "y_u": yu_traj,
            },
            "terminal": {
                "ref_x": ref_x_traj[-1],
                "scx": scx_traj[-1],
                "y_x": yx_traj[-1],
            },
        }
        if (
            os.environ.get("DIFFCOORD_KINODYN_FORWARD_LOAD_KFB", "0") == "1"
            and getattr(self, "_load_policy_solver_single_jit", None) is not None
        ):
            jax_start_time = TM.time()
            xs_j, us_j, K_j = self._load_policy_solver_single_jit(x0, u_guess, params)
            _block_jax_tree_ready((xs_j, us_j, K_j))
            jax_time_ms = (TM.time() - jax_start_time) * 1000
            xs_np = np.asarray(xs_j, dtype=np.float64)
            us_np = np.asarray(us_j, dtype=np.float64)
            K_np = np.asarray(K_j, dtype=np.float64)
            opt_sol = {"xl_traj": xs_np, "ul_traj": us_np, "K_FB": [K_np[k] for k in range(N)]}
            return (opt_sol, jax_time_ms) if return_timing else opt_sol
        jax_start_time = TM.time()
        xs_j, us_j = self._load_traj_solver_single_jit(x0, u_guess, params)
        _block_jax_tree_ready((xs_j, us_j))
        jax_time_ms = (TM.time() - jax_start_time) * 1000
        xs_np = np.asarray(xs_j, dtype=np.float64)
        us_np = np.asarray(us_j, dtype=np.float64)
        opt_sol = {
            "xl_traj": xs_np,
            "ul_traj": us_np,
            "K_FB": [np.zeros((nu, nx), dtype=np.float64) for _ in range(N)],
        }
        return (opt_sol, jax_time_ms) if return_timing else opt_sol

    def jax_MPC_Cable_DDP_Planning_SubP1(self, ParaC, need_derivs=False):
        if os.environ.get("DIFFCOORD_SUBP1_EXACT_JAX", "1") == "1":
            B, N, nx, nu = len(ParaC), int(self.N), int(self.nxi), int(self.nui)
            x0_list, u_init_list, params_list = [], [], []
            for i in range(B):
                parai = np.asarray(ParaC[i]).reshape(-1)
                idx = 0
                xi_fb = parai[idx:idx+nx]; idx += nx
                ref_x = parai[idx:idx+nx*(N+1)]; idx += nx*(N+1)
                ref_ui = parai[idx:idx+nu]; idx += nu
                scxi = parai[idx:idx+nx*(N+1)]; idx += nx*(N+1)
                scxI = parai[idx:idx+nx*(N+1)]; idx += nx*(N+1)
                scui = parai[idx:idx+nu*N]; idx += nu*N
                scuI = parai[idx:idx+nu*N]; idx += nu*N
                weight_para = parai[idx:-1]
                i_admm = parai[-1]
                rho_ix = float(self.open_loop_penalty(float(weight_para[-4]), float(weight_para[-2]), float(i_admm), self.max_iter_ADMM))
                rho_iu = float(self.open_loop_penalty(float(weight_para[-3]), float(weight_para[-1]), float(i_admm), self.max_iter_ADMM))
                x0_list.append(jnp.asarray(xi_fb, dtype=jnp.float64))
                ref_u_vec = jnp.asarray(ref_ui, dtype=jnp.float64)
                u_init_list.append(jnp.tile(ref_u_vec, (N, 1)))
                ref_x_traj = jnp.asarray(ref_x, dtype=jnp.float64).reshape(N + 1, nx)
                scx_traj = jnp.asarray(scxi, dtype=jnp.float64).reshape(N + 1, nx)
                yx_traj = jnp.asarray(scxI, dtype=jnp.float64).reshape(N + 1, nx)
                scu_traj = jnp.asarray(scui, dtype=jnp.float64).reshape(N, nu)
                yu_traj = jnp.asarray(scuI, dtype=jnp.float64).reshape(N, nu)
                weight_j = jnp.asarray(weight_para, dtype=jnp.float64)
                params_list.append({
                    "static": {
                        "rho_ix": rho_ix,
                        "rho_iu": rho_iu,
                        "dt": float(self.dt),
                        "Qi_weight": weight_j[0:nx],
                        "Ri_weight": weight_j[2*nx:2*nx+nu],
                        "Qi_terminal_weight": weight_j[nx:2*nx],
                    },
                    "stage": {
                        "ref_x_i": ref_x_traj[:-1],
                        "ref_u_i": jnp.tile(ref_u_vec[None, :], (N, 1)),
                        "scx_i": scx_traj[:-1],
                        "scu_i": scu_traj,
                        "y_x_i": yx_traj[:-1],
                        "y_u_i": yu_traj,
                    },
                    "terminal": {
                        "ref_x_i": ref_x_traj[-1],
                        "scx_i": scx_traj[-1],
                        "y_x_i": yx_traj[-1],
                    },
                })
            params_b = jax.tree_util.tree_map(lambda *args: jnp.stack(args).astype(jnp.float64), *params_list)
            if not need_derivs and getattr(self, "_cable_traj_solver_jit", None) is not None:
                x0_stack = jnp.stack(x0_list).astype(jnp.float64)
                u_stack = jnp.stack(u_init_list).astype(jnp.float64)
                if (
                    os.environ.get("DIFFCOORD_KINODYN_CABLE_JAX_THREADS", "0") == "1"
                    and B > 1
                    and getattr(self, "_cable_traj_solver_single_jit", None) is not None
                ):
                    def solve_one(i):
                        params_i = jax.tree_util.tree_map(lambda a: a[i], params_b)
                        return self._cable_traj_solver_single_jit(x0_stack[i], u_stack[i], params_i)

                    workers = min(B, int(os.environ.get("DIFFCOORD_KINODYN_CABLE_JAX_WORKERS", str(B))))
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        per_cable = list(executor.map(solve_one, range(B)))
                    xi_np = np.asarray([np.asarray(item[0], dtype=np.float64) for item in per_cable], dtype=np.float64)
                    ui_np = np.asarray([np.asarray(item[1], dtype=np.float64) for item in per_cable], dtype=np.float64)
                elif B == getattr(self, "_cable_solver_pmap_size", 0) and getattr(self, "_cable_traj_solver_pmap", None) is not None:
                    traj_result = self._cable_traj_solver_pmap(
                        x0_stack,
                        u_stack,
                        params_b,
                    )
                    xi_np, ui_np = [
                        np.asarray(v, dtype=np.float64) for v in traj_result
                    ]
                else:
                    traj_result = self._cable_traj_solver_jit(
                        x0_stack,
                        u_stack,
                        params_b,
                    )
                    xi_np, ui_np = [
                        np.asarray(v, dtype=np.float64) for v in traj_result
                    ]
                opt_solc = {"xc_traj": [xi_np[i] for i in range(B)],
                            "uc_traj": [ui_np[i] for i in range(B)]}
                return opt_solc, None
            if B == getattr(self, "_cable_solver_pmap_size", 0) and getattr(self, "_cable_solver_pmap", None) is not None:
                result = self._cable_solver_pmap(
                    jnp.stack(x0_list).astype(jnp.float64),
                    jnp.stack(u_init_list).astype(jnp.float64),
                    params_b,
                )
            else:
                result = self._cable_solver_jit(
                    jnp.stack(x0_list).astype(jnp.float64),
                    jnp.stack(u_init_list).astype(jnp.float64),
                    params_b,
                )
            (xi_np, ui_np, payload_xi_np, payload_ui_np, Vxx_np, Vx_np, K_np, Hxx_np, Qxu_np, Hxu_np, Huu_np,
             Quu_inv_np, Fx_np, Fu_np, hist_active_np, hist_ratio_np, hist_qu_np) = [
                np.asarray(v, dtype=np.float64) for v in result
            ]
            hist_active_np = hist_active_np.astype(bool)
            if os.environ.get("DIFFCOORD_SUBP1_PRINTS", "0") == "1":
                for i in range(B):
                    iteration = 1
                    for active, ratio, qu_2 in zip(hist_active_np[i], hist_ratio_np[i], hist_qu_np[i]):
                        if active:
                            print('iteration:', iteration, 'ratio=', np.array([[ratio]]), 'Qu_2=', float(qu_2))
                            iteration += 1
            OPt_sol_c = []
            for i in range(B):
                OPt_sol_c.append({
                    "xi_traj": xi_np[i],
                    "ui_traj": ui_np[i],
                    "_payload_x_traj": payload_xi_np[i],
                    "_payload_u_traj": payload_ui_np[i],
                    "Vxx": [Vxx_np[i, k] for k in range(N + 1)],
                    "Vx": [Vx_np[i, k].reshape(nx, 1) for k in range(N + 1)],
                    "K_FB": [K_np[i, k] for k in range(N)],
                    "Hxx": [Hxx_np[i, k] for k in range(N)],
                    "Qxu": [Qxu_np[i, k] for k in range(N)],
                    "Hxu": [Hxu_np[i, k] for k in range(N)],
                    "Huu": [Huu_np[i, k] for k in range(N)],
                    "Quu_inv": [Quu_inv_np[i, k] for k in range(N)],
                    "Fx": [Fx_np[i, k] for k in range(N)],
                    "Fu": [Fu_np[i, k] for k in range(N)],
                })
            opt_solc = {"xc_traj": [xi_np[i] for i in range(B)],
                        "uc_traj": [ui_np[i] for i in range(B)]}
            return opt_solc, OPt_sol_c

        return self.MPC_Cable_DDP_Planning_SubP1_process(ParaC)

    def jax_MPC_Cable_DDP_Planning_SubP1_arrays(
        self, xq_fb, ref_xq, ref_uq, scxc_traj_tp, scxC_traj_tp,
        scuc_traj_tp, scuC_traj_tp, weight2, i_admm, return_timing=False
    ):
        B, N, nx, nu = int(self.nq), int(self.N), int(self.nxi), int(self.nui)
        weight_arr = np.asarray(weight2, dtype=np.float64).reshape(-1)
        rho_ix = float(self.open_loop_penalty(float(weight_arr[-4]), float(weight_arr[-2]), float(i_admm), self.max_iter_ADMM))
        rho_iu = float(self.open_loop_penalty(float(weight_arr[-3]), float(weight_arr[-1]), float(i_admm), self.max_iter_ADMM))
        x0_stack = jnp.asarray(np.asarray(xq_fb, dtype=np.float64).reshape(B, nx), dtype=jnp.float64)
        ref_x_stack = jnp.asarray(np.asarray(ref_xq, dtype=np.float64).reshape(B, N + 1, nx), dtype=jnp.float64)
        ref_ui_stack = jnp.asarray(np.asarray(ref_uq, dtype=np.float64).reshape(B, nu), dtype=jnp.float64)
        u_stack = jnp.tile(ref_ui_stack[:, None, :], (1, N, 1))
        scx_stack = jnp.asarray(np.asarray(scxc_traj_tp, dtype=np.float64).reshape(B, N + 1, nx), dtype=jnp.float64)
        yx_stack = jnp.asarray(np.asarray(scxC_traj_tp, dtype=np.float64).reshape(B, N + 1, nx), dtype=jnp.float64)
        scu_stack = jnp.asarray(np.asarray(scuc_traj_tp, dtype=np.float64).reshape(B, N, nu), dtype=jnp.float64)
        yu_stack = jnp.asarray(np.asarray(scuC_traj_tp, dtype=np.float64).reshape(B, N, nu), dtype=jnp.float64)
        weight_j = jnp.asarray(weight_arr, dtype=jnp.float64)
        params_b = {
            "static": {
                "rho_ix": jnp.full((B,), rho_ix, dtype=jnp.float64),
                "rho_iu": jnp.full((B,), rho_iu, dtype=jnp.float64),
                "dt": jnp.full((B,), float(self.dt), dtype=jnp.float64),
                "Qi_weight": jnp.tile(weight_j[0:nx], (B, 1)),
                "Ri_weight": jnp.tile(weight_j[2*nx:2*nx+nu], (B, 1)),
                "Qi_terminal_weight": jnp.tile(weight_j[nx:2*nx], (B, 1)),
            },
            "stage": {
                "ref_x_i": ref_x_stack[:, :-1],
                "ref_u_i": jnp.tile(ref_ui_stack[:, None, :], (1, N, 1)),
                "scx_i": scx_stack[:, :-1],
                "scu_i": scu_stack,
                "y_x_i": yx_stack[:, :-1],
                "y_u_i": yu_stack,
            },
            "terminal": {
                "ref_x_i": ref_x_stack[:, -1],
                "scx_i": scx_stack[:, -1],
                "y_x_i": yx_stack[:, -1],
            },
        }
        jax_start_time = TM.time()
        if B == getattr(self, "_cable_solver_pmap_size", 0) and getattr(self, "_cable_traj_solver_pmap", None) is not None:
            traj_result = self._cable_traj_solver_pmap(x0_stack, u_stack, params_b)
        else:
            traj_result = self._cable_traj_solver_jit(x0_stack, u_stack, params_b)
        _block_jax_tree_ready(traj_result)
        jax_time_ms = (TM.time() - jax_start_time) * 1000
        xi_j, ui_j = traj_result
        xi_np = np.asarray(xi_j, dtype=np.float64)
        ui_np = np.asarray(ui_j, dtype=np.float64)
        opt_sol = {"xc_traj": [xi_np[i] for i in range(B)],
                   "uc_traj": [ui_np[i] for i in range(B)]}
        return (opt_sol, None, jax_time_ms) if return_timing else (opt_sol, None)

    def close_cable_ddp_process_pool(self):
        pool = getattr(self, "_cable_ddp_process_pool", None)
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
        self._cable_ddp_process_pool = None
        self._cable_ddp_process_pool_config = None

    def _get_cable_ddp_process_pool(self):
        workers_cap = int(os.environ.get("DIFFCOORD_KINODYN_CABLE_DDP_WORKERS_CAP", str(max(1, int(self.nq)))))
        workers = int(os.environ.get(
            "DIFFCOORD_KINODYN_CABLE_DDP_WORKERS",
            str(min(max(1, int(self.nq)), os.cpu_count() or 1, max(1, workers_cap))),
        ))
        workers = max(1, min(int(self.nq), workers))
        config = (workers,)
        if self._cable_ddp_process_pool is not None and self._cable_ddp_process_pool_config == config:
            return self._cable_ddp_process_pool, workers
        self.close_cable_ddp_process_pool()
        global _KINODYN_CABLE_DDP_PROCESS_STATE
        _KINODYN_CABLE_DDP_PROCESS_STATE = {"planner": self}
        ctx = MP.get_context("fork")
        with _no_debugger_trace_during_fork():
            self._cable_ddp_process_pool = ctx.Pool(
                processes=workers,
                initializer=_disable_debugger_trace_in_worker,
            )
        self._cable_ddp_process_pool_config = config
        return self._cable_ddp_process_pool, workers

    def MPC_Cable_DDP_Planning_SubP1_process(self, ParaC):
        nq = int(self.nq)
        pool, _ = self._get_cable_ddp_process_pool()
        items = [(i, np.asarray(ParaC[i], dtype=np.float64).reshape(-1)) for i in range(nq)]
        results = pool.map(_kinodyn_cable_ddp_process_solve, items, chunksize=1)
        OPt_sol_c = [None for _ in range(nq)]
        for i, opt_sol_i in results:
            OPt_sol_c[i] = opt_sol_i
        xc_traj = [OPt_sol_c[i]["xi_traj"] for i in range(nq)]
        uc_traj = [OPt_sol_c[i]["ui_traj"] for i in range(nq)]
        opt_solc = {"xc_traj": xc_traj, "uc_traj": uc_traj}
        return opt_solc, OPt_sol_c

    def MPC_Cable_DDP_Planning_SubP1(self,ParaC,need_derivs=True): # checked, correct, Apr.1 2025
        backend = os.environ.get("DIFFCOORD_KINODYN_CABLE_DDP_BACKEND", "jax").strip().lower()
        if (
            backend in ("jax", "jit", "batched")
            and jax is not None
            and os.environ.get("DIFFCOORD_SUBP1_EXACT_JAX", "1") == "1"
            and getattr(self, "_cable_solver_jit", None) is not None
        ):
            return self.jax_MPC_Cable_DDP_Planning_SubP1(ParaC, need_derivs=need_derivs)
        if backend in ("process", "proc", "mp") and int(self.nq) > 1:
            return self.MPC_Cable_DDP_Planning_SubP1_process(ParaC)
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

    
    def SetConstriants(self, pob1, pob2, pob3, pob4):
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
        self.Gi3    = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 3rd obstacle
        self.Gi4    = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 4th obstacle
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
        
        po1   = (pl_k[0:2]-pob1).T@(pl_k[0:2]-pob1) - ((self.ro)+self.rq/2)**2 # vertical obstacles
        self.po1_fn= Function('pl1_admm',[self.scxl],[po1],['scxl0'],['po1f_admm'])
        self.G_lo += -self.p_bar * log(po1)
        po2   = (pl_k[0:2]-pob2).T@(pl_k[0:2]-pob2) - ((self.ro)+self.rq/2)**2 # vertical obstacles
        self.po2_fn= Function('pl2_admm',[self.scxl],[po2],['scxl0'],['po2f_admm'])
        self.G_lo += -self.p_bar * log(po2)
        po3   = (pl_k[1:3]-pob3).T@(pl_k[1:3]-pob3) - ((self.rov)+self.rq/2)**2 # horizontal obstacles, y-z
        self.po3_fn= Function('pl2_admm',[self.scxl],[po3],['scxl0'],['po3f_admm'])
        self.G_lo += -self.p_bar * log(po3)
        po4   = (pl_k[1:3]-pob4).T@(pl_k[1:3]-pob4) - ((self.rov)+self.rq/2)**2 # horizontal obstacles, y-z
        self.po4_fn= Function('pl2_admm',[self.scxl],[po4],['scxl0'],['po4f_admm'])
        self.G_lo += -self.p_bar * log(po3)

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
                    diso3 = pi_k[1:3]-pob3
                    go3  = diso3.T@diso3 - ((self.rq + self.ro)+self.rq)**2 # safe constriant between the obstacle 3 and the ith quadrotor, which should be positive. 
                    go3_fn = Function('go3'+str(i),[self.scxl,self.scxc],[go3],['scxl0','scxc0'],['go3f'+str(i)])
                    self.gco += -self.p_bar * log(go3)
                    self.Gi3 += [go3_fn]
                    diso4 = pi_k[1:3]-pob4
                    go4  = diso4.T@diso4 - ((self.rq + self.ro)+self.rq)**2 # safe constriant between the obstacle 4 and the ith quadrotor, which should be positive
                    go4_fn = Function('go4'+str(i),[self.scxl,self.scxc],[go4],['scxl0','scxc0'],['go4f'+str(i)])
                    self.gco += -self.p_bar * log(go4)
                    self.Gi4 += [go4_fn]
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
                    gij    = disij.T@disij - (kc/(num_dis)*6*self.rq)**2 # 4rq in training, 5rq in evaluation
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
        self.J_2_soft_k    = self.Jl_P2_k + self.gco + self.gij + self.Tcon + self.ql_n + self.din + self.sumfi + self.h_wcons + self.Uicon
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
        self.J_2_soft_k_orig =  self.gco + self.gij + self.Tcon + self.ql_n + self.din + self.sumfi + self.h_wcons + self.Uicon
    

    def SetADMMSubP2_SoftCost_N(self):
        # at the terminal step N
        self.J_2_soft_N    = self.Jl_P2_N  + self.gco + self.gij + self.Tcon + self.ql_n + self.din 
        for i in range(int(self.nq)):
            xi      = self.xc[i*self.nxi:(i+1)*self.nxi]   # cable primal state
            scxi    = self.scxc[i*self.nxi:(i+1)*self.nxi] # safe copy state of each cable
            scxI    = self.scxC[i*self.nxi:(i+1)*self.nxi] # Lagrangian multiplier
            resid_x = xi - scxi + scxI/self.pix_dis
            self.J_2_soft_N    += self.pix_dis/2*resid_x.T@resid_x 
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
        # load safe constraints are important!
        g2         += [self.po1_fn(scxl0=scxl_k)['po1f_admm']]
        self.lbg2  += [1e-2]
        self.ubg2  += [1e4]
        g2         += [self.po2_fn(scxl0=scxl_k)['po2f_admm']]
        self.lbg2  += [1e-2]
        self.ubg2  += [1e4]
        g2         += [self.po3_fn(scxl0=scxl_k)['po3f_admm']]
        self.lbg2  += [1e-2]
        self.ubg2  += [1e4]
        g2         += [self.po4_fn(scxl0=scxl_k)['po4f_admm']]
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
            # safe constriant between the obstacle 3 and the ith quadrotor
            goi3       = self.Gi3[i](scxl0=scxl_k,scxc0=scxc_k)['go3f'+str(i)]
            g2        += [goi3]
            self.lbg2 += [1e-2]
            self.ubg2 += [1e4] # add an upbound for numerical stability
            # safe constriant between the obstacle 4 and the ith quadrotor
            goi4       = self.Gi4[i](scxl0=scxl_k,scxc0=scxc_k)['go4f'+str(i)]
            g2        += [goi4]
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
            # gijc       = self.Gijc[k](scxl0=scxl_k,scxc0=scxc_k)['gcf'+str(k)]
            # g2        += [gijc]
            # self.lbg2 += [1e-2]
            # self.ubg2 += [1e4]
        
        # control consensus constraint
        g_wc       = self.W_cons_fn(scxl0=scxl_k,scul0=scul_k,scxc0=scxc_k)['W_consf']
        g2        += [g_wc]
        self.lbg2 += self.nul*[0]
        self.ubg2 += self.nul*[0] 

        # create an NLP solver and solve it
        optsi2 = {}
        optsi2['ipopt.tol'] = 1e-8
        optsi2['ipopt.print_level'] = 0
        optsi2['ipopt.sb'] = 'yes'
        optsi2['print_time'] = 0
        optsi2['ipopt.warm_start_init_point']='yes'
        optsi2['ipopt.max_iter']=2e3
        optsi2['ipopt.acceptable_tol']=1e-8
        # optsi2['ipopt.mu_strategy']='adaptive'
        # optsi2['ipopt.bound_relax_factor']=1e-12
        # optsi2['ipopt.limited_memory_max_history'] = 20
        # optsi2['ipopt.nlp_scaling_method']='gradient-based'
        # optsi2['ipopt.limited_memory_initialization'] = 'scalar1'
        prob2 = {'f': J2, 
                'x': vertcat(*w2), 
                'p': Para2,
                'g': vertcat(*g2)}
        self.solver2 = nlpsol('solver2', 'ipopt', prob2, optsi2)  
        self._solver2_map_cache = {}
        self._solver2_map_bound_cache = {}
        self.close_solver2_process_pool()



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
        g2N         += [self.po3_fn(scxl0=scxl_k)['po3f_admm']]
        self.lbg2N  += [1e-2]
        self.ubg2N  += [1e4]
        g2N         += [self.po4_fn(scxl0=scxl_k)['po4f_admm']]
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
            # safe constriant between the obstacle 3 and the ith quadrotor
            goi3        = self.Gi3[i](scxl0=scxl_k,scxc0=scxc_k)['go3f'+str(i)]
            g2N        += [goi3]
            self.lbg2N += [1e-2]
            self.ubg2N += [1e4] # add an upbound for numerical stability
            # safe constriant between the obstacle 4 and the ith quadrotor
            goi4        = self.Gi4[i](scxl0=scxl_k,scxc0=scxc_k)['go4f'+str(i)]
            g2N        += [goi4]
            self.lbg2N += [1e-2]
            self.ubg2N += [1e4] # add an upbound for numerical stability
            # direction unit norm
            g2N        += [self.Di[i](scxc0=scxc_k)['dn'+str(i)]]
            self.lbg2N += [1]
            self.ubg2N += [1]
            # avoidance of cable-crossing constraints
            gio        = self.Gio[i](scxl0=scxl_k,scxc0=scxc_k)['gcf'+str(i)]
            g2N        += [gio]
            self.lbg2N += [-self.rl]
            self.ubg2N += [self.rl+self.cl0] 
        
        for k in range(len(self.Gij)):
            gij         = self.Gij[k](scxl0=scxl_k,scxc0=scxc_k)['gf'+str(k)]
            g2N        += [gij]
            self.lbg2N += [1e-2]
            self.ubg2N += [1e4]
            # gijc       = self.Gijc[k](scxl0=scxl_k,scxc0=scxc_k)['gcf'+str(k)]
            # g2N        += [gijc]
            # self.lbg2N += [1e-2]
            # self.ubg2N += [1e4]

        # create an NLP solver and solve it
        optsi2 = {}
        optsi2['ipopt.tol'] = 1e-8
        optsi2['ipopt.print_level'] = 0
        optsi2['ipopt.sb'] = 'yes'
        optsi2['print_time'] = 0
        optsi2['ipopt.warm_start_init_point']='yes'
        optsi2['ipopt.max_iter']=2e3
        optsi2['ipopt.acceptable_tol']=1e-8
        optsi2['ipopt.mu_strategy']='adaptive'
        # optsi2['ipopt.bound_relax_factor']=1e-12
        # optsi2['ipopt.limited_memory_max_history'] = 20
        # optsi2['ipopt.nlp_scaling_method']='gradient-based'
        # optsi2['ipopt.limited_memory_initialization'] = 'scalar1'
        prob2N = {'f': J2, 
                'x': vertcat(*w2N), 
                'p': Para2,
                'g': vertcat(*g2N)}
        self.solver2N = nlpsol('solver2N', 'ipopt', prob2N, optsi2)  
        self.close_solver2_process_pool()
        backend = os.environ.get("DIFFCOORD_KINODYN_SUBP2_BACKEND", "process").strip().lower()
        if os.environ.get("DIFFCOORD_KINODYN_SUBP2_PREWARM", "1") == "1" and backend in ("mapped", "map", "thread", "openmp", "casadi"):
            self._get_solver2_map()
        elif os.environ.get("DIFFCOORD_KINODYN_SUBP2_PREWARM", "1") == "1" and backend in ("process", "pool", "fork", "mp", "multiprocess"):
            self._get_solver2_process_pool()


    
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

    def _get_solver2_map(self):
        mode = os.environ.get("DIFFCOORD_KINODYN_SUBP2_MAP_MODE", "thread").strip().lower()
        cap = int(os.environ.get("DIFFCOORD_KINODYN_SUBP2_THREADS_CAP", "8"))
        default_threads = min(int(self.N), os.cpu_count() or 1, max(1, cap))
        threads = int(os.environ.get("DIFFCOORD_KINODYN_SUBP2_THREADS", str(default_threads)))
        threads = max(1, min(int(self.N), threads))
        key = (int(self.N), mode, threads)
        if key not in self._solver2_map_cache:
            actual_mode = mode
            if mode in ("thread", "openmp"):
                try:
                    self._solver2_map_cache[key] = self.solver2.map(self.N, mode, threads)
                except RuntimeError:
                    actual_mode = "thread"
                    fallback_key = (int(self.N), "thread", threads)
                    if fallback_key not in self._solver2_map_cache:
                        self._solver2_map_cache[fallback_key] = self.solver2.map(self.N, "thread", threads)
                    self._solver2_map_cache[key] = self._solver2_map_cache[fallback_key]
            else:
                self._solver2_map_cache[key] = self.solver2.map(self.N, mode)
            if os.environ.get("DIFFCOORD_KINODYN_SUBP2_PRINT_CONFIG", "0") == "1":
                print(f"SubP2 map config: mode={actual_mode} threads={threads} horizon={int(self.N)}")
        return self._solver2_map_cache[key]

    def _get_solver2_map_bounds(self):
        key = (int(self.N), len(self.lbw2), len(self.lbg2))
        if key not in self._solver2_map_bound_cache:
            lbx_vec = np.asarray(self.lbw2, dtype=np.float64).reshape(-1, 1)
            ubx_vec = np.asarray(self.ubw2, dtype=np.float64).reshape(-1, 1)
            lbg_vec = np.asarray(self.lbg2, dtype=np.float64).reshape(-1, 1)
            ubg_vec = np.asarray(self.ubg2, dtype=np.float64).reshape(-1, 1)
            self._solver2_map_bound_cache[key] = (
                np.asfortranarray(np.tile(lbx_vec, (1, self.N))),
                np.asfortranarray(np.tile(ubx_vec, (1, self.N))),
                np.asfortranarray(np.tile(lbg_vec, (1, self.N))),
                np.asfortranarray(np.tile(ubg_vec, (1, self.N))),
            )
        return self._solver2_map_bound_cache[key]

    def close_solver2_process_pool(self):
        pool = getattr(self, "_solver2_process_pool", None)
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
        self._solver2_process_pool = None
        self._solver2_process_pool_config = None

    def _get_solver2_process_pool(self):
        workers_cap = int(os.environ.get("DIFFCOORD_KINODYN_SUBP2_WORKERS_CAP", "16"))
        workers = int(os.environ.get(
            "DIFFCOORD_KINODYN_SUBP2_WORKERS",
            str(min(self.N + 1, os.cpu_count() or 1, max(1, workers_cap))),
        ))
        workers = max(1, min(self.N + 1, workers))
        config = (workers,)
        if self._solver2_process_pool is not None and self._solver2_process_pool_config == config:
            return self._solver2_process_pool, workers
        self.close_solver2_process_pool()
        global _KINODYN_SUBP2_PROCESS_STATE
        _KINODYN_SUBP2_PROCESS_STATE = {
            "solver2": self.solver2,
            "solver2N": self.solver2N,
            "lbw2": self.lbw2,
            "ubw2": self.ubw2,
            "lbg2": self.lbg2,
            "ubg2": self.ubg2,
            "lbw2N": self.lbw2N,
            "ubw2N": self.ubw2N,
            "lbg2N": self.lbg2N,
            "ubg2N": self.ubg2N,
        }
        ctx = MP.get_context("fork")
        with _no_debugger_trace_during_fork():
            self._solver2_process_pool = ctx.Pool(
                processes=workers,
                initializer=_disable_debugger_trace_in_worker,
            )
        self._solver2_process_pool_config = config
        return self._solver2_process_pool, workers

    def _build_admm_subp2_process_tasks(self, Para2_cable):
        Para2_cable = np.asarray(Para2_cable, dtype=np.float64).reshape(-1)
        n_start_pl = 3*self.nxl*(self.N+1)+3*self.nul*self.N+3*self.nxi*int(self.nq)*(self.N+1)+2*self.nui*int(self.nq)*self.N + self.nui*int(self.nq)
        para_l = Para2_cable[n_start_pl:n_start_pl+self.npl]
        para_i = Para2_cable[n_start_pl+self.npl:n_start_pl+self.npl+self.npi]
        a = Para2_cable[-1]
        n_start_xl = self.nxl*(self.N+1)+self.nul*self.N+self.nxi*int(self.nq)*(self.N+1)+self.nui*int(self.nq)
        n_start_scxL = n_start_xl + self.nxl*(self.N+1)
        n_start_ul = n_start_scxL + self.nxl*(self.N+1)
        n_start_scuL = n_start_ul + self.nul*self.N
        n_start_xc = n_start_scuL + self.nul*self.N
        n_start_scxC = n_start_xc + self.nxi*int(self.nq)*(self.N+1)
        n_start_uc = n_start_scxC + self.nxi*int(self.nq)*(self.N+1)
        n_start_scuC = n_start_uc + self.nui*int(self.nq)*self.N
        ref_xq_start = self.nxl*(self.N+1)+self.nul*self.N
        ref_uq_start = ref_xq_start+self.nxi*int(self.nq)*(self.N+1)
        tasks = []
        for k in range(self.N):
            xl_ref = Para2_cable[k*self.nxl:(k+1)*self.nxl]
            ul_ref = Para2_cable[2*self.nxl*(self.N+1)+k*self.nul:2*self.nxl*(self.N+1)+(k+1)*self.nul]
            x0_parts = [xl_ref, ul_ref]
            xq_ref_k = Para2_cable[ref_xq_start+k*self.nxi*int(self.nq):ref_xq_start+(k+1)*self.nxi*int(self.nq)]
            for i in range(int(self.nq)):
                x0_parts.append(xq_ref_k[i*self.nxi:(i+1)*self.nxi])
                x0_parts.append(Para2_cable[ref_uq_start+i*self.nui:ref_uq_start+(i+1)*self.nui])
            x0_k = np.concatenate(x0_parts)
            xl_k = Para2_cable[n_start_xl+k*self.nxl:n_start_xl+(k+1)*self.nxl]
            scxL_k = Para2_cable[n_start_scxL+k*self.nxl:n_start_scxL+(k+1)*self.nxl]
            ul_k = Para2_cable[n_start_ul+k*self.nul:n_start_ul+(k+1)*self.nul]
            scuL_k = Para2_cable[n_start_scuL+k*self.nul:n_start_scuL+(k+1)*self.nul]
            xc_k = Para2_cable[n_start_xc+k*self.nxi*int(self.nq):n_start_xc+(k+1)*self.nxi*int(self.nq)]
            scxC_k = Para2_cable[n_start_scxC+k*self.nxi*int(self.nq):n_start_scxC+(k+1)*self.nxi*int(self.nq)]
            uc_k = Para2_cable[n_start_uc+k*self.nui*int(self.nq):n_start_uc+(k+1)*self.nui*int(self.nq)]
            scuC_k = Para2_cable[n_start_scuC+k*self.nui*int(self.nq):n_start_scuC+(k+1)*self.nui*int(self.nq)]
            p_k = np.concatenate((xl_k, scxL_k, ul_k, scuL_k, xc_k, scxC_k, uc_k, scuC_k, para_l, para_i, [a]))
            tasks.append(("k", k, x0_k, p_k))
        xl_ref_N = Para2_cable[self.N*self.nxl:(self.N+1)*self.nxl]
        xq_ref_N = Para2_cable[ref_xq_start+self.N*self.nxi*int(self.nq):ref_xq_start+(self.N+1)*self.nxi*int(self.nq)]
        x0N_parts = [xl_ref_N]
        for i in range(int(self.nq)):
            x0N_parts.append(xq_ref_N[i*self.nxi:(i+1)*self.nxi])
        x0_N = np.concatenate(x0N_parts)
        xl_N = Para2_cable[n_start_xl+self.N*self.nxl:n_start_xl+(self.N+1)*self.nxl]
        scxL_N = Para2_cable[n_start_scxL+self.N*self.nxl:n_start_scxL+(self.N+1)*self.nxl]
        xc_N = Para2_cable[n_start_xc+self.N*self.nxi*int(self.nq):n_start_xc+(self.N+1)*self.nxi*int(self.nq)]
        scxC_N = Para2_cable[n_start_scxC+self.N*self.nxi*int(self.nq):n_start_scxC+(self.N+1)*self.nxi*int(self.nq)]
        p_N = np.concatenate((xl_N, scxL_N, xc_N, scxC_N, para_l, para_i, [a]))
        tasks.append(("N", self.N, x0_N, p_N))
        return tasks

    def ADMM_SubP2_process(self, Para2_cable):
        scxl_traj = np.zeros((self.N+1,self.nxl))
        scul_traj = np.zeros((self.N,self.nul))
        scxc_traj = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        scuc_traj = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))]
        tasks = self._build_admm_subp2_process_tasks(Para2_cable)
        pool, _ = self._get_solver2_process_pool()
        chunk_size = int(os.environ.get("DIFFCOORD_KINODYN_SUBP2_CHUNKSIZE", "1"))
        results = pool.map(_kinodyn_subp2_process_solve, tasks, chunksize=chunk_size)
        for kind, k, w_opt in results:
            if kind == "N":
                sol_trajN = np.reshape(w_opt, (-1, self.nxl + self.nxi*int(self.nq)))
                scxl_traj[self.N:self.N+1,:] = sol_trajN[:,0:self.nxl]
                scxc_optN = sol_trajN[:,self.nxl:self.nxl+self.nxi*int(self.nq)]
                for i in range(int(self.nq)):
                    scxc_traj[i][self.N:self.N+1,:] = scxc_optN[:,i*self.nxi:(i+1)*self.nxi]
            else:
                sol_traj = np.reshape(w_opt, (-1, self.nxl + self.nul + (self.nxi+self.nui)*int(self.nq)))
                scxl_traj[k:k+1,:] = sol_traj[:,0:self.nxl]
                scul_traj[k:k+1,:] = sol_traj[:,self.nxl:self.nxl+self.nul]
                scc_opt = sol_traj[:,self.nxl+self.nul:self.nxl+self.nul+(self.nxi+self.nui)*int(self.nq)]
                for i in range(int(self.nq)):
                    scxc_traj[i][k:k+1,:] = scc_opt[:,i*(self.nxi+self.nui):i*(self.nxi+self.nui)+self.nxi]
                    scuc_traj[i][k:k+1,:] = scc_opt[:,i*(self.nxi+self.nui)+self.nxi:(i+1)*(self.nxi+self.nui)]
        return {"scxl_traj": scxl_traj, "scul_traj": scul_traj, "scxc_traj": scxc_traj, "scuc_traj": scuc_traj}

    def ADMM_SubP2_mapped(self, Para2_cable):
        scxl_traj = np.zeros((self.N+1,self.nxl))
        scul_traj = np.zeros((self.N,self.nul))
        scxc_traj = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        scuc_traj = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))]
        tasks = self._build_admm_subp2_process_tasks(Para2_cable)
        stage_tasks = tasks[:-1]
        terminal_task = tasks[-1]
        decision_dim = len(self.lbw2)
        param_dim = len(stage_tasks[0][3])
        x0_mat = np.zeros((decision_dim, self.N), dtype=np.float64)
        p_mat = np.zeros((param_dim, self.N), dtype=np.float64)
        for _kind, k, x0_k, p_k in stage_tasks:
            x0_mat[:, k] = x0_k
            p_mat[:, k] = p_k
        solver2_map = self._get_solver2_map()
        lbx_mat, ubx_mat, lbg_mat, ubg_mat = self._get_solver2_map_bounds()
        sol = solver2_map(
            x0=np.asfortranarray(x0_mat),
            lbx=lbx_mat,
            ubx=ubx_mat,
            p=np.asfortranarray(p_mat),
            lbg=lbg_mat,
            ubg=ubg_mat,
        )
        w_opt_mat = np.asarray(sol["x"].full(), dtype=np.float64).reshape((decision_dim, self.N), order="F")
        for k in range(self.N):
            sol_traj = np.reshape(w_opt_mat[:, k], (-1, self.nxl + self.nul + (self.nxi+self.nui)*int(self.nq)))
            scxl_traj[k:k+1,:] = sol_traj[:,0:self.nxl]
            scul_traj[k:k+1,:] = sol_traj[:,self.nxl:self.nxl+self.nul]
            scc_opt = sol_traj[:,self.nxl+self.nul:self.nxl+self.nul+(self.nxi+self.nui)*int(self.nq)]
            for i in range(int(self.nq)):
                scxc_traj[i][k:k+1,:] = scc_opt[:,i*(self.nxi+self.nui):i*(self.nxi+self.nui)+self.nxi]
                scuc_traj[i][k:k+1,:] = scc_opt[:,i*(self.nxi+self.nui)+self.nxi:(i+1)*(self.nxi+self.nui)]
        _kind, _k, x0_N, p_N = terminal_task
        sol2N = self.solver2N(
            x0=x0_N,
            lbx=self.lbw2N,
            ubx=self.ubw2N,
            p=p_N,
            lbg=self.lbg2N,
            ubg=self.ubg2N,
        )
        w_opt2N = np.asarray(sol2N["x"].full(), dtype=np.float64).reshape(-1)
        sol_trajN = np.reshape(w_opt2N, (-1, self.nxl + self.nxi*int(self.nq)))
        scxl_traj[self.N:self.N+1,:] = sol_trajN[:,0:self.nxl]
        scxc_optN = sol_trajN[:,self.nxl:self.nxl+self.nxi*int(self.nq)]
        for i in range(int(self.nq)):
            scxc_traj[i][self.N:self.N+1,:] = scxc_optN[:,i*self.nxi:(i+1)*self.nxi]
        return {"scxl_traj": scxl_traj, "scul_traj": scul_traj, "scxc_traj": scxc_traj, "scuc_traj": scuc_traj}

    def ADMM_SubP2_parallel(self, Para2_cable):
        backend = os.environ.get("DIFFCOORD_KINODYN_SUBP2_BACKEND", "process").strip().lower()
        if backend in ("seq", "serial", "sequential", "none"):
            return self.ADMM_SubP2(Para2_cable)
        if backend in ("process", "pool", "fork", "mp", "multiprocess"):
            return self.ADMM_SubP2_process(Para2_cable)
        return self.ADMM_SubP2_mapped(Para2_cable)

    def PrewarmSubP2(self, Ref_xl, Ref_ul, ref_xq, ref_uq, paral, paraC):
        backend = os.environ.get("DIFFCOORD_KINODYN_SUBP2_BACKEND", "process").strip().lower()
        if backend in ("seq", "serial", "sequential", "none"):
            return
        if os.environ.get("DIFFCOORD_KINODYN_SUBP2_SOLVE_PREWARM", "1") != "1":
            if backend in ("process", "pool", "fork", "mp", "multiprocess"):
                self._get_solver2_process_pool()
            else:
                self._get_solver2_map()
            return

        scxl0 = np.asarray(Ref_xl, dtype=np.float64).reshape(-1)
        scul0 = np.asarray(Ref_ul, dtype=np.float64).reshape(-1)
        ref_uq = np.asarray(ref_uq, dtype=np.float64).reshape(-1)
        paral = np.asarray(paral, dtype=np.float64).reshape(-1)
        paraC = np.asarray(paraC, dtype=np.float64).reshape(-1)
        scxc0 = np.zeros((self.N+1)*int(self.nq)*self.nxi)
        for k in range(self.N):
            ref_xc_k = np.zeros(int(self.nq)*self.nxi)
            for i in range(int(self.nq)):
                ref_xi = np.asarray(ref_xq[i], dtype=np.float64).reshape(-1)
                ref_xc_k[i*self.nxi:(i+1)*self.nxi] = ref_xi[k*self.nxi:(k+1)*self.nxi]
            scxc0[k*int(self.nq)*self.nxi:(k+1)*int(self.nq)*self.nxi] = ref_xc_k
        ref_xc_N = np.zeros(int(self.nq)*self.nxi)
        for i in range(int(self.nq)):
            ref_xi = np.asarray(ref_xq[i], dtype=np.float64).reshape(-1)
            ref_xc_N[i*self.nxi:(i+1)*self.nxi] = ref_xi[self.N*self.nxi:(self.N+1)*self.nxi]
        scxc0[self.N*int(self.nq)*self.nxi:(self.N+1)*int(self.nq)*self.nxi] = ref_xc_N

        Para2_cable = np.concatenate((
            scxl0,
            scul0,
            scxc0,
            ref_uq,
            scxl0,
            np.zeros((self.N+1)*self.nxl),
            scul0,
            np.zeros(self.N*self.nul),
            scxc0,
            np.zeros((self.N+1)*int(self.nq)*self.nxi),
            np.tile(ref_uq, self.N),
            np.zeros(self.N*int(self.nq)*self.nui),
            paral,
            paraC,
            [0.0],
        ))
        self.ADMM_SubP2_parallel(Para2_cable)
    

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
        px_dis        = self.open_loop_penalty(px,gammax,i_admm,ADMM_max)
        pu_dis        = self.open_loop_penalty(pu,gammau,i_admm,ADMM_max)
        pix_dis       = self.open_loop_penalty(pix,gammaix,i_admm,ADMM_max)
        piu_dis       = self.open_loop_penalty(piu,gammaiu,i_admm,ADMM_max)
        if jax is not None and os.environ.get("DIFFCOORD_KINODYN_USE_JAX_SUBP3", "0") == "1":
            if self._subp3_update_jit is None:
                def _subp3_step(curr_k, primal_k, safe_k, penalty):
                    return curr_k + penalty * (primal_k - safe_k)
                load_time_update = jax.vmap(
                    _subp3_step,
                    in_axes=(0, 0, 0, None),
                    out_axes=0,
                )
                cable_time_update = jax.vmap(
                    _subp3_step,
                    in_axes=(0, 0, 0, None),
                    out_axes=0,
                )
                cable_update = jax.vmap(
                    cable_time_update,
                    in_axes=(0, 0, 0, None),
                    out_axes=0,
                )
                self._subp3_update_jit = jax.jit(
                    lambda xL, uL, xl, scxl, ul, scul, xC, uC, xc, scxc, uc, scuc, pxv, puv, pixv, piuv: (
                        load_time_update(xL, xl, scxl, pxv),
                        load_time_update(uL, ul, scul, puv),
                        cable_update(xC, xc, scxc, pixv),
                        cable_update(uC, uc, scuc, piuv),
                    )
                )
            scxC_stack = np.stack(scxC_traj, axis=0)
            scuC_stack = np.stack(scuC_traj, axis=0)
            xc_stack = np.stack(xc_traj, axis=0)
            scxc_stack = np.stack(scxc_traj, axis=0)
            uc_stack = np.stack(uc_traj, axis=0)
            scuc_stack = np.stack(scuc_traj, axis=0)
            scxL_traj_new, scuL_traj_new, scxC_new_stack, scuC_new_stack = self._subp3_update_jit(
                jnp.asarray(scxL_traj, dtype=jnp.float64),
                jnp.asarray(scuL_traj, dtype=jnp.float64),
                jnp.asarray(xl_traj, dtype=jnp.float64),
                jnp.asarray(scxl_traj, dtype=jnp.float64),
                jnp.asarray(ul_traj, dtype=jnp.float64),
                jnp.asarray(scul_traj, dtype=jnp.float64),
                jnp.asarray(scxC_stack, dtype=jnp.float64),
                jnp.asarray(scuC_stack, dtype=jnp.float64),
                jnp.asarray(xc_stack, dtype=jnp.float64),
                jnp.asarray(scxc_stack, dtype=jnp.float64),
                jnp.asarray(uc_stack, dtype=jnp.float64),
                jnp.asarray(scuc_stack, dtype=jnp.float64),
                np.float64(px_dis),
                np.float64(pu_dis),
                np.float64(pix_dis),
                np.float64(piu_dis),
            )
            scxL_traj_new = np.asarray(scxL_traj_new, dtype=np.float64)
            scuL_traj_new = np.asarray(scuL_traj_new, dtype=np.float64)
            scxC_new_stack = np.asarray(scxC_new_stack, dtype=np.float64)
            scuC_new_stack = np.asarray(scuC_new_stack, dtype=np.float64)
            scxC_traj_new = [scxC_new_stack[i] for i in range(int(self.nq))]
            scuC_traj_new = [scuC_new_stack[i] for i in range(int(self.nq))]
        else:
            scxL_traj_new = np.asarray(scxL_traj, dtype=np.float64) + px_dis*(np.asarray(xl_traj, dtype=np.float64) - np.asarray(scxl_traj, dtype=np.float64))
            scuL_traj_new = np.asarray(scuL_traj, dtype=np.float64) + pu_dis*(np.asarray(ul_traj, dtype=np.float64) - np.asarray(scul_traj, dtype=np.float64))
            scxC_traj_new = [
                np.asarray(scxC_traj[i], dtype=np.float64) + pix_dis*(np.asarray(xc_traj[i], dtype=np.float64) - np.asarray(scxc_traj[i], dtype=np.float64))
                for i in range(int(self.nq))
            ]
            scuC_traj_new = [
                np.asarray(scuC_traj[i], dtype=np.float64) + piu_dis*(np.asarray(uc_traj[i], dtype=np.float64) - np.asarray(scuc_traj[i], dtype=np.float64))
                for i in range(int(self.nq))
            ]

        opt_sol3 = {"scxL_traj_new":scxL_traj_new,
                    "scuL_traj_new":scuL_traj_new,
                    "scxC_traj_new":scxC_traj_new,
                    "scuC_traj_new":scuC_traj_new
                    }
        
        return opt_sol3
    
    def PrewarmJaxSubP1(self, Ref_xl, Ref_ul, ref_xq, ref_uq, xl_fb, xq_fb, paral, paraC, max_iter_ADMM):
        if jax is None or os.environ.get("DIFFCOORD_KINODYN_PREWARM_JAX_SUBP1", "1") != "1":
            return
        self.max_iter_ADMM = max_iter_ADMM
        max_iter = 10
        e_tol = 1e-2
        scxl_traj_tp = np.zeros(((self.N+1)*self.nxl))
        scul_traj_tp = np.asarray(Ref_ul, dtype=np.float64).reshape(-1)
        scxL_traj_tp = np.zeros(((self.N+1)*self.nxl))
        scuL_traj_tp = np.zeros((self.N*self.nul))
        if getattr(self, "_load_traj_solver_single_jit", None) is not None:
            self.jax_MPC_Load_DDP_Planning_SubP1_arrays(
                xl_fb, Ref_xl, Ref_ul, scxl_traj_tp, scxL_traj_tp,
                scul_traj_tp, scuL_traj_tp, paral, 0
            )
        else:
            self.DDP_Load_ADMM_Subp1(
                xl_fb, Ref_xl, Ref_ul, paral, scxl_traj_tp, scul_traj_tp,
                scxL_traj_tp, scuL_traj_tp, max_iter, e_tol, 0, need_derivs=False
            )
        if os.environ.get("DIFFCOORD_KINODYN_FORWARD_LOAD_DERIVS", "0") == "1":
            self.DDP_Load_ADMM_Subp1(
                xl_fb, Ref_xl, Ref_ul, paral, scxl_traj_tp, scul_traj_tp,
                scxL_traj_tp, scuL_traj_tp, max_iter, e_tol, 0, need_derivs=True
            )
        para_cables = []
        for i in range(int(self.nq)):
            xi_fb = np.asarray(xq_fb[i*self.nxi:(i+1)*self.nxi], dtype=np.float64).reshape(-1)
            ref_xi = np.asarray(ref_xq[i], dtype=np.float64).reshape(-1)
            ref_ui = np.asarray(ref_uq[i*self.nui:(i+1)*self.nui], dtype=np.float64).reshape(-1)
            parai = np.concatenate((
                xi_fb,
                ref_xi,
                ref_ui,
                np.zeros((self.N+1)*self.nxi),
                np.zeros((self.N+1)*self.nxi),
                np.zeros(self.N*self.nui),
                np.zeros(self.N*self.nui),
                np.asarray(paraC, dtype=np.float64).reshape(-1),
                [0.0],
            ))
            para_cables.append(parai)
        self.jax_MPC_Cable_DDP_Planning_SubP1_arrays(
            xq_fb, ref_xq, ref_uq, [np.zeros((self.N+1)*self.nxi) for _ in range(int(self.nq))],
            [np.zeros((self.N+1)*self.nxi) for _ in range(int(self.nq))],
            [np.zeros(self.N*self.nui) for _ in range(int(self.nq))],
            [np.zeros(self.N*self.nui) for _ in range(int(self.nq))],
            paraC, 0
        )
        if os.environ.get("DIFFCOORD_KINODYN_FORWARD_CABLE_DERIVS", "0") == "1":
            self.MPC_Cable_DDP_Planning_SubP1(para_cables, need_derivs=True)
        zeros_lx = np.zeros((self.N+1,self.nxl))
        zeros_lu = np.zeros((self.N,self.nul))
        zeros_cx = [np.zeros((self.N+1,self.nxi)) for _ in range(int(self.nq))]
        zeros_cu = [np.zeros((self.N,self.nui)) for _ in range(int(self.nq))]
        self.ADMM_SubP3(
            zeros_lx, zeros_lx, zeros_lx, zeros_lu, zeros_lu, zeros_lu,
            zeros_cx, zeros_cx, zeros_cx, zeros_cu, zeros_cu, zeros_cu,
            paral[-4], paral[-3], paral[-2], paral[-1],
            paraC[-4], paraC[-3], paraC[-2], paraC[-1],
            max_iter_ADMM, 0
        )
    
    

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
        total_subp1_load_time = 0.0
        total_subp1_cables_time = 0.0
        total_subp1_time = 0.0
        total_subp2_time = 0.0
        total_subp3_time = 0.0
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
        
        subp1_executor = None
        if os.environ.get("DIFFCOORD_KINODYN_SUBP1_CONCURRENT", "1") == "1":
            subp1_executor = ThreadPoolExecutor(max_workers=2)
        for i_admm in range(self.max_iter_ADMM):
            # solve Subproblem 1-load and Subproblem 1-cables in parallel.
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
            need_load_derivs = os.environ.get("DIFFCOORD_KINODYN_FORWARD_LOAD_DERIVS", "0") == "1"
            need_cable_derivs = os.environ.get("DIFFCOORD_KINODYN_FORWARD_CABLE_DERIVS", "0") == "1"

            def solve_load_subp1():
                _disable_debugger_trace_in_worker()
                start_time = TM.time()
                if not need_load_derivs and getattr(self, "_load_traj_solver_single_jit", None) is not None:
                    opt_sol_load, subp1_inner_ms = self.jax_MPC_Load_DDP_Planning_SubP1_arrays(
                        xl_fb, Ref_xl, Ref_ul, scxl_traj_tp, scxL_traj_tp,
                        scul_traj_tp, scuL_traj_tp, paral, i_admm,
                        return_timing=True,
                    )
                else:
                    opt_sol_load = self.DDP_Load_ADMM_Subp1(
                        xl_fb, Ref_xl, Ref_ul, paral, scxl_traj_tp, scul_traj_tp,
                        scxL_traj_tp, scuL_traj_tp, max_iter, e_tol, i_admm,
                        need_derivs=need_load_derivs,
                    )
                    subp1_inner_ms = (TM.time() - start_time) * 1000
                return opt_sol_load, subp1_inner_ms

            def solve_cable_subp1():
                _disable_debugger_trace_in_worker()
                start_time = TM.time()
                if need_cable_derivs:
                    opt_sol_cable, opt_detail_cable = self.MPC_Cable_DDP_Planning_SubP1(ParaC, need_derivs=True)
                    subp1_inner_ms = (TM.time() - start_time) * 1000
                else:
                    opt_sol_cable, opt_detail_cable, subp1_inner_ms = self.jax_MPC_Cable_DDP_Planning_SubP1_arrays(
                        xq_fb, ref_xq, ref_uq, scxc_traj_tp, scxC_traj_tp,
                        scuc_traj_tp, scuC_traj_tp, paraC, i_admm,
                        return_timing=True,
                    )
                return opt_sol_cable, opt_detail_cable, subp1_inner_ms

            if subp1_executor is not None:
                with _no_debugger_trace_during_fork():
                    load_future = subp1_executor.submit(solve_load_subp1)
                    cable_future = subp1_executor.submit(solve_cable_subp1)
                opt_sol_l, load_ms = load_future.result()
                opt_solc, OPt_sol_c, cable_ms = cable_future.result()
            else:
                opt_sol_l, load_ms = solve_load_subp1()
                opt_solc, OPt_sol_c, cable_ms = solve_cable_subp1()
            if subp1_executor is not None:
                total_subp1_time += max(load_ms, cable_ms)
            else:
                total_subp1_time += load_ms + cable_ms
            total_subp1_load_time += load_ms
            total_subp1_cables_time += cable_ms
            xl_traj = opt_sol_l['xl_traj']
            ul_traj = opt_sol_l['ul_traj']
            Kfbl_traj  = opt_sol_l['K_FB']
            xl_traj_tp = np.reshape(xl_traj,(self.N+1)*self.nxl)
            ul_traj_tp = np.reshape(ul_traj,self.N*self.nul)
            px_dis      = self.open_loop_penalty(paral[-4],paral[-2],i_admm,max_iter_ADMM)
            pu_dis      = self.open_loop_penalty(paral[-3],paral[-1],i_admm,max_iter_ADMM)
            pix_dis     = self.open_loop_penalty(paraC[-4],paraC[-2],i_admm,max_iter_ADMM)
            piu_dis     = self.open_loop_penalty(paraC[-3],paraC[-1],i_admm,max_iter_ADMM)
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
            opt_sol2    = self.ADMM_SubP2_parallel(Para2_cable)
            mpctime = (TM.time() - start_time)*1000
            total_subp2_time += mpctime
            scxl_traj   = opt_sol2['scxl_traj']
            scul_traj   = opt_sol2['scul_traj']
            scxc_traj   = opt_sol2['scxc_traj']
            scuc_traj   = opt_sol2['scuc_traj']
            # solve Subproblem 3
            start_time  = TM.time()
            opt_sol3    = self.ADMM_SubP3(xl_traj,scxl_traj,scxL_traj,ul_traj,scul_traj,scuL_traj,xc_traj,scxc_traj,scxC_traj,uc_traj,scuc_traj,scuC_traj,paral[-4],paral[-3],paral[-2],paral[-1],paraC[-4],paraC[-3],paraC[-2],paraC[-1],max_iter_ADMM,i_admm)
            total_subp3_time += (TM.time() - start_time)*1000
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
        if subp1_executor is not None:
            subp1_executor.shutdown(wait=True)
        
        total_subproblem_time = total_subp1_time + total_subp2_time + total_subp3_time
        total_forward_time = total_subproblem_time
        self.last_subp1_load_time_ms = total_subp1_load_time
        self.last_subp1_cables_time_ms = total_subp1_cables_time
        self.last_subp1_time_ms = total_subp1_time
        self.last_subp2_time_ms = total_subp2_time
        self.last_subp3_time_ms = total_subp3_time
        self.last_subproblem_time_ms = total_subproblem_time
        self.last_forward_inner_time_ms = total_forward_time
        print("total subprblem1_load:--- %s ms ---" % format(total_subp1_load_time,'.2f'))
        print("total subprblem1_cables:--- %s ms ---" % format(total_subp1_cables_time,'.2f'))
        print("total subprblem1:--- %s ms ---" % format(total_subp1_time,'.2f'))
        print("total subprblem2:--- %s ms ---" % format(total_subp2_time,'.2f'))
        print("total subprblem3:--- %s ms ---" % format(total_subp3_time,'.2f'))
        print("total subproblems:--- %s ms ---" % format(total_subproblem_time,'.2f'))
        print("total forward inner:--- %s ms ---" % format(total_forward_time,'.2f'))
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
    
    def adaptive_meta_loss_weights(self,loss_t,loss_rp,wt,mu_up=1.05,mu_down=1.05,tau_t=1.25,tau_rp=1.25, beta_s=0.6):
        if loss_t >mu_up*loss_rp:
            wt_newc = np.clip(tau_t*wt,0.2,5)
        elif loss_rp > mu_down*loss_t:
            wt_newc = np.clip(wt/tau_rp,0.2,5)
        else:
            wt_newc = wt
        wt_new = (1-beta_s)*wt + beta_s*wt_newc
     
        return wt_new

     
    # def adaptive_meta_loss_weights(self,loss_t,loss_rp,wrp,mu_up=1.25,mu_down=1.05,tau_up=1.2,tau_down=2, beta_s=0.6):
    #     if loss_rp >mu_up*loss_t:
    #         wrp_newc = np.clip(tau_up*wrp,0.1,1.9)
    #     elif loss_t > mu_down*loss_rp:
    #         wrp_newc = np.clip(wrp/tau_down,0.1,1.9)
    #     else:
    #         wrp_newc = wrp
    #     wrp_new      = (1-beta_s)*wrp + beta_s*wrp_newc
    #     wt_new       = 2-wrp_new
    #     return wt_new,wrp_new


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

        return dldw, loss, loss_track, loss_resid
  














    





        

    

                


                    







    


        



    


    

    

    

        
        
            





    
