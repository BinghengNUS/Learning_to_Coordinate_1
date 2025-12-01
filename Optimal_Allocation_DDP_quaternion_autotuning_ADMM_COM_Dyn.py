from casadi import *
import numpy as np
from numpy import linalg as LA
import math
from scipy.spatial.transform import Rotation as Rot
from scipy import linalg as sLA
from scipy.linalg import null_space
import time as TM

class MPC_Planner:
    def __init__(self, sysm_para, dt_ctrl, horizon, e_abs, e_rel):
        # Payload's parameters
        self.ml     = sysm_para[0] # the payload's mass [kg]
        self.m2     = sysm_para[1] # the added mass [kg]
        self.rl     = sysm_para[5] # the radius of load [m]
        # Quadrotor's parameters
        self.nq     = sysm_para[6] # the number of quadrotors
        self.rq     = sysm_para[7] # the radius of quadrotor [m]
        # Cable and obstacle's parameters
        self.cl0    = sysm_para[8] # the cable length [m]
        self.ro     = sysm_para[9] # the radius of obstacle [m]
        # Unit direction vector free of coordinate
        self.ex     = np.array([[1, 0, 0]]).T
        self.ey     = np.array([[0, 1, 0]]).T
        self.ez     = np.array([[0, 0, 1]]).T
        # Gravitational acceleration
        self.g      = 9.81      
        self.dt     = dt_ctrl
        # MPC's horizon
        self.N      = horizon
        # Tolerances used in ADMM
        self.e_abs  = e_abs
        self.e_rel  = e_rel
        # barrier parameter
        self.p_bar  = 1e-6

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
        self.P_pinv = self.Pt.T@LA.inv(self.Pt@self.Pt.T) # pseudo-inverse of P
        self.P_ns   = null_space(self.Pt) # null-space of P, 3nq-by-(3nq-6)

    def skew_sym_numpy(self, v):
        v_cross = np.array([
            [0, -v[2, 0], v[1, 0]],
            [v[2, 0], 0, -v[0, 0]],
            [-v[1, 0], v[0, 0], 0]]
        )
        return v_cross

    def SetStateVariable(self, xl):
        self.xl    = xl # payload's state
        self.n_xl  = xl.numel() # 12
        self.xl_lb = self.n_xl*[-1e19]# state constraint (infinity)
        self.xl_ub = self.n_xl*[1e19]
        self.sc_xl = SX.sym('sc_xl',self.n_xl,1) # safe copy state
        self.sc_xL = SX.sym('sc_xL',self.n_xl,1) # Lagrangian multiplier associated with the safe copy state

    def SetCtrlVariable(self, Wl):
        self.Wl    = Wl # load's control, 6-by-1 vector, wrench in the load body frame
        self.n_Wl  = Wl.numel()
        self.sc_Wl = SX.sym('sc_Wl',self.n_Wl,1) # safe copy control
        self.sc_WL = SX.sym('sc_WL',self.n_Wl,1) # Lagrangian multiplier associated with the safe copy control
        self.nv    = SX.sym('nv',3*int(self.nq)-6,1) # null-space vector
        self.n_nv  = self.nv.numel()
        self.ul    = vertcat(self.Wl,self.nv)
        self.n_ul  = Wl.numel() + self.nv.numel() # including the dimension of the null-space vector, totally 3nq
        self.Wl_lb = self.n_Wl*[-1e19] #Wl_lb # having the appropriate box constraints on control is very important
        self.Tm_lb = (3*int(self.nq)-6)*[-1e19]
        self.Wl_ub = self.n_Wl*[1e19] #Wl_ub
        self.Tm_ub = (3*int(self.nq)-6)*[1e19]

    def SetDyn(self, model_l):
        self.Modell   = self.xl + self.dt*model_l # 4th-order Runge-Kutta discrete-time dynamics model
        self.MDynl_fn = Function('MDynl',[self.xl, self.ul],[self.Modell],['xl0', 'ul0'],['MDynlf'])
        self.MDynl_fn_admm = Function('MDynl_admm',[self.xl,self.Wl],[self.Modell],['xl0','Wl0'],['MDynlf_admm'])

    def dir_cosine(self, Euler):
        # Euler angles for roll, pitch and yaw
        roll, pitch, yaw = Euler[0,0], Euler[1,0], Euler[2,0]
        # below rotation matrice are used to convert a vector from body frame to world frame
        Rx  = vertcat(
            horzcat(1,0,0),
            horzcat(0,cos(roll),-sin(roll)),
            horzcat(0,sin(roll),cos(roll))
        ) # rotation about x axis that converts a vector in {B} to {I}
        Ry  = vertcat(
            horzcat(cos(pitch),0,sin(pitch)),
            horzcat(0,1,0),
            horzcat(-sin(pitch),0,cos(pitch))
        ) # rotation about y axis that converts a vector in {B} to {I}
        Rz  = vertcat(
            horzcat(cos(yaw),-sin(yaw),0),
            horzcat(sin(yaw),cos(yaw),0),
            horzcat(0,0,1)
        ) # rotation about z axis that converts a vector in {B} to {I}
        # 3-2-1 rotation sequence that rotates the basis of {I} to the basis of {B}.
        # In other words, a body frame is obtained by rotating {I} through the 3-2-1 rotation sequence
        R_bw = Rz@Ry@Rx       # rotation matrix that transfers a vector in {B} to {I}

        return R_bw
    
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
    
    def SetLearnablePara(self):
        self.n_wsl    = self.n_xl # the dimension of the payload state weightings
        self.p1       = SX.sym('p1') # the penalty parameter 1 in ADMM
        self.P1_l     = SX.sym('P1_l',1,(2*self.n_wsl+self.n_Wl)) # the hyperparameters of Subproblem 1 in ADMM
        self.n_P1     = self.P1_l.numel()
        self.P2_l     = SX.sym('P2',1,3) # the weights of the cable tension deviation
        self.n_P2     = self.P2_l.numel()
        self.P_auto   = horzcat(self.P1_l,self.P2_l,self.p1) # the total learnable parameters
        self.n_Pauto  = self.P_auto.numel()

    
    def SetCostDyn_ADMM(self):
        self.ref_xl   = SX.sym('ref_xl',self.n_xl,1)
        self.ref_Wl   = SX.sym('ref_Wl',self.n_Wl,1)
        self.a        = SX.sym('a',1) # the ADMM iteration index
        scql          = self.sc_xl[6:10,0]
        track_error_l = self.xl - self.ref_xl
        ctrl_error_l  = self.Wl - self.ref_Wl
        self.dis_r    = self.Discount_rate(self.a) # iteration-specific discount rate, a global variable within the class
        self.p1_dis   = self.dis_r*self.p1
        xl_const_v    = self.xl - self.sc_xl + self.sc_xL/self.p1_dis # the state constraint violation
        Wl_const_v    = self.Wl - self.sc_Wl + self.sc_WL/self.p1_dis # the control constraint violation
        self.Ql_k     = diag(self.P1_l[0,0:self.n_wsl])
        self.Ql_N     = diag(self.P1_l[0,self.n_wsl:2*self.n_wsl])
        self.Rl_k     = diag(self.P1_l[0,2*self.n_wsl:2*self.n_wsl+self.n_Wl])
        # path cost of Subproblem 1 (MPC)
        self.Jl_k_bar_admm = 1/2 * (track_error_l.T@self.Ql_k@track_error_l + ctrl_error_l.T@self.Rl_k@ctrl_error_l) #+ self.h_norm
        self.Jl_k_hat_admm = self.p1_dis/2*xl_const_v.T@xl_const_v + self.p1_dis/2*Wl_const_v.T@Wl_const_v 
        self.Jl_k_admm     = self.Jl_k_bar_admm + self.Jl_k_hat_admm 
        self.Jl_kfn_admm   = Function('Jl_k_admm',[self.xl, self.Wl, self.ref_xl, self.ref_Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1, self.a],[self.Jl_k_admm],['xl0', 'Wl0', 'refxl0', 'refWl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P1l0', 'p10', 'a0'],['Jl_kf_admm'])
        # terminal cost of Subproblem 1 (MPC)
        self.Jl_N_admm     = 1/2 * track_error_l.T@self.Ql_N@track_error_l 
        self.Jl_Nfn_admm   = Function('Jl_N_admm',[self.xl, self.ref_xl, self.P1_l],[self.Jl_N_admm],['xl0', 'refxl0', 'P1l0'],['Jl_Nf_admm'])
        # path cost of Subproblem 2 (Static optimization)
        scFl               = self.sc_Wl[0:3]
        scMl               = self.sc_Wl[3:6]
        scRl               = self.q_2_rotation(scql)
        Wlb                = vertcat(scRl.T@scFl,scMl) 
        t                  = self.P_pinv@Wlb + self.P_ns@self.nv
        self.J2_k          = 0
        self.ref_ti        = vertcat(0,0,self.ml*self.g/self.nq)
        self.R2_k          = diag(self.P2_l)
        for i in range(int(self.nq)):
            ti   = t[(3*i):(3*(i+1))]
            tension_error = ti-self.ref_ti
            self.J2_k    += 1/2 *tension_error.T@self.R2_k@tension_error
        self.J2_k_admm     = self.J2_k + self.p1_dis/2*xl_const_v.T@xl_const_v + self.p1_dis/2*Wl_const_v.T@Wl_const_v 
        self.J2_kfn_admm   = Function('J2_k_admm',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1, self.a],[self.J2_k_admm],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10', 'a0'],['J2_kf_admm'])
        self.J2_k_soft = self.J2_k_admm + self.T_cons + self.G_ij + self.G_io + self.sch_norm
        self.J2_k_orig = self.J2_k + self.T_cons + self.G_ij + self.G_io + self.sch_norm


    def system_derivatives_DDP_ADMM(self):
        self.Vx     = SX.sym('Vx',self.n_xl)
        self.Vxx    = SX.sym('Vxx',self.n_xl,self.n_xl)
        # gradients of the system dynamics, the cost function, and the Q value function
        self.Fx     = jacobian(self.Modell,self.xl)
        self.Fx_fn  = Function('Fx',[self.xl,self.Wl],[self.Fx],['xl0','Wl0'],['Fx_f'])
        self.Fu     = jacobian(self.Modell,self.Wl)
        self.Fu_fn  = Function('Fu',[self.xl,self.Wl],[self.Fu],['xl0','Wl0'],['Fu_f'])
        self.lx     = jacobian(self.Jl_k_admm,self.xl)
        self.lxN    = jacobian(self.Jl_N_admm,self.xl)
        self.lxN_fn = Function('lxN',[self.xl,self.ref_xl,self.P1_l],[self.lxN],['xl0','refxl0','P1l0'],['lxNf'])
        self.lu     = jacobian(self.Jl_k_admm,self.Wl)
        self.Qx     = self.lx.T + self.Fx.T@self.Vx
        self.Qx_fn  = Function('Qx',[self.xl,self.Wl,self.Vx,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1, self.a],[self.Qx],['xl0','Wl0','Vx0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10','a0'],['Qxf'])
        self.Qu     = self.lu.T + self.Fu.T@self.Vx
        self.Qu_fn  = Function('Qu',[self.xl,self.Wl,self.Vx,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1, self.a],[self.Qu],['xl0','Wl0','Vx0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10','a0'],['Quf'])
        # hessians of the system dynamics, the cost function, and the Q value function
        self.FxVx   = self.Fx.T@self.Vx
        self.dFxVxdx= jacobian(self.FxVx,self.xl) # the hessian of the system dynamics may cause heavy computational burden
        self.dFxVxdu= jacobian(self.FxVx,self.Wl)
        self.FuVx   = self.Fu.T@self.Vx
        self.dFuVxdu= jacobian(self.FuVx,self.Wl)
        self.lxx    = jacobian(self.lx,self.xl)
        self.lxxN   = jacobian(self.lxN,self.xl)
        self.lxxN_fn= Function('lxxN',[self.P1_l],[self.lxxN],['P1l0'],['lxxNf'])
        self.lxu    = jacobian(self.lx,self.Wl)
        self.luu    = jacobian(self.lu,self.Wl)
        self.Qxx_bar    = self.lxx #+ alpha*self.dFxVxdx  # removing this model hessian can enhance the DDP stability for a larger time step!!!! The removal can also accelerate the DDP computation significantly!
        self.Qxx_bar_fn = Function('Qxx_bar',[self.xl,self.Wl,self.Vx,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1, self.a],[self.Qxx_bar],['xl0','Wl0','Vx0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10','a0'],['Qxx_bar_f'])
        self.Qxx_hat    = self.Fx.T@self.Vxx@self.Fx
        self.Qxx_hat_fn = Function('Qxx_hat',[self.xl,self.Wl,self.Vxx],[self.Qxx_hat],['xl0','Wl0','Vxx0'],['Qxx_hat_f'])
        self.Qxu_bar    = self.lxu #+ alpha*self.dFxVxdu  # including the model hessian entails a very small time step size (e.g., 0.01s)
        self.Qxu_bar_fn = Function('Qxu_bar',[self.xl,self.Wl,self.Vx,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1, self.a],[self.Qxu_bar],['xl0','Wl0','Vx0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10','a0'],['Qxu_bar_f'])
        self.Qxu_hat    = self.Fx.T@self.Vxx@self.Fu
        self.Qxu_hat_fn = Function('Qxu_hat',[self.xl,self.Wl,self.Vxx],[self.Qxu_hat],['xl0','Wl0','Vxx0'],['Qxu_hat_f'])
        self.Quu_bar    = self.luu #+ alpha*self.dFuVxdu
        self.Quu_bar_fn = Function('Quu_bar',[self.xl,self.Wl,self.Vx,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1, self.a],[self.Quu_bar],['xl0','Wl0','Vx0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10','a0'],['Quu_bar_f'])
        self.Quu_hat    = self.Fu.T@self.Vxx@self.Fu 
        self.Quu_hat_fn = Function('Quu_hat',[self.xl,self.Wl,self.Vxx],[self.Quu_hat],['xl0','Wl0','Vxx0'],['Quu_hat_f'])
        # hessians w.r.t. the hyperparameters
        self.lxp    = jacobian(self.lx,self.P_auto)
        self.lxp_fn = Function('lxp',[self.xl,self.Wl,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1, self.a],[self.lxp],['xl0','Wl0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10','a0'],['lxpf'])
        self.lup    = jacobian(self.lu,self.P_auto)
        self.lup_fn = Function('lup',[self.xl,self.Wl,self.ref_xl,self.ref_Wl,self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P1_l, self.p1, self.a],[self.lup],['xl0','Wl0','refxl0','refWl0','scxl0','scxL0','scWl0','scWL0','P1l0','p10','a0'],['lupf'])
        self.lxNp   = jacobian(self.lxN,self.P_auto)
        self.lxNp_fn= Function('lxNp',[self.xl,self.ref_xl,self.P1_l],[self.lxNp],['xl0','refxl0','P1l0'],['lxNpf'])
    

    def Get_AuxSys_DDP(self,opt_sol1,Ref_xl,Ref_Wl,scxl_opt,scWl_opt,Y_l,Eta_l,weight1,p1,i_admm):
        xl_opt   = opt_sol1['xl_opt']
        Wl_opt   = opt_sol1['Wl_opt']
        LxNp  = self.lxNp_fn(xl0=xl_opt[-1,:],refxl0=Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl],P1l0=weight1)['lxNpf'].full()
        LxxN  = self.lxxN_fn(P1l0=weight1)['lxxNf'].full()
        Lxp = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        Lup = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        for k in range(self.N):
            Lxp[k] = self.lxp_fn(xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                scxl0=scxl_opt[k,:],scxL0=Y_l[:,k],
                                scWl0=scWl_opt[k,:],scWL0=Eta_l[:,k],P1l0=weight1,p10=p1,a0=i_admm)['lxpf'].full()
            Lup[k] = self.lup_fn(xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                scxl0=scxl_opt[k,:],scxL0=Y_l[:,k],
                                scWl0=scWl_opt[k,:],scWL0=Eta_l[:,k],P1l0=weight1,p10=p1,a0=i_admm)['lupf'].full()
        
        auxSys1 = { "HxxN":LxxN,
                    "HxNp":LxNp,
                    "Hxp":Lxp,
                    "Hup":Lup
                    }
        
        return auxSys1

    def symmetry(self,A):
        return 0.5*(A+A.T)

    def chol_solve(self,L, B):
        # Solve (L L^T) X = B
        Y = LA.solve(L, B)
        return LA.solve(L.T, Y)

    def try_cholesky(self,A, jitter0=0.0, max_tries=10):
        """Try Cholesky with growing jitter on the diagonal."""
        jitter = jitter0
        for _ in range(max_tries):
            try:
                return LA.cholesky(A + jitter*np.eye(A.shape[0])), jitter
            except LA.LinAlgError:
                jitter = max(1e-12, 10*(jitter if jitter>0 else 1e-12))
        raise LA.LinAlgError("Cholesky failed even with jitter")


    def DDP_ADMM_Subp1(self,xl_0,Ref_xl,Ref_Wl,weight1,scxl_opt_hat,scWl_opt_hat,Y_l,Eta_l,p1,max_iter,e_tol,i,i_admm):
        reg          = 1e-6 # Regularization term
        reg_max      = 1    # cap to avoid runaway
        reg_up       = 10.0 # how much to bump when ill-conditioned
        alpha_init   = 1 # Initial alpha for line search
        alpha_min    = 1e-2  # Minimum allowable alpha
        alpha_factor = 0.5 # 
        max_line_search_steps = 5
        iteration = 1
        ratio = 10
        X_nominal = np.zeros((self.n_xl,self.N+1))
        U_nominal = np.zeros((self.n_Wl,self.N))
        X_nominal[:,0:1] = np.reshape(xl_0,(self.n_xl,1))
        
        # Initial trajectory and initial cost 
        cost_prev = 0
        if i ==0:
            for k in range(self.N):
                u_k    = np.reshape(Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],(self.n_Wl,1))
                X_nominal[:,k+1:k+2] = self.MDynl_fn_admm(xl0=X_nominal[:,k],Wl0=u_k)['MDynlf_admm'].full()
                U_nominal[:,k:k+1]   = u_k
                cost_prev     += self.Jl_kfn_admm(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                             scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                             scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1,a0=i_admm)['Jl_kf_admm'].full()
            cost_prev += self.Jl_Nfn_admm(xl0=X_nominal[:,-1],refxl0=Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl],P1l0=weight1)['Jl_Nf_admm'].full()
        else:
            for k in range(self.N):
                X_nominal[:,k+1:k+2] = np.reshape(scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],(self.n_xl,1))
                U_nominal[:,k:k+1]   = np.reshape(scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],(self.n_Wl,1))
                cost_prev     += self.Jl_kfn_admm(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                             scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                             scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],p10=p1,a0=i_admm)['Jl_kf_admm'].full()
            cost_prev += self.Jl_Nfn_admm(xl0=X_nominal[:,-1],refxl0=Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl],P1l0=weight1)['Jl_Nf_admm'].full()

        
        Qxx_bar     = self.N*[np.zeros((self.n_xl,self.n_xl))]
        Qxu_bar     = self.N*[np.zeros((self.n_xl,self.n_Wl))]
        Quu_bar     = self.N*[np.zeros((self.n_Wl,self.n_Wl))]
        Qxu         = self.N*[np.zeros((self.n_xl,self.n_Wl))]
        Quuinv      = self.N*[np.zeros((self.n_Wl,self.n_Wl))]
        Fx          = self.N*[np.zeros((self.n_xl,self.n_xl))]
        Fu          = self.N*[np.zeros((self.n_xl,self.n_Wl))]
        Vx          = (self.N+1)*[np.zeros((self.n_xl,1))]
        Vxx         = (self.N+1)*[np.zeros((self.n_xl,self.n_xl))]
        K_fb        = self.N*[np.zeros((self.n_Wl,self.n_xl))] # feedback
        k_ff        = self.N*[np.zeros((self.n_Wl,1))] # feedforward
        Qu_2        = 1000
        I_u         = np.identity(self.n_Wl)
        while Qu_2>e_tol and iteration<=max_iter:
            Vx[self.N] = self.lxN_fn(xl0=X_nominal[:,self.N],refxl0=Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl],P1l0=weight1)['lxNf'].full()
            Vxx[self.N]= self.lxxN_fn(P1l0=weight1)['lxxNf'].full()
            # backward pass
            Qu_2    = 0
            chol_failed = False
            for k in reversed(range(self.N)): # N-1, N-2,...,0
                Qx_k  = self.Qx_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                    scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1,a0=i_admm)['Qxf'].full()
                Qu_k  = self.Qu_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                    scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1,a0=i_admm)['Quf'].full()
                Qxx_bar_k = self.Qxx_bar_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                    scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1,a0=i_admm)['Qxx_bar_f'].full()
                Qxx_hat_k = self.Qxx_hat_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vxx0=Vxx[k+1])['Qxx_hat_f'].full()
                Qxx_k     = Qxx_bar_k + Qxx_hat_k
                Qxu_bar_k = self.Qxu_bar_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                    scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1,a0=i_admm)['Qxu_bar_f'].full()
                Qxu_hat_k = self.Qxu_hat_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vxx0=Vxx[k+1])['Qxu_hat_f'].full()
                Qxu_k     = Qxu_bar_k + Qxu_hat_k
                Quu_bar_k = self.Quu_bar_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vx0=Vx[k+1],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                    scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                    scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1,a0=i_admm)['Quu_bar_f'].full()
                Quu_hat_k = self.Quu_hat_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k],Vxx0=Vxx[k+1])['Quu_hat_f'].full()
                Quu_k     = Quu_bar_k + Quu_hat_k
                Quu_reg_k = Quu_k + reg*I_u
                L, _jitter   = self.try_cholesky(Quu_reg_k, jitter0=0.0)
                if L is None:
                    chol_failed = True
                    break
                Quu_inv      = self.chol_solve(L, I_u) # only for computing the gradients
                k_ff[k]      = self.chol_solve(L, -Qu_k)
                K_fb[k]      = self.chol_solve(L, -Qxu_k.T)
                Vx[k]        = Qx_k + Qxu_k @ k_ff[k]
                Vxx[k]       = self.symmetry(Qxx_k + Qxu_k @ K_fb[k])
                Fx[k]    = self.Fx_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k])['Fx_f'].full()
                Fu[k]    = self.Fu_fn(xl0=X_nominal[:,k],Wl0=U_nominal[:,k])['Fu_f'].full()
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
            alpha    = alpha_init
            accepted = False
            for _ in range(max_line_search_steps):
                X_new = np.zeros((self.n_xl,self.N+1))
                U_new = np.zeros((self.n_Wl,self.N))
                X_new[:,0:1] = np.reshape(xl_0,(self.n_xl,1))
                cost_new = 0
                for k in range(self.N):
                    delta_x = np.reshape(X_new[:,k] - X_nominal[:,k],(self.n_xl,1))
                    u_1     = np.reshape(U_nominal[:,k],(self.n_Wl,1))
                    u_2     = K_fb[k]@delta_x
                    u_3     = alpha*k_ff[k]
                    u_k     = u_1 + u_2 + u_3
                    u_k     = np.reshape(u_k,(self.n_Wl,1))
                    X_new[:,k+1:k+2]  = self.MDynl_fn_admm(xl0=np.reshape(X_new[:,k],(self.n_xl,1)),Wl0=u_k)['MDynlf_admm'].full()
                    U_new[:,k:k+1]    = u_k
                    cost_new   += self.Jl_kfn_admm(xl0=X_new[:,k],Wl0=u_k,refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl],refWl0=Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],
                                             scxl0=scxl_opt_hat[k*self.n_xl:(k+1)*self.n_xl],scxL0=Y_l[k*self.n_xl:(k+1)*self.n_xl],
                                             scWl0=scWl_opt_hat[k*self.n_Wl:(k+1)*self.n_Wl],scWL0=Eta_l[k*self.n_Wl:(k+1)*self.n_Wl],P1l0=weight1,p10=p1,a0=i_admm)['Jl_kf_admm'].full()
                cost_new   += self.Jl_Nfn_admm(xl0=X_new[:,-1],refxl0=Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl],P1l0=weight1)['Jl_Nf_admm'].full()
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
        
        opt_sol={"xl_opt":X_nominal.T,
                 "Wl_opt":U_nominal.T,
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

    def SetConstraints_ADMM_Subp2(self, pob1, pob2):
        pl   = self.sc_xl[0:3]
        ql   = self.sc_xl[6:10]
        scFl = self.sc_Wl[0:3]
        scMl = self.sc_Wl[3:6]
        Rl   = self.q_2_rotation(ql)
        Wlb  = vertcat(Rl.T@scFl,scMl)
        ql_knorm = ql.T@ql
        self.sch_norm = 1/(2*self.p_bar)*(ql_knorm-1)**2
        self.ql_fn    = Function('norm_ql',[self.sc_xl],[ql_knorm],['scxl0'],['norm_qlf'])
        t    = self.P_pinv@Wlb + self.P_ns@self.nv # 3nq-by-1 total tension vector in the load's body frame, at the kth step
        Pi   = [] # list that stores all the quadrotors' planar positions in the world frame
        Pil  = [] # list that stores all the quadrotors' planar positions in the body frame
        self.Gij_admm   = [] # list that stores all the safe inter-robot inequality constraints
        self.Gi1_admm   = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 1st obstacle
        self.Gi2_admm   = [] # list that stores the obstacle-avoidance constraints of all the quadrotors for the 2nd obstacle
        self.Ti_admm    = [] # list that stores the tension magnitudes
        self.T_cons = 0 # barrier functions of box constraints on the tension magnitude
        for i in range(int(self.nq)):
            ri   = np.reshape(self.ra[:,i],(3,1)) # ith attachment point in the load's body frame
            ti   = t[(3*i):(3*(i+1))] # ith tension, a 3-by-1 vector, in the load's body frame
            pi   = pl + Rl@(ri+self.cl0*ti/norm_2(ti)) # ith quadrotor's position in the world frame
            pil  = ri+self.cl0*ti/norm_2(ti) # load body frame
            Pi  += [pi[0:2]]
            Pil += [pil[0:2]]
            ti_m = ti.T@ti
            ti_fn= Function('t_admm'+str(i),[self.sc_xl,self.sc_Wl,self.nv],[ti_m],['scxl0','scWl0','tm0'],['t'+str(i)+'f_admm'])
            self.t_min = 0.01
            self.t_max = 10
            self.T_cons += -self.p_bar * log(ti_m-self.t_min**2)
            self.T_cons += -self.p_bar * log(self.t_max**2-ti_m)
            self.Ti_admm += [ti_fn]
        k = 0
        self.G_ij    = 0 # barrier functions of safe inter-robot constraints on quadrotors' planar positions
        self.G_io    = 0 # barrier functions of safe collision-avoidance constraints on quadrotors' planar positions 
        
        for i in range(int(self.nq)):
            go1   = (Pi[i]-pob1).T@(Pi[i]-pob1) - ((self.rq + self.ro)+self.rq/2)**2 # safe constriant between the obstacle 1 and the ith quadrotor, which should be positive
            go1_fn= Function('go1_admm'+str(i),[self.sc_xl,self.sc_Wl,self.nv],[go1],['scxl0','scWl0','tm0'],['go1'+str(i)+'f_admm'])
            self.G_io += -self.p_bar * log(go1)
            self.Gi1_admm  += [go1_fn]
            go2   = (Pi[i]-pob2).T@(Pi[i]-pob2) - ((self.rq + self.ro)+self.rq/2)**2 # safe constriant between the obstacle 2 and the ith quadrotor, which should be positive
            go2_fn= Function('go2_admm'+str(i),[self.sc_xl,self.sc_Wl,self.nv],[go2],['scxl0','scWl0','tm0'],['go2'+str(i)+'f_admm'])
            self.G_io += -self.p_bar * log(go2)
            self.Gi2_admm  += [go2_fn]

            for j in range(i+1,int(self.nq)):
                gij   = (Pil[i]-Pil[j]).T@(Pil[i]-Pil[j])- (4*self.rq)**2 # safe constraint between two quadrotors
                gij_fn= Function('g_admm'+str(k),[self.sc_xl,self.sc_Wl,self.nv],[gij],['scxl0','scWl0','tm0'],['g'+str(k)+'f_admm'])
                self.G_ij += -self.p_bar * log(gij)
                self.Gij_admm  += [gij_fn]
                k += 1

    
    def ADMM_SubP2_Init(self):
        # start with an empty NLP
        w        = []  # optimal trajectory list
        self.lbw2 = [] # lower boundary list of optimal variables
        self.ubw2 = [] # upper boundary list of optimal variables
        g        = []  # equality and inequality constraint list
        self.lbg2 = [] # lower boundary list of constraints
        self.ubg2 = [] # upper boundary list of constraints
        
        # hyperparameters + external signals
        P2l      = SX.sym('P2', (self.n_P2 # the hyperparameters of Subproblem2
                                +self.n_xl # the state of subproblem 1 at step k
                                +self.n_xl # the Lagrangian multiplier associated with the state of subproblem 1 at step k
                                +self.n_Wl # the control of subproblem 1 at step k
                                +self.n_Wl # the Lagrangian multiplier associated with the control of subproblem 1 at step k
                                +1 # the ADMM penalty parameter
                                +1 # the ADMM iteration index
                                ))
        
        # formulate the NLP
        P2_l     = P2l[0:self.n_P2]
        p1       = P2l[self.n_P2+2*self.n_xl+2*self.n_Wl]
        i_admm   = P2l[-1]
        Xk          = SX.sym('x',self.n_xl,1) # safe-copy state
        w          += [Xk]
        self.lbw2  += self.xl_lb
        self.ubw2  += self.xl_ub
        xl_k        = P2l[(self.n_P2):(self.n_P2+self.n_xl)]
       
        Wk          = SX.sym('w',self.n_Wl,1) # safe-copy control
        w          += [Wk]
        self.lbw2  += self.Wl_lb
        self.ubw2  += self.Wl_ub
        Wl_k        = P2l[(self.n_P2+2*self.n_xl):(self.n_P2+2*self.n_xl+self.n_Wl)]
       
        Tk          = SX.sym('tm',self.n_nv,1) # null-space vector, tension modifier
        w          += [Tk]
        self.lbw2  += self.Tm_lb
        self.ubw2  += self.Tm_ub
        xL_k        = P2l[(self.n_P2+self.n_xl):(self.n_P2+self.n_xl+self.n_xl)]
        WL_k        = P2l[(self.n_P2+2*self.n_xl+self.n_Wl):(self.n_P2+2*self.n_xl+self.n_Wl+self.n_Wl)]
        J           = self.J2_kfn_admm(nv0=Tk,xl0=xl_k,Wl0=Wl_k,scxl0=Xk,scxL0=xL_k,scWl0=Wk,scWL0=WL_k,P2l0=P2_l,p10=p1,a0=i_admm)['J2_kf_admm']
        g          += [self.ql_fn(scxl0=Xk)['norm_qlf']]
        self.lbg2  += [1]
        self.ubg2  += [1]
        # add inequality tension magnitude safe constraints
        for i in range(int(self.nq)):
            ti_k= self.Ti_admm[i](scxl0=Xk,scWl0=Wk,tm0=Tk)['t'+str(i)+'f_admm']
            g  += [ti_k]
            self.lbg2 += [self.t_min**2] # to prevent it from being slack
            self.ubg2 += [self.t_max**2]
        # add inequality obstacle-avoidance constraints
        for i in range(int(self.nq)):
            gi1 = self.Gi1_admm[i](scxl0=Xk,scWl0=Wk,tm0=Tk)['go1'+str(i)+'f_admm']
            g += [gi1]
            self.lbg2 += [1e-2]
            self.ubg2 += [1e4] # add an upbound for numerical stability
            gi2 = self.Gi2_admm[i](scxl0=Xk,scWl0=Wk,tm0=Tk)['go2'+str(i)+'f_admm']
            g += [gi2]
            self.lbg2 += [1e-2]
            self.ubg2 += [1e4] # add an upbound for numerical stability

        # add inequality safe inter-robot constraints
        for i in range(len(self.Gij_admm)):
            gij = self.Gij_admm[i](scxl0=Xk,scWl0=Wk,tm0=Tk)['g'+str(i)+'f_admm']
            g += [gij]
            self.lbg2 += [1e-2]
            self.ubg2 += [1e4] # add an upbound for numerical stability
          
        # create an NLP solver and solve it
        opts = {}
        opts['ipopt.tol'] = 1e-8
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.warm_start_init_point']='yes'
        opts['ipopt.max_iter']=1e3
        opts['ipopt.acceptable_tol']=1e-8
        opts['ipopt.mu_strategy']='adaptive'
        
        prob = {'f': J, 
                'x': vertcat(*w), 
                'p': P2l,
                'g': vertcat(*g)}
        
        self.solver2 = nlpsol('solver', 'ipopt', prob, opts)  
        # self.solver2 = nlpsol('solver', 'sqpmethod', prob, opts) 
    
    def ADMM_SubP2(self,P2l):
        P2_l     = P2l[0:self.n_P2]
        p1       = P2l[self.n_P2+2*self.N*self.n_xl+2*self.N*self.n_Wl+self.N*self.n_nv]
        i_admm   = P2l[-1]
        sc_xl_traj = np.zeros((self.N,self.n_xl))
        sc_Wl_traj = np.zeros((self.N,self.n_Wl))
        Tl_traj    = np.zeros((self.N,self.n_nv))
        ctrl_traj  = np.zeros((self.N,self.n_ul))
        for k in range(self.N):
            X0 = []
            W0 = []
            T0 = []
            self.w02  = [] # initial guess list of optimal trajectory 
            xl_k        = P2l[(self.n_P2+k*self.n_xl):(self.n_P2+(k+1)*self.n_xl)]
            for i in range(self.n_xl):
                X0 += [xl_k[i]]
            self.w02 += X0
            xL_k        = P2l[(self.n_P2+self.N*self.n_xl+k*self.n_xl):(self.n_P2+self.N*self.n_xl+(k+1)*self.n_xl)]
            Wl_k        = P2l[(self.n_P2+2*self.N*self.n_xl+k*self.n_Wl):(self.n_P2+2*self.N*self.n_xl+(k+1)*self.n_Wl)]
            for i in range(self.n_Wl):
                W0 += [Wl_k[i]]
            self.w02 += W0
            WL_k        = P2l[(self.n_P2+2*self.N*self.n_xl+self.N*self.n_Wl+k*self.n_Wl):(self.n_P2+2*self.N*self.n_xl+self.N*self.n_Wl+(k+1)*self.n_Wl)]
            Tl_0k       = P2l[(self.n_P2+2*self.N*self.n_xl+2*self.N*self.n_Wl+k*self.n_nv):(self.n_P2+2*self.N*self.n_xl+2*self.N*self.n_Wl+(k+1)*self.n_nv)]
            for i in range(self.n_nv):
                T0 += [Tl_0k[i]]
            self.w02   += T0
            para2       = np.concatenate((P2_l,xl_k))
            para2       = np.concatenate((para2,xL_k))
            para2       = np.concatenate((para2,Wl_k))
            para2       = np.concatenate((para2,WL_k))
            para2       = np.concatenate((para2,[p1]))
            para2       = np.concatenate((para2,[i_admm]))
            # Solve the NLP

            sol = self.solver2(x0=self.w02, 
                          lbx=self.lbw2, 
                          ubx=self.ubw2, 
                          p=para2,
                          lbg=self.lbg2, 
                          ubg=self.ubg2)
        
            w_opt = sol['x'].full().flatten()
            # take the optimal control and state
            sol_traj = np.reshape(w_opt, (-1, self.n_xl + self.n_Wl + self.n_nv))
            state_traj_opt = sol_traj[:, 0:self.n_xl]
            Wl_traj_opt    = sol_traj[:, self.n_xl:(self.n_xl+self.n_Wl)]
            Tl_traj_opt    = sol_traj[:, (self.n_xl+self.n_Wl):]
            control_traj_opt = sol_traj[:, self.n_xl:]
            sc_xl_traj[k:k+1,:] = state_traj_opt
            sc_Wl_traj[k:k+1,:] = Wl_traj_opt
            Tl_traj[k:k+1,:]    = Tl_traj_opt
            ctrl_traj[k:k+1,:]  = control_traj_opt

        # output
        opt_sol2 = {"scxl_opt":sc_xl_traj,
                   "scWl_opt":sc_Wl_traj,
                   "Tl_opt":Tl_traj,
                  "ul_opt":ctrl_traj
                  }
        return opt_sol2 
    

    def system_derivatives_SubP2_ADMM(self):
        # gradients of the Lagrangian (augmented cost function with the soft constraints)
        self.Lscxl        = jacobian(self.J2_k_soft,self.sc_xl)
        self.LscWl        = jacobian(self.J2_k_soft,self.sc_Wl)
        self.Lnv          = jacobian(self.J2_k_soft,self.nv)
        # gradients of the original Lagrangian (augmented cost with the soft constraints but without the ADMM penalties)
        self.Lscxl_o      = jacobian(self.J2_k_orig,self.sc_xl)
        self.LscWl_o      = jacobian(self.J2_k_orig,self.sc_Wl)
        self.Lnv_o        = jacobian(self.J2_k_orig,self.nv)
        # hessians
        self.Lscxlscxl    = jacobian(self.Lscxl,self.sc_xl)
        self.Lscxlscxl_fn = Function('Lscxlscxl',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1, self.a],[self.Lscxlscxl],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10', 'a0'],['Lscxlscxlf']) 
        self.LscxlscWl    = jacobian(self.Lscxl,self.sc_Wl)
        self.LscxlscWl_fn = Function('LscxlscWl',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1, self.a],[self.LscxlscWl],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10', 'a0'],['LscxlscWlf'])
        self.Lscxlnv      = jacobian(self.Lscxl,self.nv)
        self.Lscxlnv_fn   = Function('Lscxlnv',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1, self.a],[self.Lscxlnv],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10', 'a0'],['Lscxlnvf'])
        self.LscWlscWl    = jacobian(self.LscWl,self.sc_Wl)
        self.LscWlscWl_fn = Function('LscWlscWl',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1, self.a],[self.LscWlscWl],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10', 'a0'],['LscWlscWlf'])
        self.LscWlnv      = jacobian(self.LscWl,self.nv)
        self.LscWlnv_fn   = Function('LscWlnv',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1, self.a],[self.LscWlnv],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10', 'a0'],['LscWlnvf'])
        self.Lnvnv        = jacobian(self.Lnv,self.nv)
        self.Lnvnv_fn     = Function('Lnvnv',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1, self.a],[self.Lnvnv],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10', 'a0'],['Lnvnvf'])
        # hessians of the original Lagrangian
        self.Lscxlscxl_o   = jacobian(self.Lscxl_o,self.sc_xl)
        self.Lscxlscxl_fno = Function('Lscxlscxlo',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l],[self.Lscxlscxl_o],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0'],['Lscxlscxlof']) 
        self.LscxlscWl_o   = jacobian(self.Lscxl_o,self.sc_Wl)
        self.LscxlscWl_fno = Function('LscxlscWlo',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l],[self.LscxlscWl_o],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0'],['LscxlscWlof'])
        self.Lscxlnv_o     = jacobian(self.Lscxl_o,self.nv)
        self.Lscxlnv_fno   = Function('Lscxlnvo',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l],[self.Lscxlnv_o],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0'],['Lscxlnvof'])
        self.LscWlscWl_o   = jacobian(self.LscWl_o,self.sc_Wl)
        self.LscWlscWl_fno = Function('LscWlscWlo',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l],[self.LscWlscWl_o],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0'],['LscWlscWlof'])
        self.LscWlnv_o     = jacobian(self.LscWl_o,self.nv)
        self.LscWlnv_fno   = Function('LscWlnvo',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l],[self.LscWlnv_o],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0'],['LscWlnvof'])
        self.Lnvnv_o       = jacobian(self.Lnv_o,self.nv)
        self.Lnvnv_fno     = Function('Lnvnvo',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l],[self.Lnvnv_o],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0'],['Lnvnvof'])
        
        # hessians w.r.t. the hyperparameters
        self.Lscxlp       = jacobian(self.Lscxl,self.P_auto)
        self.Lscxlp_fn    = Function('Lscxlp',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1, self.a],[self.Lscxlp],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10', 'a0'],['Lscxlpf'])
        self.LscWlp       = jacobian(self.LscWl,self.P_auto)
        self.LscWlp_fn    = Function('LscWlp',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1, self.a],[self.LscWlp],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10', 'a0'],['LscWlpf'])
        self.Lnvp         = jacobian(self.Lnv,self.P_auto)
        self.Lnvp_fn      = Function('Lnvp',[self.nv, self.xl, self.Wl, self.sc_xl, self.sc_xL, self.sc_Wl, self.sc_WL, self.P2_l, self.p1, self.a],[self.Lnvp],['nv0', 'xl0', 'Wl0', 'scxl0', 'scxL0', 'scWl0', 'scWL0', 'P2l0', 'p10', 'a0'],['Lnvpf'])


    def Get_AuxSys_SubP2(self,opt_sol1,opt_sol2,scxL_opt,scWL_opt,weight2,p1,i_admm):
        xl_opt   = opt_sol1['xl_opt']
        Wl_opt   = opt_sol1['Wl_opt']
        scxl_opt = opt_sol2['scxl_opt']
        scWl_opt = opt_sol2['scWl_opt']
        Tl_opt   = opt_sol2['Tl_opt']
        Lscxlscxl_l      = self.N*[np.zeros((self.n_xl,self.n_xl))]
        LscxlscWl_l      = self.N*[np.zeros((self.n_xl,self.n_Wl))]
        Lscxlnv_l        = self.N*[np.zeros((self.n_xl,self.n_nv))]
        LscWlscWl_l      = self.N*[np.zeros((self.n_Wl,self.n_Wl))]
        LscWlnv_l        = self.N*[np.zeros((self.n_Wl,self.n_nv))]
        Lnvnv_l          = self.N*[np.zeros((self.n_nv,self.n_nv))]
        Lscxlp_l         = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        LscWlp_l         = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        Lnvp_l           = self.N*[np.zeros((self.n_nv,self.n_Pauto))]
        # hessians of the original Lagrangian for computing the minimal eigenvalue
        Lscxlscxl_lo      = self.N*[np.zeros((self.n_xl,self.n_xl))]
        LscxlscWl_lo      = self.N*[np.zeros((self.n_xl,self.n_Wl))]
        Lscxlnv_lo        = self.N*[np.zeros((self.n_xl,self.n_nv))]
        LscWlscWl_lo      = self.N*[np.zeros((self.n_Wl,self.n_Wl))]
        LscWlnv_lo        = self.N*[np.zeros((self.n_Wl,self.n_nv))]
        Lnvnv_lo          = self.N*[np.zeros((self.n_nv,self.n_nv))]
        for k in range(self.N):
            Lscxlscxl_l[k] = self.Lscxlscxl_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1,a0=i_admm)['Lscxlscxlf'].full()
            LscxlscWl_l[k] = self.LscxlscWl_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1,a0=i_admm)['LscxlscWlf'].full()
            Lscxlnv_l[k]   = self.Lscxlnv_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1,a0=i_admm)['Lscxlnvf'].full()
            LscWlscWl_l[k] = self.LscWlscWl_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1,a0=i_admm)['LscWlscWlf'].full()
            LscWlnv_l[k]   = self.LscWlnv_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1,a0=i_admm)['LscWlnvf'].full()
            Lnvnv_l[k]     = self.Lnvnv_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1,a0=i_admm)['Lnvnvf'].full()
            Lscxlp_l[k]    = self.Lscxlp_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1,a0=i_admm)['Lscxlpf'].full()
            LscWlp_l[k]    = self.LscWlp_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1,a0=i_admm)['LscWlpf'].full()
            Lnvp_l[k]      = self.Lnvp_fn(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2,p10=p1,a0=i_admm)['Lnvpf'].full()
            # hessians of the original Lagrangian
            Lscxlscxl_lo[k] = self.Lscxlscxl_fno(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2)['Lscxlscxlof'].full()
            LscxlscWl_lo[k] = self.LscxlscWl_fno(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2)['LscxlscWlof'].full()
            Lscxlnv_lo[k]   = self.Lscxlnv_fno(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2)['Lscxlnvof'].full()
            LscWlscWl_lo[k] = self.LscWlscWl_fno(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2)['LscWlscWlof'].full()
            LscWlnv_lo[k]   = self.LscWlnv_fno(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2)['LscWlnvof'].full()
            Lnvnv_lo[k]     = self.Lnvnv_fno(nv0=Tl_opt[k,:],xl0=xl_opt[k,:],Wl0=Wl_opt[k,:],scxl0=scxl_opt[k,:],scxL0=scxL_opt[:,k],scWl0=scWl_opt[k,:],scWL0=scWL_opt[:,k],P2l0=weight2)['Lnvnvof'].full()
        auxSys2 = {
                    "Lscxlscxl_l":Lscxlscxl_l,
                    "LscxlscWl_l":LscxlscWl_l,
                    "Lscxlnv_l":Lscxlnv_l,
                    "LscWlscWl_l":LscWlscWl_l,
                    "LscWlnv_l":LscWlnv_l,
                    "Lnvnv_l":Lnvnv_l,
                    "Lscxlp_l":Lscxlp_l,
                    "LscWlp_l":LscWlp_l,
                    "Lnvp_l":Lnvp_l,
                    "Lscxlscxl_lo":Lscxlscxl_lo,
                    "LscxlscWl_lo":LscxlscWl_lo,
                    "Lscxlnv_lo":Lscxlnv_lo,
                    "LscWlscWl_lo":LscWlscWl_lo,
                    "LscWlnv_lo":LscWlnv_lo,
                    "Lnvnv_lo":Lnvnv_lo
        }
        return auxSys2
    

    def ADMM_SubP3(self,xL_opt,WL_opt,xl_opt,scxl_opt,Wl_opt,scWl_opt,p1,i_admm):
        Y_new   = np.zeros((self.n_xl,self.N))
        Eta_new = np.zeros((self.n_Wl,self.N))
        dis_rn  = self.Discount_rate(i_admm) # the numerical value of the iteration-specific ADMM penalty parameter
        for k in range(self.N):
            y_k        = np.reshape(xL_opt[:,k],(self.n_xl,1)) # old Lagrangian multiplier associated with the safe copy state
            xl_k       = np.reshape(xl_opt[k,:],(self.n_xl,1))
            scxl_k     = np.reshape(scxl_opt[k,:],(self.n_xl,1))
            eta_k      = np.reshape(WL_opt[:,k],(self.n_Wl,1)) # old Lagrangian multiplier associated with the safe copy control
            Wl_k       = np.reshape(Wl_opt[k,:],(self.n_Wl,1))
            scWl_k     = np.reshape(scWl_opt[k,:],(self.n_Wl,1))
            y_k_new    = y_k + dis_rn*p1*(xl_k - scxl_k)
            eta_k_new  = eta_k + dis_rn*p1*(Wl_k - scWl_k)
            Y_new[:,k:k+1] = y_k_new
            Eta_new[:,k:k+1] = eta_k_new

        return Y_new, Eta_new
    

    def system_derivatives_SubP3_ADMM(self):
        scxl_update = self.dis_r*self.p1*(self.xl - self.sc_xl)
        scWl_update = self.dis_r*self.p1*(self.Wl - self.sc_Wl)
        self.dscxl_updatedp = jacobian(scxl_update,self.P_auto)
        self.dscxl_updatedp_fn = Function('dscxl_update',[self.xl,self.sc_xl,self.a],[self.dscxl_updatedp],['xl0','scxl0','a0'],['dscxl_updatef'])
        self.dscWl_updatedp = jacobian(scWl_update,self.P_auto)
        self.dscWl_updatedp_fn = Function('dscWl_update',[self.Wl,self.sc_Wl,self.a],[self.dscWl_updatedp],['Wl0','scWl0','a0'],['dscWl_updatef'])


    def Get_AuxSys_SubP3(self,opt_sol1,opt_sol2,i_admm):
        xl_opt   = opt_sol1['xl_opt']
        Wl_opt   = opt_sol1['Wl_opt']
        scxl_opt = opt_sol2['scxl_opt']
        scWl_opt = opt_sol2['scWl_opt']
        dscxl_updatedp_l = self.N*[self.n_xl,self.n_Pauto]
        dscWl_updatedp_l = self.N*[self.n_Wl,self.n_Pauto]
        for k in range(self.N):
            dscxl_updatedp_l[k] = self.dscxl_updatedp_fn(xl0=xl_opt[k,:],scxl0=scxl_opt[k,:],a0=i_admm)['dscxl_updatef'].full()
            dscWl_updatedp_l[k] = self.dscWl_updatedp_fn(Wl0=Wl_opt[k,:],scWl0=scWl_opt[k,:],a0=i_admm)['dscWl_updatef'].full()
        
        auxSys3 = {
                    "dscxl_updp":dscxl_updatedp_l,
                    "dscWl_updp":dscWl_updatedp_l
                }
        return auxSys3


    def ADMM_forward_MPC_DDP(self,xl_fb,Ref_xl,Ref_Wl,weight1,weight2,p1,max_iter_ADMM,adaptiveADMM):
        r_primal = 1e2 # primal residual
        r_dual   = 1e2 # dual residual
        max_iter_DDP = 10
        e_tol    = 1e-2
        xl_traj  = Ref_xl[0:self.N*self.n_xl]
        Wl_traj  = Ref_Wl
        xw       = np.concatenate((xl_traj,Wl_traj)) # collection of all local states and controls
        scxl_opt1 = np.zeros(((self.N)*self.n_xl)) # this initial guess is very important, the worse the initial guess, the larger the initial loss!
        scWl_opt = np.zeros((self.N*self.n_Wl)) # this initial guess is very important
        for k in range(self.N):
            u_k    = np.reshape(Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],(self.n_Wl,1)) # this initial guess is very important
            scxl_opt1[(k)*self.n_xl:(k+1)*self.n_xl] = np.reshape(self.MDynl_fn_admm(xl0=scxl_opt1[k*self.n_xl:(k+1)*self.n_xl],Wl0=u_k)['MDynlf_admm'].full(),self.n_xl)
            scWl_opt[k*self.n_Wl:(k+1)*self.n_Wl] = np.reshape(u_k,self.n_Wl)
        scxl_opt = scxl_opt1[0:self.N*self.n_xl]
        sc_xw    = np.concatenate((scxl_opt,scWl_opt)) # collection of all local safe copy states and controls
        norm_xw  = np.array([LA.norm(xw),LA.norm(sc_xw)])
        e_pri    = np.sqrt(2)*self.e_abs + self.e_rel * np.max(norm_xw)
        e_dual   = np.sqrt(2)*self.e_abs + self.e_rel * (p1*LA.norm(scxl_opt)+p1*LA.norm(scWl_opt))
        Y        = np.zeros((self.n_xl,self.N)) # Lagrangian multiplier trajectory associated with the safe copy state
        Eta      = np.zeros((self.n_Wl,self.N)) # Lagrangian multiplier trajectory associated with the safe copy control
        self.max_iter_ADMM = max_iter_ADMM
        Opt_Sol1 = []
        Opt_Sol2 = []
        Opt_Y    = []
        Opt_Eta  = []
        # initial guess of Tl0
        Tl0      = np.zeros(self.N*(3*int(self.nq)-6))
        i        = 0
        for i_admm in range(int(self.max_iter_ADMM)):
            if adaptiveADMM == 'f':
                i_admm = 1e2 # a very large i_admm makes the sigmoid function almost 1
            Y_l   = np.reshape(Y.T,self.N*self.n_xl) # old state Lagrangian multiplier trajectory
            Eta_l = np.reshape(Eta.T,self.N*self.n_Wl) # old control Lagrangian multiplier trajectory
            # solve Subproblem 1
            start_time = TM.time()
            opt_sol1   = self.DDP_ADMM_Subp1(xl_fb,Ref_xl,Ref_Wl,weight1,scxl_opt,scWl_opt,Y_l,Eta_l,p1,max_iter_DDP,e_tol,i,i_admm)
            mpctime    = (TM.time() - start_time)*1000
            print("subprblem1:--- %s ms ---" % format(mpctime,'.2f'))
            xl_opt   = opt_sol1['xl_opt']
            xl_optr  = np.reshape(xl_opt,(self.N+1)*self.n_xl)
            xl_traj  = xl_optr[0:self.N*self.n_xl]
            Wl_opt   = opt_sol1['Wl_opt']
            Wl_traj  = np.reshape(Wl_opt,self.N*self.n_Wl)
            # solve Subproblem 2
            para2    = np.concatenate((weight2,xl_traj))
            para2    = np.concatenate((para2,Y_l))
            para2    = np.concatenate((para2,Wl_traj))
            para2    = np.concatenate((para2,Eta_l))
            para2    = np.concatenate((para2,Tl0))
            para2    = np.concatenate((para2,[p1]))
            para2    = np.concatenate((para2,[i_admm]))
            start_time = TM.time()
            opt_sol2 = self.ADMM_SubP2(para2)
            mpctime    = (TM.time() - start_time)*1000
            print("subprblem2:--- %s ms ---" % format(mpctime,'.2f'))
            scxl_traj= np.reshape(opt_sol2['scxl_opt'],self.N*self.n_xl) # new safe copy state trajectory
            scWl_traj= np.reshape(opt_sol2['scWl_opt'],self.N*self.n_Wl) # new safe copy control trajectory
            # solve Subproblem 3
            Y_new, Eta_new   = self.ADMM_SubP3(Y,Eta,np.reshape(xl_traj,(self.N,self.n_xl)),np.reshape(scxl_traj,(self.N,self.n_xl)),
                                       np.reshape(Wl_traj,(self.N,self.n_Wl)),np.reshape(scWl_traj,(self.N,self.n_Wl)),p1,i_admm)
            # compute residual (for adaptive penalty parameters)
            r_px        = LA.norm(xl_traj-scxl_traj)
            r_dx        = LA.norm(p1*(scxl_traj-scxl_opt))
            r_pw        = LA.norm(Wl_traj-scWl_traj)
            r_dw        = LA.norm(p1*(scWl_traj-scWl_opt))
            r_primal    = np.sqrt(r_px**2+r_pw**2)
            r_dual      = np.sqrt(r_dx**2+r_dw**2)
            # update the safe copy trajectories, Lagrangian multipliers, and error
            scxl_opt  = scxl_traj 
            scWl_opt  = scWl_traj
            Y         = Y_new
            Eta       = Eta_new
            sc_xw     = np.concatenate((scxl_opt,scWl_opt))
            xw        = np.concatenate((xl_traj,Wl_traj))
            norm_xw   = np.array([LA.norm(xw),LA.norm(sc_xw)])
            e_pri     = np.sqrt(2)*self.e_abs + self.e_rel * np.max(norm_xw)
            e_dual    = np.sqrt(2)*self.e_abs + self.e_rel * (p1*LA.norm(scxl_opt)+p1*LA.norm(scWl_opt))
            dis_rn    = self.Discount_rate(i_admm)
            print('ADMM iteration:',i+1,'r_primal=',r_primal,'r_dual=',r_dual,'e_pri=',e_pri,'e_dual=',e_dual,'current_p1=',dis_rn*p1)
            Opt_Sol1 += [opt_sol1]
            Opt_Sol2 += [opt_sol2]
            Opt_Y    += [Y]
            Opt_Eta  += [Eta]
            # update the initial guess
            Tl0       = np.reshape(opt_sol2['Tl_opt'],self.N*self.n_nv)
            i        += 1

        return Opt_Sol1, Opt_Sol2, Opt_Y, Opt_Eta
    
    def DDP_Gradient(self,opt_sol,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad, p1, i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSys1['HxNp'], auxSys1['Hxp'], auxSys1['Hup']
        S          = (self.N+1)*[np.zeros((self.n_xl,self.n_Pauto))]
        S[self.N]  = HxNp
        v_FF       = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        xl_grad    = (self.N+1)*[np.zeros((self.n_xl,self.n_Pauto))] 
        Wl_grad    = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        dis_rn     = self.Discount_rate(i_admm)
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): # N-1, N-2,...,0
            Hxp_k    = Hxp[k] + scxL_grad[k] - dis_rn*p1*scxl_grad[k]
            Hup_k    = Hup[k] + scWL_grad[k] - dis_rn*p1*scWl_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
        #-------Forward recursion-------#
        for k in range(self.N):
            Wl_grad[k]  = K_fb[k]@xl_grad[k]+v_FF[k]
            xl_grad[k+1]= F[k]@xl_grad[k]+G[k]@Wl_grad[k]

        grad_out ={"xl_grad":xl_grad,
                   "Wl_grad":Wl_grad
                }
        
        return grad_out
    
    
    def Cao_Gradient_s(self,opt_sol,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad, p1, i_admm):
        Quuinv, Qxu, K_fb, F, G  = opt_sol['Quu_inv'], opt_sol['Qxu'], opt_sol['K_FB'], opt_sol['Fx'], opt_sol['Fu']
        HxNp, Hxp, Hup = auxSys1['HxNp'], auxSys1['Hxp'], auxSys1['Hup']
        S           = (self.N+1)*[np.zeros((self.n_xl,self.n_Pauto))] # Vxp
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        S[self.N]   = HxNp
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        v_FF        = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        xl_grad     = (self.N+1)*[np.zeros((self.n_xl,self.n_Pauto))] 
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        Wl_grad     = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        dis_rn      = self.Discount_rate(i_admm)
        #-------Backward recursion-------#         
        for k in reversed(range(self.N)): # N-1, N-2,...,0
            Hpp_k    = np.zeros((self.n_Pauto,self.n_Pauto))
            Hxp_k    = Hxp[k] + scxL_grad[k] - dis_rn*p1*scxl_grad[k]
            Hup_k    = Hup[k] + scWL_grad[k] - dis_rn*p1*scWl_grad[k]
            v_FF[k]  = -Quuinv[k]@(Hup_k + G[k].T@S[k+1])
            Vpp[k]   = Hpp_k + Vpp[k+1] + (Hup_k + G[k].T@S[k+1]).T@v_FF[k] # the augmented Riccati recursion, which is redundant
            S[k]     = Hxp_k + F[k].T@S[k+1] + Qxu[k]@v_FF[k] # s[0] not used
        #-------Foreward recursion-------#
        for k in range(self.N):
            Wl_grad[k]  = K_fb[k]@xl_grad[k]+v_FF[k]@p_grad[k] # expanding the augmented control law gives this form, which is exactly the same as ours
            xl_grad[k+1]= F[k]@xl_grad[k]+G[k]@Wl_grad[k]
            p_grad[k+1] = p_grad[k] # the augmented dynamics, which is redundant
        
        grad_out_cao ={"xl_grad":xl_grad,
                   "Wl_grad":Wl_grad,
                   "p_grad":p_grad
                }
        
        return grad_out_cao
    

    def Cao_Gradient(self,opt_sol,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad,p1,i_admm):
        # solve the augmented optimal problem using one-step DDP recursion
        Hxx, Hxu, Huu, F, G = opt_sol['Hxx'], opt_sol['Hxu'], opt_sol['Huu'], opt_sol['Fx'], opt_sol['Fu']
        HxxN, HxNp, Hxp, Hup = auxSys1['HxxN'], auxSys1['HxNp'], auxSys1['Hxp'], auxSys1['Hup']
        # Vyy      = (self.N+1)*[np.zeros((self.n_Pauto+self.n_xl,self.n_Pauto+self.n_xl))] # a large matrix, leading to significant computation cost
        # we decompose Vyy into four smaller blocks
        Vpp         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_Pauto))]
        Vpx         = (self.N+1)*[np.zeros((self.n_Pauto,self.n_xl))]
        Vxp         = (self.N+1)*[np.zeros((self.n_xl,self.n_Pauto))]
        Vxx         = (self.N+1)*[np.zeros((self.n_xl,self.n_xl))]
        # Kfb_y    = self.N*[np.zeros((self.n_Wl,self.n_Pauto+self.n_xl))] # augmented feedback gain
        Kfb_p       = self.N*[np.zeros((self.n_Wl,self.n_Pauto))] # this matches exactly the feedforward gain!
        Kfb_x       = self.N*[np.zeros((self.n_Wl,self.n_xl))]    # this is the feedback gain
        p_grad      = (self.N+1)*[np.identity(self.n_Pauto)]
        xl_grad     = (self.N+1)*[np.zeros((self.n_xl,self.n_Pauto))] 
        Wl_grad     = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        # Vyy[self.N] = vertcat(
        #                         horzcat(np.zeros((self.n_Pauto,self.n_Pauto)),HxNp.T),
        #                         horzcat(HxNp,self.lxxN_fn(P1l0=weight1)['lxxNf'].full())
        #                     )
        Vpp[self.N] = np.zeros((self.n_Pauto,self.n_Pauto))
        Vpx[self.N] = HxNp.T
        Vxp[self.N] = HxNp
        Vxx[self.N] = HxxN
        dis_rn      = self.Discount_rate(i_admm)
        Iu          = np.identity(self.n_Wl)
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
            Hpx_k    = (Hxp[k]+ scxL_grad[k] - dis_rn*p1*scxl_grad[k]).T
            # Hxp_k    = Hxp[k]+ scxL_grad[k] - dis_rn*p1*scxl_grad[k]
            Hxx_k    = Hxx[k]
            Hup_k    = Hup[k]+ scWL_grad[k] - dis_rn*p1*scWl_grad[k]
            Quu_k    = Huu[k]+G[k].T@Vxx[k+1]@G[k]
            # invQuu_k = LA.solve(Quu_k,Iu) 
            invQuu_k = LA.inv(Quu_k)
            Kfb_p[k] = -invQuu_k@(Hup_k+G[k].T@Vxp[k+1]) 
            Kfb_x[k] = -invQuu_k@(Hxu[k].T+G[k].T@Vxx[k+1]@F[k])
            Vpp[k]   = Vpp[k+1] + (Hup_k.T+Vpx[k+1]@G[k])@Kfb_p[k]
            Vpx[k]   = Hpx_k + Vpx[k+1]@F[k] + Kfb_p[k].T@(Hxu[k]+F[k].T@Vxx[k+1]@G[k]).T
            # Vxp[k]   = Hxp_k + F[k].T@Vxp[k+1] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_p[k]
            Vxp[k]   = Vpx[k].T
            Vxx[k]   = Hxx_k + F[k].T@Vxx[k+1]@F[k] + (Hxu[k]+F[k].T@Vxx[k+1]@G[k])@Kfb_x[k]

        for k in range(self.N):
            Wl_grad[k]   = Kfb_p[k]@p_grad[k] + Kfb_x[k]@xl_grad[k]
            xl_grad[k+1] = F[k]@xl_grad[k]+G[k]@Wl_grad[k]
            p_grad[k+1]  = p_grad[k]
        grad_out_cao ={"xl_grad":xl_grad,
                   "Wl_grad":Wl_grad,
                   "p_grad":p_grad
                }
        
        return grad_out_cao


    def PDP_Gradient(self,auxsys_No,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad, p1, i_admm):
        Hxx, Hxu, Huu, F, G  = auxsys_No['Hxx'], auxsys_No['Hxu'], auxsys_No['Huu'], auxsys_No['Fx'], auxsys_No['Fu']
        HxxN, HxNp, Hxp, Hup = auxSys1['HxxN'], auxSys1['HxNp'], auxSys1['Hxp'], auxSys1['Hup']
        P          = (self.N+1)*[np.zeros((self.n_xl,self.n_xl))]
        S          = (self.N+1)*[np.zeros((self.n_xl,self.n_Pauto))]
        A          = self.N*[np.zeros((self.n_xl,self.n_xl))]
        R          = self.N*[np.zeros((self.n_xl,self.n_xl))]
        M_p        = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        invHuu     = self.N*[np.zeros((self.n_Wl,self.n_Wl))]
        PinvIRP    = self.N*[np.zeros((self.n_xl,self.n_xl))]
        P[self.N]  = HxxN
        S[self.N]  = HxNp
        xl_grad    = (self.N+1)*[np.zeros((self.n_xl,self.n_Pauto))] 
        Wl_grad    = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        I          = np.identity(self.n_xl)
        Iu         = np.identity(self.n_Wl)
        dis_rn     = self.Discount_rate(i_admm)
        for k in reversed(range(self.N)): # N-1, N-2,...,0
        # for k in range(self.N-1,0,-1):
            P_next = P[k+1]
            S_next = S[k+1]
            # L, _jitter   = self.try_cholesky(Huu[k], jitter0=0.0)
            # invHuu[k] = self.chol_solve(L, Iu)
            invHuu[k]   = LA.inv(Huu[k])
            GinvHuu     = G[k]@invHuu[k]
            HxuinvHuu   = Hxu[k]@invHuu[k]
            A[k]        = F[k]-GinvHuu@Hxu[k].T
            R[k]        = GinvHuu@G[k].T
            M_p[k]      = -GinvHuu@(Hup[k]+ scWL_grad[k] - dis_rn*p1*scWl_grad[k])
            Q_k         = Hxx[k]-HxuinvHuu@Hxu[k].T
            N_p_k       = Hxp[k]+ scxL_grad[k] - dis_rn*p1*scxl_grad[k] - HxuinvHuu@(Hup[k]+ scWL_grad[k] - dis_rn*p1*scWl_grad[k])
            # PinvIRP[k]  = LA.solve(IRP.T,P_next.T).T
            PinvIRP[k]  = P_next@LA.inv(I+R[k]@P_next)
            P_curr      = Q_k + A[k].T@PinvIRP[k]@A[k]
            S_curr      = A[k].T@PinvIRP[k]@(M_p[k] - R[k]@S_next) + A[k].T@S_next + N_p_k
            P[k]        = P_curr
            S[k]        = S_curr
        
        for k in range(self.N):
            Wl_grad[k]  = -invHuu[k]@((Hxu[k].T + G[k].T@PinvIRP[k]@A[k])@xl_grad[k] + G[k].T@PinvIRP[k]@(M_p[k]- R[k]@ S[k+1]) + G[k].T@S[k+1] + (Hup[k] + scWL_grad[k] - dis_rn*p1*scWl_grad[k]))
            xl_grad[k+1] = F[k]@xl_grad[k] + G[k]@Wl_grad[k]

        grad_out ={"xl_grad":xl_grad,
                   "Wl_grad":Wl_grad
                }
        
        return grad_out
    

    
    def SubP2_Gradient(self,auxSys2,grad_out,scxL_grad,scWL_grad,p1,i_admm):
        xl_grad      = grad_out['xl_grad']
        Wl_grad      = grad_out['Wl_grad']
        Lscxlscxl_l, LscxlscWl_l, Lscxlnv_l = auxSys2['Lscxlscxl_l'], auxSys2['LscxlscWl_l'], auxSys2['Lscxlnv_l']
        LscWlscWl_l, LscWlnv_l              = auxSys2['LscWlscWl_l'], auxSys2['LscWlnv_l']
        Lnvnv_l                             = auxSys2['Lnvnv_l']
        Lscxlp_l,    LscWlp_l,    Lnvp_l    = auxSys2['Lscxlp_l'], auxSys2['LscWlp_l'], auxSys2['Lnvp_l']
        # hessians of the original Lagrangian
        Lscxlscxl_lo, LscxlscWl_lo, Lscxlnv_lo = auxSys2['Lscxlscxl_lo'], auxSys2['LscxlscWl_lo'], auxSys2['Lscxlnv_lo']
        LscWlscWl_lo, LscWlnv_lo               = auxSys2['LscWlscWl_lo'], auxSys2['LscWlnv_lo']
        Lnvnv_lo                               = auxSys2['Lnvnv_lo']
        scxl_grad    = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        scWl_grad    = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        nv_grad      = self.N*[np.zeros((self.n_nv,self.n_Pauto))]
        I_xw         = np.identity(self.n_xl+self.n_Wl+self.n_nv) # identity matrix
        MIN_eigen    = []
        dis_rn       = self.Discount_rate(i_admm)
        for k in range(self.N):
            L_hessian_k = vertcat(
                            horzcat(Lscxlscxl_l[k],  LscxlscWl_l[k],Lscxlnv_l[k]),
                            horzcat(LscxlscWl_l[k].T,LscWlscWl_l[k],LscWlnv_l[k]),
                            horzcat(Lscxlnv_l[k].T,  LscWlnv_l[k].T,Lnvnv_l[k])
                            )
            L_trajp_k   = vertcat(
                            Lscxlp_l[k] - scxL_grad[k] - dis_rn*p1*xl_grad[k],
                            LscWlp_l[k] - scWL_grad[k] - dis_rn*p1*Wl_grad[k],
                            Lnvp_l[k]
                            )
            L_hessian_ko = vertcat(
                            horzcat(Lscxlscxl_lo[k],  LscxlscWl_lo[k],Lscxlnv_lo[k]),
                            horzcat(LscxlscWl_lo[k].T,LscWlscWl_lo[k],LscWlnv_lo[k]),
                            horzcat(Lscxlnv_lo[k].T,  LscWlnv_lo[k].T,Lnvnv_lo[k])
                            )
            min_eigval = np.min(LA.eigvalsh(L_hessian_ko))
            
            MIN_eigen += [min_eigval]
           
            if min_eigval<0:
                reg = -min_eigval+1e-4
            else:
                reg = 0
            L_hessian_k_sym = L_hessian_k + reg*I_xw
            L, _jitter      = self.try_cholesky(L_hessian_k_sym, jitter0=0.0)
            grad_subp2_k = self.chol_solve(L, -L_trajp_k)
            scxl_grad[k] = grad_subp2_k[0:self.n_xl,:]
            scWl_grad[k] = grad_subp2_k[self.n_xl:(self.n_xl+self.n_Wl),:]
            nv_grad[k]   = grad_subp2_k[(self.n_xl+self.n_Wl):,:]
        print('min_eigen=',np.min(MIN_eigen))    
        grad_out2 = {"scxl_grad":scxl_grad,
                     "scWl_grad":scWl_grad,
                     "nv_grad":nv_grad
                    }
        return grad_out2
    
        
    def SubP3_Gradient(self,auxSys3,grad_out,grad_out2,scxL_grad,scWL_grad,p1,i_admm):
        xl_grad      = grad_out['xl_grad']
        Wl_grad      = grad_out['Wl_grad']
        scxl_grad    = grad_out2['scxl_grad']
        scWl_grad    = grad_out2['scWl_grad']
        dscxl_updp   = auxSys3['dscxl_updp']
        dscWl_updp   = auxSys3['dscWl_updp']
        Y_grad_new   = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        Eta_grad_new = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        dis_rn       = self.Discount_rate(i_admm)
        for k in range(self.N):
            Y_grad_k   = scxL_grad[k] # old Lagrangian gradient associated with the load's state xl
            Eta_grad_k = scWL_grad[k] # old Lagrangian gradient associated with the load's control Wl
            Y_grad_new[k]   = Y_grad_k + dis_rn*p1*(xl_grad[k] - scxl_grad[k]) + dscxl_updp[k]
            Eta_grad_new[k] = Eta_grad_k + dis_rn*p1*(Wl_grad[k] - scWl_grad[k]) + dscWl_updp[k]
        
        grad_out3 = {
                     "scxL_grad":Y_grad_new,
                     "scWL_grad":Eta_grad_new
                    } 
        
        return grad_out3

    def ADMM_Gradient_Solver(self,Opt_Sol1,Opt_Sol2,Opt_Y,Opt_Eta,Ref_xl,Ref_Wl,weight1,weight2,p1,adaptiveADMM):
        # initialize the gradient trajectories of SubP2 and SubP3
        scxl_grad = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        scWl_grad = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        scxL_grad = self.N*[np.zeros((self.n_xl,self.n_Pauto))]
        scWL_grad = self.N*[np.zeros((self.n_Wl,self.n_Pauto))]
        # initial trajectories, same as those used in the ADMM recursion in the forward pass
        scxl_opt  = np.zeros((self.N+1,self.n_xl))
        scWl_opt  = np.zeros((self.N,self.n_Wl))
        for k in range(self.N):
            u_k   = np.reshape(Ref_Wl[k*self.n_Wl:(k+1)*self.n_Wl],(self.n_Wl,1))
            scxl_opt[k+1:k+2,:] = np.reshape(self.MDynl_fn_admm(xl0=scxl_opt[k,:],Wl0=u_k)['MDynlf_admm'].full(),(1,self.n_xl))
            scWl_opt[k:k+1,:] = np.reshape(u_k,(1,self.n_Wl))
        Y         = np.zeros((self.n_xl,self.N)) # Lagrangian multiplier trajectory associated with the safe copy state
        Eta       = np.zeros((self.n_Wl,self.N)) # Lagrangian multiplier trajectory associated with the safe copy control
        # lists for storing gradient trajectories
        Grad_Out1     = []
        Grad_Out2     = []
        Grad_Out3     = []
        GradTime      = []
        GradTimePDP   = []
        GradTimeCao   =[]
        GradTimeCaos  =[]
        MeanerrorCao  = []
        MeanerrorPDP  = []
        i             = 0
        for i_admm in range(int(self.max_iter_ADMM)):
            if adaptiveADMM == 'f':
                i_admm = 1e2 # a very large i_admm makes the sigmoid function almost 1
            # gradients of Subproblem1
            opt_sol1  = Opt_Sol1[i]
            auxSys1   = self.Get_AuxSys_DDP(opt_sol1, Ref_xl, Ref_Wl, scxl_opt, scWl_opt, Y, Eta,weight1, p1, i_admm)

            start_time = TM.time()
            grad_out  = self.DDP_Gradient(opt_sol1,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad, p1, i_admm)
            gradtimeOur    = (TM.time() - start_time)*1000
            # print('ADMM iteration:',i+1,"g_Our:--- %s ms ---" % format(gradtimeOur,'.2f'))

            start_time = TM.time()
            grad_out_Cao_s = self.Cao_Gradient_s(opt_sol1,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad, p1, i_admm)
            gradtimeCao_s  = (TM.time() - start_time)*1000
            # print('ADMM iteration:',i+1,"g_Cao_s:--- %s ms ---" % format(gradtimeCao_s,'.2f'))

            start_time = TM.time()
            grad_out_Cao   = self.Cao_Gradient(opt_sol1,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad, p1, i_admm)
            gradtimeCao    = (TM.time() - start_time)*1000
            # print('ADMM iteration:',i+1,"g_Cao:--- %s ms ---" % format(gradtimeCao,'.2f'))

            start_time = TM.time()
            grad_out_PDP   = self.PDP_Gradient(opt_sol1 ,auxSys1, scxl_grad, scxL_grad, scWl_grad, scWL_grad, p1, i_admm)
            gradtimePDP    = (TM.time() - start_time)*1000
            # print('ADMM iteration:',i+1,"g_PDP:--- %s ms ---" % format(gradtimePDP,'.2f'))

            # gradients of Subproblem2
            opt_sol2  = Opt_Sol2[i]
            auxSys2   = self.Get_AuxSys_SubP2(opt_sol1, opt_sol2, Y, Eta, weight2, p1, i_admm)
            grad_out2 = self.SubP2_Gradient(auxSys2, grad_out, scxL_grad, scWL_grad, p1, i_admm)
            # gradients of Subproblem3
            auxSys3   = self.Get_AuxSys_SubP3(opt_sol1, opt_sol2, i_admm)
            grad_out3 = self.SubP3_Gradient(auxSys3, grad_out, grad_out2, scxL_grad, scWL_grad, p1, i_admm)
            # update
            scxl_opt  = opt_sol2['scxl_opt']
            scWl_opt  = opt_sol2['scWl_opt']
            Y         = Opt_Y[i]
            Eta       = Opt_Eta[i]
            scxl_grad = grad_out2['scxl_grad']
            scWl_grad = grad_out2['scWl_grad']
            scxL_grad = grad_out3['scxL_grad']
            scWL_grad = grad_out3['scWL_grad']
            # save the results
            Grad_Out1    += [grad_out]
            Grad_Out2    += [grad_out2]
            Grad_Out3    += [grad_out3]
            GradTime     += [gradtimeOur]
            GradTimePDP   += [gradtimePDP]
            GradTimeCao  += [gradtimeCao]
            GradTimeCaos += [gradtimeCao_s]
            # error between two gradient trajectories
            xl_grad    = grad_out['xl_grad']
            xl_gradCao = grad_out_Cao['xl_grad']
            xl_gradPDP = grad_out_PDP['xl_grad']
            ErrorCao   = 0
            ErrorPDP   = 0
            for j in range(self.N):
                error = xl_grad[j+1] - xl_gradCao[j+1]
                ErrorCao += (LA.norm(error,ord='fro')/LA.norm(xl_grad[j+1],ord='fro'))
                error2 = xl_grad[j+1] - xl_gradPDP[j+1]
                ErrorPDP += (LA.norm(error2,ord='fro')/LA.norm(xl_grad[j+1],ord='fro')) # relative error
                
            meanerrorCao = ErrorCao/self.N
            meanerrorPDP = ErrorPDP/self.N
            if i == int(self.max_iter_ADMM) -1:
                print('ADMM iteration:',i+1,"g_Our:--- %s ms ---" % format(gradtimeOur,'.2f'))
                print('ADMM iteration:',i+1,"g_Cao_s:--- %s ms ---" % format(gradtimeCao_s,'.2f'))
                print('ADMM iteration:',i+1,"g_Cao:--- %s ms ---" % format(gradtimeCao,'.2f'))
                print('ADMM iteration:',i+1,"g_PDP:--- %s ms ---" % format(gradtimePDP,'.2f'))
                print('ADMM iteration:',i+1,'Cao_meanerror=',meanerrorCao,'PDP_meanerror=',meanerrorPDP)
            MeanerrorCao += [meanerrorCao]
            MeanerrorPDP += [meanerrorPDP]
            i      += 1

        return Grad_Out1, Grad_Out2, Grad_Out3, GradTime, GradTimePDP, GradTimeCao, GradTimeCaos, MeanerrorCao, MeanerrorPDP


        

            

class Gradient_Solver:
    def __init__(self, horizon, xl, Wl, scxl, scWl, P_auto):
        self.n_xl   = xl.numel()
        self.n_Wl   = Wl.numel()
        self.n_P    = P_auto.numel()
        self.xl     = xl
        self.Wl     = Wl
        self.scxl   = scxl
        self.scWl   = scWl
        self.xl_ref = SX.sym('x_ref',self.n_xl)
        self.N      = horizon
        # boundaries of the hyperparameters
        self.p_min  = 1e-3
        self.p_max  = 1e3
        #------------- loss definition -------------#
        # tracking loss
        tracking_error  = self.xl - self.xl_ref
        self.loss_track = tracking_error.T@tracking_error
        # primal residual loss
        r_primal_x      = self.xl - self.scxl
        r_primal_w      = self.Wl - self.scWl
        self.loss_rp    = r_primal_x.T@r_primal_x + r_primal_w.T@r_primal_w
   
    def adaptive_meta_loss_weights(self,loss_t,loss_rp,wt):
        if loss_t >5*loss_rp:
            wt_new = np.clip(1.25*wt,0.1,10)
        elif loss_rp > 5*loss_t:
            wt_new = np.clip(wt/1.25,0.1,10)
        else:
            wt_new = wt
        return wt_new
    

    def Set_Parameters(self,tunable_para):
        weight       = np.zeros(self.n_P)
        for k in range(self.n_P):
            weight[k]= self.p_min + (self.p_max - self.p_min) * 1/(1+np.exp(-tunable_para[k])) # sigmoid boundedness
            # weight[k]= self.p_min + tunable_para[k]**2

        return weight
    

    def Set_Parameters_nn(self,tunable_para):
        weight       = np.zeros(self.n_P)
        for k in range(self.n_P):
            weight[k]= self.p_min + (self.p_max - self.p_min) * tunable_para[0,k] # sigmoid boundedness
            # weight[k]= self.p_min + tunable_para[k]**2

        return weight
    
    def ChainRule_Gradient(self,tunable_para):
        Tunable      = SX.sym('Tp',1,self.n_P)
        Weight       = SX.sym('wp',1,self.n_P)
        for k in range(self.n_P):
            Weight[k]= self.p_min + (self.p_max - self.p_min) * 1/(1 + exp(-Tunable[k])) # sigmoid boundedness
        dWdT         = jacobian(Weight,Tunable)
        dWdT_fn      = Function('dWdT',[Tunable],[dWdT],['Tp0'],['dWdT_f'])
        weight_grad  = dWdT_fn(Tp0=tunable_para)['dWdT_f'].full()

        return weight_grad
    
    def ChainRule_Gradient_nn(self,tunable_para):
        Tunable      = SX.sym('Tp',1,self.n_P)
        Weight       = SX.sym('wp',1,self.n_P)
        for k in range(self.n_P):
            Weight[k]= self.p_min + (self.p_max - self.p_min) * Tunable[k] # sigmoid boundedness
        dWdT         = jacobian(Weight,Tunable)
        dWdT_fn      = Function('dWdT',[Tunable],[dWdT],['Tp0'],['dWdT_f'])
        weight_grad  = dWdT_fn(Tp0=tunable_para)['dWdT_f'].full()

        return weight_grad

    def loss(self,Opt_Sol1,Opt_Sol2,Ref_xl,wt,wrp):
        xl_opt     = Opt_Sol1[-1]['xl_opt']
        Wl_opt     = Opt_Sol1[-1]['Wl_opt']
        scxl_opt   = Opt_Sol2[-1]['scxl_opt']
        scWl_opt   = Opt_Sol2[-1]['scWl_opt']
        loss_track  = 0
        loss_rp     = 0
        for k in range(self.N):
            xl_k    = np.reshape(xl_opt[k,:],(self.n_xl,1)) 
            refxl_k = np.reshape(Ref_xl[k*self.n_xl:(k+1)*self.n_xl],(self.n_xl,1))
            # tracking_error = np.vstack((error_pl,error_vl,att_error_l,error_wl))
            tracking_error = xl_k - refxl_k
            loss_track    += tracking_error.T@tracking_error
            r_primal_x     = np.reshape(xl_opt[k,:],(self.n_xl,1)) - np.reshape(scxl_opt[k,:],(self.n_xl,1))
            r_primal_w     = np.reshape(Wl_opt[k,:],(self.n_Wl,1)) - np.reshape(scWl_opt[k,:],(self.n_Wl,1))
            loss_rp       += r_primal_x.T@r_primal_x + r_primal_w.T@r_primal_w 
        loss = wt*loss_track + wrp*loss_rp 
        return loss, loss_track, loss_rp
    
    
    def ChainRule(self,Opt_Sol1,Opt_Sol2,Ref_xl,Grad_Out1,Grad_Out2,wt,wrp):
        dltdxl          = jacobian(self.loss_track,self.xl)
        dltdxl_fn       = Function('dltdxl',[self.xl,self.xl_ref],[dltdxl],['xl0','refxl0'],['dltdxl_f'])
        dlrpdxl         = jacobian(self.loss_rp,self.xl)
        dlrpdxl_fn      = Function('dlrpdxl',[self.xl,self.scxl],[dlrpdxl],['xl0','scxl0'],['dlrpdxl_f'])
        dlrpdscxl       = jacobian(self.loss_rp,self.scxl)
        dlrpdscxl_fn    = Function('dlrpdscxl',[self.xl,self.scxl],[dlrpdscxl],['xl0','scxl0'],['dlrpdscxl_f'])
        dlrpdWl         = jacobian(self.loss_rp,self.Wl)
        dlrpdWl_fn      = Function('dlrpdWl',[self.Wl,self.scWl],[dlrpdWl],['Wl0','scWl0'],['dlrpdWl_f'])
        dlrpdscWl       = jacobian(self.loss_rp,self.scWl)
        dlrpdscWl_fn    = Function('dlrpdscWl',[self.Wl,self.scWl],[dlrpdscWl],['Wl0','scWl0'],['dlrpdscWl_f'])
        
        dltdw           = 0
        dlrpdw          = 0
       
        xl_opt          = Opt_Sol1[-1]['xl_opt']
        Wl_opt          = Opt_Sol1[-1]['Wl_opt']
        scxl_opt        = Opt_Sol2[-1]['scxl_opt']
        scWl_opt        = Opt_Sol2[-1]['scWl_opt']
     
        xl_grad         = Grad_Out1[-1]['xl_grad']
        Wl_grad         = Grad_Out1[-1]['Wl_grad']
        scxl_grad       = Grad_Out2[-1]['scxl_grad']
        scWl_grad       = Grad_Out2[-1]['scWl_grad']
   
        loss, loss_track, loss_rp    = self.loss(Opt_Sol1,Opt_Sol2,Ref_xl,wt,wrp)
        for k in range(self.N):
            # gradient of the tracking errors
            dltdxl_k       = dltdxl_fn(xl0=xl_opt[k,:],refxl0=Ref_xl[k*self.n_xl:(k+1)*self.n_xl])['dltdxl_f'].full()
            dltdw         += dltdxl_k@xl_grad[k]
            # gradient of the primal residuals
            dlrpdxl_k      = dlrpdxl_fn(xl0=xl_opt[k,:],scxl0=scxl_opt[k,:])['dlrpdxl_f'].full()
            dlrpdscxl_k    = dlrpdscxl_fn(xl0=xl_opt[k,:],scxl0=scxl_opt[k,:])['dlrpdscxl_f'].full()
            dlrpdWl_k      = dlrpdWl_fn(Wl0=Wl_opt[k,:],scWl0=scWl_opt[k,:])['dlrpdWl_f'].full()
            dlrpdscWl_k    = dlrpdscWl_fn(Wl0=Wl_opt[k,:],scWl0=scWl_opt[k,:])['dlrpdscWl_f'].full()
            dlrpdw        += dlrpdxl_k@xl_grad[k] + dlrpdscxl_k@scxl_grad[k] + dlrpdWl_k@Wl_grad[k] + dlrpdscWl_k@scWl_grad[k]
           
        dltdxl_N    = dltdxl_fn(xl0=xl_opt[-1,:],refxl0=Ref_xl[self.N*self.n_xl:(self.N+1)*self.n_xl])['dltdxl_f'].full()
        dltdw      += dltdxl_N@xl_grad[self.N]
        dldw        = wt*dltdw + wrp*dlrpdw 

        return dldw, loss, loss_track, loss_rp
        
        
            





    
