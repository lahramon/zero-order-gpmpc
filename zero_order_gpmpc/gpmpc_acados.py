import numpy as np
import casadi as cas

import torch
import gpytorch

from scipy.stats import norm
from acados_template import AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver
from zoro_acados_utils import *

from time import perf_counter
from dataclasses import dataclass

@dataclass
class ZoroAcadosData:
    n_iter: int
    sol_x: np.ndarray
    sol_u: np.ndarray 
    sol_P: np.ndarray
    timings_total: float
    timings: dict

class GPMPCAcados():
    def __init__(self, ocp, sim, prob_x, Sigma_x0, Sigma_W, B=None, gp_model=None, use_model_params=True, use_cython=True):
        """
        ocp: AcadosOcp for nominal problem
        sim: AcadosSim for nominal model
        gp_model: GPyTorch GP model
        """
        # get dimensions
        nx_nom = Sigma_x0.shape[0]
        nx = ocp.model.x.size()[0]
        nu = ocp.model.u.size()[0]
        N = ocp.dims.N
        T = ocp.solver_options.tf

        self.nx = nx
        self.nx_nom = nx_nom
        self.nu = nu
        self.np = np
        self.N = N
        self.T = T

        if B is None:
            B = np.eye(self.nx)
        
        self.nw = B.shape[1]
        self.B = B
        self.Sigma_x0 = Sigma_x0
        self.Sigma_W = Sigma_W

        # allocation
        self.x_hat_all = np.zeros((N+1, nx))
        self.u_hat_all = np.zeros((N, nu))
        self.y_hat_all = np.zeros((N,nx_nom+nu))
        self.P_bar_all = [None] * (N+1)
        self.P_bar_all_vec = [None] * (N+1)
        self.P_bar_old_vec = None
        self.P_bar_all[0] = Sigma_x0
        self.P_bar_all_vec[0] = sym_mat2vec(Sigma_x0)

        self.A_lin = np.zeros((N, nx, nx))
        self.A_hess = np.zeros((N, nx_nom, nx_nom, nx_nom+nu))
        self.B_lin = np.zeros((N, nx, nu))
        self.x_nom_all = np.zeros((N, nx))
        self.integrate_hessian_sens_adj = np.vstack((
            np.eye(self.nx_nom),
            np.zeros((self.nx-self.nx_nom,self.nx_nom))
        ))

        self.ocp = transform_ocp(ocp)

        # need to compute Hessian
        sim.solver_options.sens_hess = True

        if use_cython:
            AcadosOcpSolver.generate(ocp, json_file = 'acados_ocp_' + ocp.model.name + '.json')
            AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
            self.ocp_solver = AcadosOcpSolver.create_cython_solver('acados_ocp_' + ocp.model.name + '.json')

            AcadosSimSolver.generate(sim, json_file = 'acados_sim_' + ocp.model.name + '.json')
            AcadosSimSolver.build(sim.code_export_directory, with_cython=True)
            self.sim_solver = AcadosSimSolver.create_cython_solver('acados_sim_' + ocp.model.name + '.json')
        else:
            self.ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + ocp.model.name + '.json')
            self.sim_solver = AcadosSimSolver(sim, json_file = 'acados_sim_' + ocp.model.name + '.json')

        # self.sim_solver.options_set("sens_hess", True) 

        if gp_model is None:
            self.has_gp_model = False
        else:
            self.has_gp_model = True
            self.gp_model = gp_model
            self.gp_sensitivities = generate_gp_funs(gp_model, covar_jac=True, B=self.B)

        # timings
        self.solve_stats_default = {
            "n_iter": 0,
            "timings_total": 0.0,
            "timings": {
                "build_lin_model": 0.0,
                "query_nodes": 0.0,
                "get_gp_sensitivities": 0.0,
                "integrate_acados": 0.0,
                "integrate_acados_python": 0.0,
                "integrate_get": 0.0,
                "integrate_set": 0.0,
                "set_sensitivities": 0.0,
                "set_sensitivities_reshape": 0.0,
                "propagate_covar": 0.0,
                "get_backoffs": 0.0,
                "get_backoffs_htj_sig": 0.0,
                "get_backoffs_htj_sig_matmul": 0.0,
                "get_backoffs_add": 0.0,
                "set_tightening": 0.0,
                "phase_one": 0.0,
                "check_termination": 0.0,
                "solve_qp": 0.0,
                "solve_qp_acados": 0.0,
                "total": 0.0,
            }
        }
        self.solve_stats = self.solve_stats_default.copy()

    def solve(self, tol_nlp=1e-6, n_iter_max=30):
        time_total = perf_counter()
        self.init_solve_stats(n_iter_max)

        nx = self.nx
        nx_nom = self.nx_nom
        nu = self.nu
        nw = self.nw
        N = self.N
        mean = np.zeros((N, nw))
        mean_dy = np.zeros((nw, N, nx+nu))
        var = np.zeros((N,nw))
        var_dy = np.zeros((nw, N, nx+nu))
        f_hat = np.zeros((nx,))

        for i in range(n_iter_max):
            time_iter = perf_counter()
            # preparation rti_phase (solve() AFTER setting params to get right Jacobians)
            self.ocp_solver.options_set('rti_phase', 1)

            # get sensitivities for all stages
            for stage in range(self.N):
                # ------------------- Query nodes --------------------
                time_query_nodes = perf_counter()
                self.x_hat_all[stage,:] = self.ocp_solver.get(stage,"x")   
                self.u_hat_all[stage,:] = self.ocp_solver.get(stage,"u")   
                self.y_hat_all[stage,:] = np.hstack((self.x_hat_all[stage,0:nx_nom],self.u_hat_all[stage,:])).reshape((1,nx_nom+nu))   
                self.solve_stats["timings"]["query_nodes"][i] += perf_counter() - time_query_nodes

                # ------------------- Integrate --------------------
                time_integrate_set = perf_counter()
                self.sim_solver.set("x", self.x_hat_all[stage,:])
                self.sim_solver.set("u", self.u_hat_all[stage,:])
                self.solve_stats["timings"]["integrate_set"][i] += perf_counter() - time_integrate_set

                # loop through adjoint sensitivities for full Hessian
                for i_adj in range(self.nx_nom):
                    time_integrate_set = perf_counter()
                    self.sim_solver.set("seed_adj", self.integrate_hessian_sens_adj[:,i_adj])
                    self.solve_stats["timings"]["integrate_set"][i] += perf_counter() - time_integrate_set
                    
                    # integrate
                    time_integrate_acados_python = perf_counter()
                    status_integrator = self.sim_solver.solve()
                    self.solve_stats["timings"]["integrate_acados_python"][i] += perf_counter() - time_integrate_acados_python
                    self.solve_stats["timings"]["integrate_acados"][i] += self.sim_solver.get("time_tot")

                    # get Hessian of i_adj's dynamics component
                    time_integrate_get = perf_counter()
                    H = self.sim_solver.get("S_hess")
                    self.A_hess[stage,i_adj,:,0:nx_nom] = H[0:nx_nom,0:nx_nom]
                    self.A_hess[stage,i_adj,:,nx_nom:nx_nom+nu] = H[0:nx_nom,nx:nx+nu]
                    self.solve_stats["timings"]["integrate_get"][i] += perf_counter() - time_integrate_get

                # get Jacobians and predictions
                time_integrate_get = perf_counter()
                self.A_lin[stage,:,:] = self.sim_solver.get("Sx")
                self.B_lin[stage,:,:] = self.sim_solver.get("Su")
                self.x_nom_all[stage,:] = self.sim_solver.get("x")
                self.solve_stats["timings"]["integrate_get"][i] += perf_counter() - time_integrate_get
            
            # ------------------- GP Sensitivities --------------------
            torch.cuda.synchronize()
            time_get_gp_sensitivities = perf_counter()

            if self.has_gp_model:
                # A_lin_y = np.concatenate((
                #     self.A_lin[:,0:nx_nom,0:nx_nom],
                #     self.B_lin[:,0:nx_nom,:],
                # ), axis = 2)
                mean, mean_dy, prop, prop_dy, prop_dP = self.gp_sensitivities(self.y_hat_all, self.x_hat_all[:-1,nx_nom:], self.A_lin[:,0:nx_nom,0:nx_nom], self.A_hess)

            torch.cuda.synchronize()
            self.solve_stats["timings"]["get_gp_sensitivities"][i] += perf_counter() - time_get_gp_sensitivities
            
            # ------------------- Update stages --------------------
            for stage in range(N):
                # ------------------- Build linear model --------------------
                time_build_lin_model = perf_counter()

                # mean dy
                self.A_lin[stage,0:nx_nom,0:nx_nom] += self.B @ mean_dy[:,stage,0:nx_nom]
                self.B_lin[stage,0:nx_nom,:] += self.B @ mean_dy[:,stage,nx_nom:nx_nom+nu]
                
                # prop dy
                self.A_lin[stage,nx_nom:,0:nx_nom] += prop_dy[:,stage,0:nx_nom]
                self.B_lin[stage,nx_nom:,:] += prop_dy[:,stage,nx_nom:nx_nom+nu]

                # prop dP
                self.A_lin[stage,nx_nom:,nx_nom:] += prop_dP[:,stage,:]

                f_hat[0:nx_nom] = self.x_nom_all[stage,0:nx_nom] + self.B @ mean[stage,:] \
                    - self.A_lin[stage,0:nx_nom,:] @ self.x_hat_all[stage,:] - self.B_lin[stage,0:nx_nom,:] @ self.u_hat_all[stage,:]
                f_hat[nx_nom:] = self.x_nom_all[stage,nx_nom:] + prop[stage,:] \
                    - self.A_lin[stage,nx_nom:,:] @ self.x_hat_all[stage,:] - self.B_lin[stage,nx_nom:,:] @ self.u_hat_all[stage,:]

                self.solve_stats["timings"]["build_lin_model"][i] += perf_counter() - time_build_lin_model
                
                # ------------------- Set sensitivities --------------------
                time_set_sensitivities_reshape = perf_counter()

                A_reshape = np.reshape(self.A_lin[stage,:,:],(nx**2),order="F")
                B_reshape = np.reshape(self.B_lin[stage,:,:],(nx*nu),order="F")
                
                self.solve_stats["timings"]["set_sensitivities_reshape"][i] += perf_counter() - time_set_sensitivities_reshape
                time_set_sensitivities = perf_counter()

                self.ocp_solver.set(stage, "p", np.hstack((
                    A_reshape,
                    B_reshape,
                    f_hat,
                )))

                self.solve_stats["timings"]["set_sensitivities"][i] += perf_counter() - time_set_sensitivities

                # ------------------- Compute back off --------------------
                # time_get_backoffs = perf_counter()

                # time_get_backoffs_htj_sig = perf_counter()
                # htj_sig = self.h_tightening_jac_sig_fun(self.x_hat_all[stage,:], self.P_bar_all_vec[stage])          
                # self.solve_stats["timings"]["get_backoffs_htj_sig"][i] += perf_counter() - time_get_backoffs_htj_sig
                
                # time_get_backoffs_htj_sig_matmul = perf_counter()
                # htj_sig_matmul = htj_sig @ dP_bar_vec         
                # self.solve_stats["timings"]["get_backoffs_htj_sig_matmul"][i] += perf_counter() - time_get_backoffs_htj_sig_matmul
                
                # time_get_backoffs_add = perf_counter()
                # tightening = htj_sig_matmul
                # lh = self.ocp.constraints.lh + tightening

                # time_now = perf_counter()
                # self.solve_stats["timings"]["get_backoffs_add"][i] = time_now - time_get_backoffs_add
                # self.solve_stats["timings"]["get_backoffs"][i] += time_now - time_get_backoffs
                
                # ------------------- Set tightening --------------------
                # time_set_tightening = perf_counter()

                # # set constraints
                # # t_set_C = self.ocp_solver.constraints_set(stage,"C",h_sum_jac.full(),api="new")
                # # t_set_lg = self.ocp_solver.constraints_set(stage,"lg",lg.full().flatten())
                # # t_set_ug = self.ocp_solver.constraints_set(stage,"ug",ug.full().flatten())
                # self.ocp_solver.constraints_set(stage,"lh",lh.full().flatten())

                # self.solve_stats["timings"]["set_tightening"][i] += perf_counter() - time_set_tightening
                # self.solve_stats["timings"]["set_tightening_raw"][i] += t_set_C + t_set_lg + t_set_ug

            # feedback rti_phase
            # self.ocp_solver.options_set('rti_phase', 1)
            # ------------------- Phase 1 --------------------
            time_phase_one = perf_counter()
            status = self.ocp_solver.solve()
            self.solve_stats["timings"]["phase_one"][i] += perf_counter() - time_phase_one

            # ------------------- Solve QP --------------------
            time_solve_qp = perf_counter()

            self.ocp_solver.options_set('rti_phase', 2)
            status = self.ocp_solver.solve()
                
            self.solve_stats["timings"]["solve_qp"][i] += perf_counter() - time_solve_qp
            self.solve_stats["timings"]["solve_qp_acados"][i] += self.ocp_solver.get_stats("time_tot")

            # ------------------- Check termination --------------------
            # check on residuals and terminate loop.
            time_check_termination = perf_counter()
            
            # self.ocp_solver.print_statistics() # encapsulates: stat = self.ocp_solver.get_stats("statistics")
            residuals = self.ocp_solver.get_residuals()
            print("residuals after ", i, "SQP_RTI iterations:\n", residuals)

            self.solve_stats["timings"]["check_termination"][i] += perf_counter() - time_check_termination
            self.solve_stats["timings"]["total"][i] += perf_counter() - time_iter

            if status != 0:
                raise Exception('acados self.ocp_solver returned status {} in time step {}. Exiting.'.format(status, i))

            if max(residuals) < tol_nlp:
                break
        
        self.solve_stats["n_iter"] = i + 1
        self.solve_stats["timings_total"] = perf_counter() - time_total

    def get_solution(self):
        X = np.zeros((self.N+1, self.nx_nom))
        U = np.zeros((self.N, self.nu))
        P = np.zeros((self.N+1, self.nx_nom, self.nx_nom))

        # get data
        for i in range(self.N):
            x = self.ocp_solver.get(i, "x")
            X[i,:] = x[0:self.nx_nom]
            P[i,:,:] = vec2sym_mat(x[self.nx_nom:], self.nx_nom) 
            U[i,:] = self.ocp_solver.get(i, "u")

        x = self.ocp_solver.get(self.N, "x")
        X[self.N,:] = x[0:self.nx_nom]
        P[self.N,:,:] = vec2sym_mat(x[self.nx_nom:], self.nx_nom) 

        return X,U,P

    def print_solve_stats(self):
        n_iter = self.solve_stats["n_iter"]

        time_other = 0.0
        for key, t_arr in self.solve_stats["timings"].items():
            for i in range(n_iter):
                t_sum = np.sum(t_arr[0:n_iter])
                t_avg = t_sum / n_iter
                t_max = np.max(t_arr[0:n_iter])
                t_min = np.min(t_arr[0:n_iter])
                if key != "integrate_acados":
                    if key != "total":
                        time_other -= t_sum
                    else:
                        time_other += t_sum
            print(f"{key:20s}: {1000*t_sum:8.3f}ms ({n_iter} calls), {1000*t_avg:8.3f}/{1000*t_max:8.3f}/{1000*t_min:8.3f}ms (avg/max/min per call)")

        key = "other"
        t = time_other
        print("----------------------------------------------------------------")
        print(f"{key:20s}: {1000*t:8.3f}ms ({n_iter} calls), {1000*t/n_iter:8.3f}ms (1 call)")
    
    def init_solve_stats(self, max_iter):
        self.solve_stats = self.solve_stats_default.copy()
        for k in self.solve_stats["timings"].keys():
            self.solve_stats["timings"][k] = np.zeros((max_iter,))
    
    def get_solve_stats(self):
        X,U,P = self.get_solution()
        for k in self.solve_stats["timings"].keys():
            self.solve_stats["timings"][k] = self.solve_stats["timings"][k][0:self.solve_stats["n_iter"]]

        return ZoroAcadosData(
            self.solve_stats["n_iter"], 
            X, 
            U, 
            P, 
            self.solve_stats["timings_total"],
            self.solve_stats["timings"].copy()
        )