# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3.9.13 ('zero-order-gp-mpc-code-2CX1fffa')
#     language: python
#     name: python3
# ---

# +
import sys, os
sys.path += ["../../"]
# import dotenv
import importlib

import numpy as np
# from numpy import linalg as npla
from scipy.linalg import block_diag
from scipy.stats import norm
import casadi as cas
from acados_template import AcadosOcp, AcadosSim, AcadosSimSolver, AcadosOcpSolver, AcadosOcpOptions
import matplotlib.pyplot as plt
from dataclasses import dataclass

import inverted_pendulum_model_acados
importlib.reload(inverted_pendulum_model_acados)
from inverted_pendulum_model_acados import export_simplependulum_ode_model, \
export_ode_model_with_discrete_rk4, \
export_robust_ode_model_with_constraints, \
export_linear_model, \
export_ocp_nominal

import utils
importlib.reload(utils)
from utils import base_plot, add_plot_trajectory, EllipsoidTubeData2D

import zero_order_gpmpc
importlib.reload(zero_order_gpmpc)
from zero_order_gpmpc import ZoroAcados

import zero_order_gpmpc.zoro_acados_utils as zero_order_gpmpc_utils
importlib.reload(zero_order_gpmpc_utils)
# -

#

# +
# manual discretization
N = 30
T = 5
dT = T / N

prob_x = 0.9
prob_tighten = norm.ppf(prob_x)

# cost
cost_theta = 10
cost_omega = 1
cost_fac_e = 1
Q = np.diagflat(np.array([cost_theta, cost_omega]))
R = np.array(1)

# constraints
x0 = np.array([np.pi, 0])
lb_u = -2.0
ub_u = 2.0
lb_theta = (0./360.) * 2 * np.pi
ub_theta = (200./360.) * 2 * np.pi

# noise
# uncertainty dynamics
sigma_theta = (0.0001/360.) * 2 * np.pi
sigma_omega = (0.0001/360.) * 2 * np.pi
w_theta = 0.03
w_omega = 0.03
Sigma_x0 = cas.DM(np.array([
    [sigma_theta**2,0],
    [0,sigma_omega**2]
]))
Sigma_W = cas.DM(np.array([
    [w_theta**2, 0],
    [0, w_omega**2]
]))

# -

nx = 2
nu = 1

# +
# integrator for nominal model
sim = AcadosSim()

sim.model = export_simplependulum_ode_model(noise=False)
sim.solver_options.integrator_type = "ERK"
# sim.parameter_values = np.zeros((nx,1))

# set prediction horizon
sim.solver_options.T = dT

# acados_ocp_solver = AcadosOcpSolver(ocp_init, json_file = 'acados_ocp_' + model.name + '.json')
acados_integrator = AcadosSimSolver(sim, json_file = 'acados_sim_' + sim.model.name + '.json')

# +
ocp_init = export_ocp_nominal(N,T)
ocp_init.solver_options.integrator_type = "ERK"
ocp_init.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
ocp_init.solver_options.hessian_approx = "GAUSS_NEWTON"
ocp_init.solver_options.nlp_solver_type = "SQP"

acados_ocp_init_solver = AcadosOcpSolver(ocp_init, json_file="acados_ocp_init_simplependulum_ode.json")

# +
# get initial values
X_init = np.zeros((N+1, nx))
U_init = np.zeros((N, nu))

# xcurrent = x0
X_init[0,:] = x0

# solve
status_init = acados_ocp_init_solver.solve()

if status_init != 0:
    raise Exception('acados acados_ocp_solver returned status {}. Exiting.'.format(status_init))

# get data
for i in range(N):
    X_init[i,:] = acados_ocp_init_solver.get(i, "x")
    U_init[i,:] = acados_ocp_init_solver.get(i, "u")

X_init[N,:] = acados_ocp_init_solver.get(N, "x")
# -

acados_ocp_init_solver.get_residuals()

plt.plot(U_init)

# +
fig, ax = base_plot(lb_theta=lb_theta)

plot_data = EllipsoidTubeData2D(
    center_data = X_init,
    ellipsoid_data = None
)
add_plot_trajectory(ax, plot_data, prob_tighten=prob_tighten)

# +
# without gp model
ocp_zoro_opts = {
    "solver_options": {
        "integrator_type": "DISCRETE",
        "qp_solver": "PARTIAL_CONDENSING_HPIPM",
        "hessian_approx": "GAUSS_NEWTON",
        "nlp_solver_type": "SQP_RTI"
    }
}

ocp_zoro_nogp = export_ocp_nominal(N, T)
ocp_zoro_nogp.solver_options.integrator_type = "DISCRETE"
ocp_zoro_nogp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
ocp_zoro_nogp.solver_options.hessian_approx = "GAUSS_NEWTON"
ocp_zoro_nogp.solver_options.nlp_solver_type = "SQP_RTI"

# +
# zoro_solver_nogp = ZoroAcados(ocp_zoro_nogp, sim, prob_x, Sigma_x0, Sigma_W+Sigma_GP_prior)
from zero_order_gpmpc.zoro_acados_utils import generate_h_tightening_funs_SX

idh_tight = [0]
h_jac_x_fun, h_tighten_fun, h_tighten_jac_x_fun, h_tighten_jac_sig_fun = generate_h_tightening_funs_SX(ocp_zoro_nogp.model.con_h_expr, ocp_zoro_nogp.model.x, ocp_zoro_nogp.model.u, ocp_zoro_nogp.model.p, idh_tight)

zoro_solver_nogp = ZoroAcados(
    ocp_zoro_nogp, sim, prob_x, Sigma_x0, Sigma_W, 
    h_tightening_jac_sig_fun=h_tighten_jac_sig_fun
)

for i in range(N):
    zoro_solver_nogp.ocp_solver.set(i, "x",X_init[i,:])
    zoro_solver_nogp.ocp_solver.set(i, "u",U_init[i,:])
zoro_solver_nogp.ocp_solver.set(N, "x",X_init[N,:])

# +
zoro_solver_nogp.solve()

X_nogp,U_nogp,P_nogp = zoro_solver_nogp.get_solution()
# -

zoro_solver_nogp.solve_stats

# +
n_iter = zoro_solver_nogp.solve_stats["n_iter"]

time_other = 0.0
for key, t in zoro_solver_nogp.solve_stats["timings"].items():
    if key != "total":
        time_other -= t
    else:
        time_other += t
    print(f"{key:20s}: {1000*t:8.3f}ms ({n_iter} calls), {1000*t/n_iter:8.3f}ms (1 call)")

key = "other"
t = time_other
print(f"{key:20s}: {1000*t:8.3f}ms ({n_iter} calls), {1000*t/n_iter:8.3f}ms (1 call)")
# -

zoro_solver_nogp.print_solve_stats()

# +

fig, ax = base_plot(lb_theta=lb_theta)

# plot_data = EllipsoidTubeData2D(
#     center_data = X,
#     ellipsoid_data = np.array(P)
#     # ellipsoid_data = None
# )
# add_plot_trajectory(ax, plot_data, color_fun=plt.cm.Reds)

plot_data = EllipsoidTubeData2D(
    center_data = X_nogp,
    ellipsoid_data = np.array(P_nogp)
    # ellipsoid_data = None
)
add_plot_trajectory(ax, plot_data, color_fun=plt.cm.Blues, prob_tighten=prob_tighten)

# +
# generate training data for GP with augmented model

# "real model"
model_actual = export_simplependulum_ode_model(noise=False) # TODO: Revert NOISE-FIX
model_actual.f_expl_expr = model_actual.f_expl_expr + cas.vertcat(
    cas.DM(0),
    -0.5*cas.sin((model_actual.x[0])**2)
)
# model_actual.f_expl_expr = model_actual.f_expl_expr # EXACT MODEL (zero - GP) # TODO: Revert NOISE-FIX
model_actual.f_impl_expr = model_actual.xdot - model_actual.f_expl_expr
model_actual.name = model_actual.name + "_actual"

# acados integrator
sim_actual = AcadosSim()
sim_actual.model = model_actual
sim_actual.solver_options.integrator_type = "ERK"
# sim.parameter_values = np.zeros((nx,1)) # TODO: Revert NOISE-FIX

# set prediction horizon
sim_actual.solver_options.Tsim = dT

# acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp_' + model.name + '.json')
acados_integrator_actual = AcadosSimSolver(sim_actual, json_file = 'acados_sim_' + model_actual.name + '.json')

# +
import gp_hyperparam_training
importlib.reload(gp_hyperparam_training)
from gp_hyperparam_training import generate_train_inputs_zoro, generate_train_outputs_zoro, train_gp_model, get_gp_param_names_values, set_gp_param_value

random_seed = 817238
N_sim_per_x0 = 1
N_x0 = 10
x0_rand_scale = 0.1

x_train, x0_arr = generate_train_inputs_zoro(zoro_solver_nogp, x0, N_sim_per_x0, N_x0, random_seed=random_seed, x0_rand_scale=x0_rand_scale)
y_train = generate_train_outputs_zoro(x_train, acados_integrator, acados_integrator_actual, Sigma_W)

# +
import torch
import gpytorch
import tqdm

import gp_utils, gp_model
importlib.reload(gp_utils)
importlib.reload(gp_model)

from gp_utils import gp_data_from_model_and_path, gp_derivative_data_from_model_and_path, plot_gp_data, generate_grid_points
from gp_hyperparam_training import train_gp_model, get_gp_param_names_values, set_gp_param_value
from gp_model import IndependentGPModel, MultitaskGPModel

nout = nx

x_train_tensor = torch.Tensor(x_train)
y_train_tensor = torch.Tensor(y_train)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks = nout
)
gp_model = MultitaskGPModel(x_train_tensor, y_train_tensor, likelihood, nout, rank = nout)

list(get_gp_param_names_values(gp_model))

# +

training_iterations = 500
rng_seed = 124145

# set task covariance to identity
# gp_model.likelihood.raw_noise.requires_grad = False
# set_gp_param_value(gp_model, "likelihood.raw_noise", torch.Tensor([w_theta]))

# gp_model.covar_module.task_covar_module.covar_factor.requires_grad = False
# set_gp_param_value(gp_model, "covar_module.task_covar_module.covar_factor", torch.Tensor(np.eye(nx)))

# # gp_model.covar_module.task_covar_module.raw_var.requires_grad = False
# # set_gp_param_value(gp_model, "covar_module.task_covar_module.raw_var", torch.Tensor(np.array([0.1, 0.1])))
# gp_model.covar_module.data_covar_module.raw_lengthscale.requires_grad = False
# set_gp_param_value(gp_model, "covar_module.data_covar_module.raw_lengthscale", torch.Tensor(np.array([[1, 1]])))

# gp_model.covar_module.data_covar_module.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(0,10))
# gp_model.covar_module.data_covar_module.register_constraint("raw_lengthscale", gpytorch.constraints.GreaterThan(0))

gp_model, likelihood = train_gp_model(gp_model, torch_seed=rng_seed, training_iterations=training_iterations)

# save GP model training
# gp_model_train_dict = gp_model.state_dict().copy()
# likelihood_train = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nout)
# gp_model_train = MultitaskGPModel(x_train_tensor, y_train_tensor, likelihood, nout, rank = nout)
# gp_model_train.load_state_dict(gp_model_train_dict)

# EVAL MODE
gp_model.eval()
likelihood.eval()
# -

list(get_gp_param_names_values(gp_model))

gp_model.state_dict()

# +
num_samples = 5
use_likelihood = False

num_points_between_samples = 30
t_lin = np.linspace(0,1,num_points_between_samples,endpoint=False)

x_plot_waypts = np.hstack((
    X_nogp[1:,:],
    U_nogp
)) 
x_plot = []
for i in range(x_plot_waypts.shape[0]-1):
    x_plot += [x_plot_waypts[i,:] + (x_plot_waypts[i+1,:] - x_plot_waypts[i,:]) * t for t in t_lin]
x_plot = np.vstack(zip(x_plot))

gp_data = gp_data_from_model_and_path(gp_model, likelihood, x_plot, num_samples=num_samples, use_likelihood=use_likelihood)
plot_gp_data([gp_data], marker_size_lim=[1, 15])
# -

gp_derivative_data = gp_derivative_data_from_model_and_path(gp_model, likelihood, x_plot, num_samples=0)
plot_gp_data([gp_derivative_data], marker_size_lim=[5, 20], plot_train_data=False)

# +
# plot along axis
x_dim_lims = np.array([
    [0, np.pi],
    [-2, 1],
    [-2, 2]
    ])
x_dim_slice = np.array([
    1 * np.pi,
    0,
    0
])
x_dim_plot = 2
x_grid = generate_grid_points(x_dim_lims, x_dim_slice, x_dim_plot, num_points=800)

gp_grid_data = gp_data_from_model_and_path(gp_model, likelihood, x_grid, num_samples=num_samples, use_likelihood=use_likelihood)
fig, ax = plot_gp_data([gp_grid_data], marker_size_lim=[5, 50])

y_lim_0 = ax[0].get_ylim()
y_lim_1 = ax[1].get_ylim()

# +
gp_derivative_grid_data = gp_derivative_data_from_model_and_path(gp_model, likelihood, x_grid, num_samples=0)
fig, ax = plot_gp_data([gp_derivative_grid_data], marker_size_lim=[5, 50], plot_train_data=False)

ax[0].set_ylim(*y_lim_0)
ax[1].set_ylim(*y_lim_1)
plt.draw()
# -

ocp_zoro = export_ocp_nominal(N, T)
zoro_solver = ZoroAcados(ocp_zoro, sim, prob_x, Sigma_x0, Sigma_W, gp_model=gp_model)

# +
# initial condition
# initialize nominal variables (initial guess)
for i in range(N):
    zoro_solver.ocp_solver.set(i, "x",X_init[i,:])
    zoro_solver.ocp_solver.set(i, "u",U_init[i,:])
zoro_solver.ocp_solver.set(N, "x",X_init[N,:])

zoro_solver.solve()
# -

zoro_solver.print_solve_stats()

# +
X,U,P = zoro_solver.get_solution()

fig, ax = base_plot(lb_theta=lb_theta)

# plot_data = EllipsoidTubeData2D(
#     center_data = simX_GP,
#     ellipsoid_data = np.array(P_mat_list_GP)
#     # ellipsoid_data = None
# )
# add_plot_trajectory(ax, plot_data, color_fun=plt.cm.Oranges)

plot_data = EllipsoidTubeData2D(
    center_data = X,
    ellipsoid_data = np.array(P)
    # ellipsoid_data = None
)
add_plot_trajectory(ax, plot_data, color_fun=plt.cm.Reds)

# +
ocp_zoro_gpprior = export_ocp_nominal(N, T)
ocp_zoro_gpprior = set_ocp_options(ocp_zoro_gpprior, ocp_zoro_opts)

gp_model.eval()
y_test = torch.Tensor(np.array([[1,1,1]]))
Sigma_GP_prior = gp_model.covar_module(y_test).numpy()

Sigma_GP_prior
# -

Sigma_W

Sigma_W+Sigma_GP_prior

# +
zoro_solver_gpprior = ZoroAcados(ocp_zoro_gpprior, sim, prob_x, Sigma_x0, Sigma_W+Sigma_GP_prior)

for i in range(N):
    zoro_solver_gpprior.ocp_solver.set(i, "x",X_init[i,:])
    zoro_solver_gpprior.ocp_solver.set(i, "u",U_init[i,:])
zoro_solver_gpprior.ocp_solver.set(N, "x",X_init[N,:])

# +
zoro_solver_gpprior.solve()

X_gpprior,U_gpprior,P_gpprior = zoro_solver_gpprior.get_solution()

# +

fig, ax = base_plot(lb_theta=lb_theta)

plot_data = EllipsoidTubeData2D(
    center_data = X_gpprior,
    ellipsoid_data = np.array(P_gpprior)
    # ellipsoid_data = None
)
add_plot_trajectory(ax, plot_data, color_fun=plt.cm.Blues)

plot_data = EllipsoidTubeData2D(
    center_data = X,
    ellipsoid_data = np.array(P)
    # ellipsoid_data = None
)
add_plot_trajectory(ax, plot_data, color_fun=plt.cm.Reds)
# -


