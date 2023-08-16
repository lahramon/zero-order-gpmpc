# %% [markdown]
#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# %%
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi import SX, MX, vertcat, sin, cos, Function
from scipy.linalg import block_diag


def export_linear_model(nx, nu):
    model_name = "linear_model_with_params"

    # linear dynamics for every stage
    A = MX.sym("A",nx,nx)
    B = MX.sym("B",nx,nu)
    x = MX.sym("x",nx,1)
    u = MX.sym("x",nu,1)
    w = MX.sym("x",nx,1)
    xdot = MX.sym("x",nx,1)

    f_expl = A @ x + B @ u + w
    f_impl = xdot - f_expl

    # parameters
    p = vertcat(
        A.reshape((nx**2,1)),
        B.reshape((nx*nu,1)),
        w
    )

    # acados model
    model = AcadosModel()
    model.disc_dyn_expr = f_expl
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    # model.con_h_expr = con_h_expr
    model.name = model_name

    return model

# %%
def export_simplependulum_ode_model(noise=False):

    model_name = 'simplependulum_ode'

    # set up states & controls
    theta   = SX.sym('theta')
    dtheta  = SX.sym('dtheta')
    
    x = vertcat(theta, dtheta)
    u = SX.sym('u')
    
    # xdot
    theta_dot   = SX.sym('theta_dot')
    dtheta_dot  = SX.sym('dtheta_dot')
    xdot = vertcat(theta_dot, dtheta_dot)

    # algebraic variables
    # z = None

    # parameters
    p = []
    if noise:
        w = SX.sym("w",2,1)
        p += [w]
        p = vertcat(*p)
    
    # dynamics
    f_expl = vertcat(
            dtheta,
            -sin(theta) + u
        )
    if noise:
        f_expl += w
    
    f_impl = xdot - f_expl

    # constraints
    con_h_expr = vertcat(
        theta, # theta
    )

    # acados model
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.con_h_expr = con_h_expr
    model.name = model_name

    return model

# %%
def export_ode_model_with_discrete_rk4(model, dT):

    x = model.x
    u = model.u

    ode = Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1,u)
    k3 = ode(x+dT/2*k2,u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf
    print("built RK4 for pendulum model with dT = ", dT)
    print(xf)
    return model

# %%
def export_augmented_pendulum_model():
    # pendulum model augmented with algebraic variable just for testing
    model = export_pendulum_ode_model()
    model_name = 'augmented_pendulum'

    z = SX.sym('z')

    f_impl = vertcat( model.xdot - model.f_expl_expr, \
        z - model.u*model.u)

    model.f_impl_expr = f_impl
    model.z = z
    model.name = model_name

    return model

def export_robust_ode_model_with_constraints():

    model = export_simplependulum_ode_model(noise=True)

    # add constraints

    model.con_h_expr = vertcat(
        model.x[0], # theta
    )
    
    # constraints = types.SimpleNamespace()
    # constraints.lh = np.array([0])
    # constraints.uh = np.array([np.pi])

    # robustify


    return model

def export_ocp_nominal(N, T, ocp_opts=None):
    # constraints
    x0 = np.array([np.pi, 0])
    lb_u = -2.0
    ub_u = 2.0
    lb_theta = (0./360.) * 2 * np.pi
    ub_theta = (200./360.) * 2 * np.pi

    # cost
    cost_theta = 5
    cost_omega = 1
    cost_fac_e = 4
    Q = np.diagflat(np.array([cost_theta, cost_omega]))
    R = np.array(1)

    model = export_simplependulum_ode_model(noise=False)

    # generate acados OCP for INITITIALIZATION
    ocp = AcadosOcp()

    ocp.model = model
    ocp.dims.N = N

    # dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    ocp.dims.nx = nx
    ocp.dims.nu = nu

    # cost
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = block_diag(Q, R)
    ocp.cost.W_0 = ocp.cost.W
    ocp.cost.W_e = cost_fac_e * Q

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    ocp.cost.Vu = np.zeros((ny, nu))
    ocp.cost.Vu[nx:nx+nu, :] = np.eye(nu)

    ocp.cost.Vx_e = np.eye(ny_e)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # constraints
    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = np.array([lb_u])
    ocp.constraints.ubu = np.array([ub_u])
    ocp.constraints.idxbu = np.array(range(nu))

    ocp.constraints.uh = np.array([
        ub_theta
    ])
    ocp.constraints.lh = np.array([
        lb_theta
    ])

    # terminal constraints
    ocp.constraints.C_e = ocp.constraints.C
    ocp.constraints.lg_e = ocp.constraints.lg
    ocp.constraints.ug_e = ocp.constraints.ug

    ocp.constraints.x0 = x0

    # solver options

    ocp.solver_options.integrator_type = 'ERK'
    # ocp.solver_options.integrator_type = 'DISCRETE' # 'IRK'

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'

    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    # ocp.solver_options.hessian_approx = 'EXACT'

    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    # ocp.solver_options.nlp_solver_type = 'SQP' # , SQP_RTI

    ocp.solver_options.tf = T
        
    return ocp



