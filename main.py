#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

This experiment runs a closed–loop simulation where an LQR controller is applied to the system:
    xₜ₊₁ = A xₜ + B uₜ + wₜ,    yₜ = C xₜ + vₜ,
    
A steady–state LQR gain is computed offline using specified cost matrices Q_lqr and R_lqr.
The nominal parameters are obtained via EM (covariances only) while the known mean
vectors (x0_mean, mu_w, mu_v) are used for the means.
Then, five filters (KF, KF_inf, DRKF, BCOT, and risk–sensitive filter)
(from the folder LQR_with_estimator) are run in closed–loop with exactly the same controller
for each candidate robust parameter value.
For each simulation run, the mean squared error (MSE) trajectory and the LQR cost
computed as:
    J = Σₜ [(x[t]-x_d[t])ᵀ Q_lqr (x[t]-x_d[t]) + u[t]ᵀ R_lqr u[t]] + (x[T]-x_d[T])ᵀ Q_lqr (x[T]-x_d[T])
are returned. For each filter the candidate robust parameter with the lowest average
LQR cost is chosen as “optimal.” Finally, the performance (averaged MSE over the entire
horizon and cost) of each filter using its optimal robust parameter is saved for comparison,
and a 2D trajectory plot is saved.
    
Usage example:
    python main7.py --dist normal
    python main7.py --dist quadratic
"""

import numpy as np
import argparse
import os
import pickle
from joblib import Parallel, delayed
from pykalman import KalmanFilter
from scipy.linalg import solve_discrete_are
from scipy.linalg import expm
import matplotlib.pyplot as plt

# Import filter implementations.
from LQR_with_estimator.KF import KF
from LQR_with_estimator.KF_inf import KF_inf
from LQR_with_estimator.DRKF_ours_inf import DRKF_ours_inf
from LQR_with_estimator.BCOT import BCOT 
from LQR_with_estimator.risk_sensitive import RiskSensitive

# --- Distribution Sampling Functions ---
def normal(mu, Sigma, N=1):
    return np.random.multivariate_normal(mu[:, 0], Sigma, size=N).T

def uniform(a, b, N=1):
    n = a.shape[0]
    x = a + (b - a) * np.random.rand(N, n)
    return x.T

def quad_inverse(x, b, a):
    row, col = x.shape
    for i in range(row):
        for j in range(col):
            beta = (a[j] + b[j]) / 2.0
            alpha = 12.0 / ((b[j] - a[j]) ** 3)
            tmp = 3 * x[i, j] / alpha - (beta - a[j]) ** 3
            if tmp >= 0:
                x[i, j] = beta + tmp ** (1.0/3.0)
            else:
                x[i, j] = beta - (-tmp) ** (1.0/3.0)
    return x

def quadratic(w_max, w_min, N=1):
    n = w_min.shape[0]
    x = np.random.rand(N, n)
    x = quad_inverse(x, w_max, w_min)
    return x.T

# --- True Data Generation ---
def generate_data(T, nx, ny, A, C, mu_w, Sigma_w, mu_v, M,
                  x0_mean, x0_cov, x0_max, x0_min, w_max, w_min, v_max, v_min, dist):
    x_true_all = np.zeros((T + 1, nx, 1))
    y_all = np.zeros((T, ny, 1))
    if dist == "normal":
        x_true = normal(x0_mean, x0_cov)
    elif dist == "quadratic":
        x_true = quadratic(x0_max, x0_min)
    x_true_all[0] = x_true
    for t in range(T):
        if dist == "normal":
            true_w = normal(mu_w, Sigma_w)
            true_v = normal(mu_v, M)
        elif dist == "quadratic":
            true_w = quadratic(w_max, w_min)
            true_v = quadratic(v_max, v_min)
        y_t = C @ x_true + true_v
        y_all[t] = y_t
        x_true = A @ x_true + true_w
        x_true_all[t+1] = x_true
    return x_true_all, y_all

def save_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def is_stabilizable(A, B, tol=1e-9):
    n = A.shape[0]
    eigenvals, _ = np.linalg.eig(A)
    for eig in eigenvals:
        if np.abs(eig) >= 1 - tol:
            M_mat = np.hstack([eig * np.eye(n) - A, B])
            if np.linalg.matrix_rank(M_mat, tol) < n:
                return False
    return True

def is_detectable(A, C, tol=1e-9):
    n = A.shape[0]
    eigenvals, _ = np.linalg.eig(A)
    for eig in eigenvals:
        if np.abs(eig) >= 1 - tol:
            M_mat = np.vstack([eig * np.eye(n) - A, C])
            if np.linalg.matrix_rank(M_mat, tol) < n:
                return False
    return True

def is_positive_definite(M, tol=1e-9):
    if not np.allclose(M, M.T, atol=tol):
        return False
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False

def enforce_positive_definiteness(Sigma, epsilon=1e-4):
    Sigma = (Sigma + Sigma.T) / 2
    eigvals = np.linalg.eigvalsh(Sigma)
    min_eig = np.min(eigvals)
    if min_eig < epsilon:
        Sigma += (epsilon - min_eig) * np.eye(Sigma.shape[0])
    return Sigma

# --- LQR Controller Computation ---
def compute_lqr_gain(A, B, Q_lqr, R_lqr):
    P = solve_discrete_are(A, B, Q_lqr, R_lqr)
    K = np.linalg.inv(B.T @ P @ B + R_lqr) @ (B.T @ P @ A)
    return K

# --- Helper Function to Generate Desired Trajectory ---
def generate_desired_trajectory(T_total):
    Amp = 5.0      # Amplitude for curvy (sine) trajectory.
    slope = 1.0    # Slope for linear trajectory.
    radius = 5.0   # Radius for circular trajectory.
    omega = 0.5    # Angular frequency.
    dt = 0.2
    time_steps = int(T_total / dt) + 1
    time = np.linspace(0, T_total, time_steps)

    x_d = Amp * np.sin(omega * time)
    vx_d = Amp * omega * np.cos(omega * time)
    y_d = slope * time
    vy_d = slope * np.ones(time_steps)

    return np.vstack((x_d, vx_d, y_d, vy_d))

# --- Modified LQR Cost Computation for Trajectory Tracking ---
def compute_lqr_cost(result, Q_lqr, R_lqr, K_lqr, desired_traj):
    x = result['state_traj']
    x_est = result['est_state_traj']
    T_steps = x.shape[0]
    cost = 0.0
    for t in range(T_steps):
        error = x[t] - desired_traj[:, t].reshape(-1, 1)
        u = -K_lqr @ (x_est[t] - desired_traj[:, t].reshape(-1, 1))
        cost += (error.T @ Q_lqr @ error)[0, 0] + (u.T @ R_lqr @ u)[0, 0]
    return cost

# --- Modified Experiment Function ---
def run_experiment(exp_idx, dist, num_sim, seed_base, robust_val, T_total, desired_traj):
    np.random.seed(seed_base + exp_idx)
    
    dt = 0.2
    time_steps = int(T_total / dt) + 1
    T = time_steps - 1  # simulation horizon
    
    # Use the initial desired state as the system's initial state.
    initial_trajectory = desired_traj[:, 0].reshape(-1, 1)
    
    # Discrete-time system dynamics (4D double integrator)
    A = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    B = np.array([[0.5 * dt**2, 0],
                  [dt, 0],
                  [0, 0.5 * dt**2],
                  [0, dt]])
    C = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    
    system_data = (A, C)
    
    nx = 4; nw = 4; nu = 2; ny = 2
    if dist == "normal":
        mu_w = 0.0 * np.ones((nw, 1))
        Sigma_w = 0.01 * np.eye(nw)
        x0_mean = 0.0 * np.ones((nx, 1))
        x0_cov = 0.01 * np.eye(nx)
        mu_v = 0.0 * np.ones((ny, 1))
        Sigma_v = 0.01 * np.eye(ny)
        v_max = v_min = None
        w_max = w_min = x0_max = x0_min = None
    elif dist == "quadratic":
        w_max = 0.1 * np.ones(nx)
        w_min = -0.1 * np.ones(nx)
        mu_w = (0.5 * (w_max + w_min))[:, None]
        Sigma_w = 3.0/20.0 * np.diag((w_max - w_min)**2)
        x0_max = 0.1 * np.ones(nx)
        x0_min = -0.1 * np.ones(nx)
        x0_mean = (0.5 * (x0_max + x0_min))[:, None]
        x0_cov = 3.0/20.0 * np.diag((x0_max - x0_min)**2)
        v_min = -0.1 * np.ones(ny)
        v_max = 0.1 * np.ones(ny)
        mu_v = (0.5 * (v_max + v_min))[:, None]
        Sigma_v = 3.0/20.0 * np.diag((v_max - v_min)**2)
    else:
        raise ValueError("Unsupported noise distribution.")
    
    # --- Generate Data for EM ---
    N_data = 5
    _, y_all_em = generate_data(N_data, nx, ny, A, C,
                                mu_w, Sigma_w, mu_v, Sigma_v,
                                x0_mean, x0_cov, x0_max, x0_min,
                                w_max, w_min, v_max, v_min, dist)
    y_all_em = y_all_em.squeeze()
    
    # --- EM Estimation of Nominal Covariances ---
    mu_w_hat = np.zeros((nx, 1))
    mu_v_hat = np.zeros((ny, 1))
    mu_x0_hat = x0_mean.copy()
    Sigma_w_hat = np.eye(nx)
    Sigma_v_hat = np.eye(ny)
    Sigma_x0_hat = x0_cov.copy()
    
    kf_em = KalmanFilter(transition_matrices=A,
                           observation_matrices=C,
                           transition_covariance=Sigma_w_hat,
                           observation_covariance=Sigma_v_hat,
                           transition_offsets=mu_w_hat.squeeze(),
                           observation_offsets=mu_v_hat.squeeze(),
                           initial_state_mean=mu_x0_hat.squeeze(),
                           initial_state_covariance=Sigma_x0_hat,
                           em_vars=['transition_covariance', 'observation_covariance',
                                    'transition_offsets', 'observation_offsets'])
    max_iter = 100
    eps_log = 1e-4
    loglikelihoods = np.zeros(max_iter)
    for i in range(max_iter):
        kf_em = kf_em.em(X=y_all_em, n_iter=1)
        loglikelihoods[i] = kf_em.loglikelihood(y_all_em)
        Sigma_w_hat = kf_em.transition_covariance
        Sigma_v_hat = kf_em.observation_covariance
        mu_x0_hat = kf_em.initial_state_mean
        Sigma_x0_hat = kf_em.initial_state_covariance
        if i > 0 and (loglikelihoods[i] - loglikelihoods[i-1] <= eps_log):
            break
    
    Sigma_w_hat = enforce_positive_definiteness(Sigma_w_hat)
    Sigma_v_hat = enforce_positive_definiteness(Sigma_v_hat)
    Sigma_x0_hat = enforce_positive_definiteness(Sigma_x0_hat)
    
    nominal_mu_w    = mu_w
    nominal_Sigma_w = Sigma_w_hat.copy()
    nominal_mu_v    = mu_v
    nominal_Sigma_v = Sigma_v_hat.copy()
    nominal_x0_mean = x0_mean
    nominal_x0_cov  = Sigma_x0_hat.copy()
    
    # --- Compute LQR Gain ---
    Q_lqr = np.diag([10, 1, 10, 1])
    R_lqr = 0.1 * np.eye(2)
    K_lqr = compute_lqr_gain(A, B, Q_lqr, R_lqr)
    if not is_stabilizable(A, B):
        print("Warning: (A, B) is not stabilizable!")
        exit()
    if not is_detectable(A, C):
        print("Warning: (A, C) is not detectable!")
        exit()
    if not is_positive_definite(nominal_Sigma_w):
        print("Warning: nominal_Sigma_w is not positive definite!")
        exit()
    if not is_positive_definite(nominal_Sigma_v):
        print("Warning: nominal_M (noise covariance) is not positive definite!")
        exit()
    
    # --- Simulation Functions for Each Filter ---
    def run_simulation_finite(sim_idx_local):
        estimator = KF(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_M=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_M=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track(desired_traj)
        cost = compute_lqr_cost(res, Q_lqr, R_lqr, K_lqr, desired_traj)
        return res, cost

    def run_simulation_inf_kf(sim_idx_local):
        estimator = KF_inf(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_M=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_M=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track(desired_traj)
        cost = compute_lqr_cost(res, Q_lqr, R_lqr, K_lqr, desired_traj)
        return res, cost

    def run_simulation_inf_drkf(sim_idx_local):
        estimator = DRKF_ours_inf(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_Sigma_v=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min,
            theta_x=robust_val, theta_v=robust_val)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track(desired_traj)
        cost = compute_lqr_cost(res, Q_lqr, R_lqr, K_lqr, desired_traj)
        return res, cost

    def run_simulation_bcot(sim_idx_local):
        estimator = BCOT(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_M=Sigma_v,
            nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=nominal_mu_v, nominal_M=nominal_Sigma_v,
            radius=robust_val, maxit=20,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track(desired_traj)
        cost = compute_lqr_cost(res, Q_lqr, R_lqr, K_lqr, desired_traj)
        return res, cost

    def run_simulation_risk(sim_idx_local):
        estimator = RiskSensitive(
            T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
            true_x0_mean=x0_mean, true_x0_cov=x0_cov,
            true_mu_w=mu_w, true_Sigma_w=Sigma_w,
            true_mu_v=mu_v, true_M=Sigma_v,
            nominal_x0_mean=x0_mean,  # known initial state mean
            nominal_x0_cov=nominal_x0_cov,
            nominal_mu_w=mu_w,        # known process noise mean
            nominal_Sigma_w=nominal_Sigma_w,
            nominal_mu_v=mu_v,        # known measurement noise mean
            nominal_M=nominal_Sigma_v,
            theta_rs=robust_val,
            x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min, v_max=v_max, v_min=v_min)
        estimator.K_lqr = K_lqr
        res = estimator.forward_track(desired_traj)
        cost = compute_lqr_cost(res, Q_lqr, R_lqr, K_lqr, desired_traj)
        return res, cost

    results_finite = [run_simulation_finite(i) for i in range(num_sim)]
    mse_mean_finite = np.mean([np.mean(r[0]['mse']) for r in results_finite])
    cost_mean_finite = np.mean([r[1] for r in results_finite])
    rep_state_finite = results_finite[0][0]['state_traj']

    results_inf = [run_simulation_inf_kf(i) for i in range(num_sim)]
    mse_mean_inf = np.mean([np.mean(r[0]['mse']) for r in results_inf])
    cost_mean_inf = np.mean([r[1] for r in results_inf])
    rep_state_inf = results_inf[0][0]['state_traj']

    results_drkf = [run_simulation_inf_drkf(i) for i in range(num_sim)]
    mse_mean_drkf = np.mean([np.mean(r[0]['mse']) for r in results_drkf])
    cost_mean_drkf = np.mean([r[1] for r in results_drkf])
    rep_state_drkf = results_drkf[0][0]['state_traj']

    results_bcot = [run_simulation_bcot(i) for i in range(num_sim)]
    mse_mean_bcot = np.mean([np.mean(r[0]['mse']) for r in results_bcot])
    cost_mean_bcot = np.mean([r[1] for r in results_bcot])
    rep_state_bcot = results_bcot[0][0]['state_traj']

    results_risk = [run_simulation_risk(i) for i in range(num_sim)]
    mse_mean_risk = np.mean([np.mean(r[0]['mse']) for r in results_risk])
    cost_mean_risk = np.mean([r[1] for r in results_risk])
    rep_state_risk = results_risk[0][0]['state_traj']
    
    overall_results = {
        'finite': mse_mean_finite,
        'finite_state': rep_state_finite,
        'inf': mse_mean_inf,
        'inf_state': rep_state_inf,
        'drkf_inf': mse_mean_drkf,
        'drkf_inf_state': rep_state_drkf,
        'bcot': mse_mean_bcot,
        'bcot_state': rep_state_bcot,
        'risk': mse_mean_risk,
        'risk_state': rep_state_risk,
        'cost': {
            'finite': cost_mean_finite,
            'inf': cost_mean_inf,
            'drkf_inf': cost_mean_drkf,
            'bcot': cost_mean_bcot,
            'risk': cost_mean_risk
        }
    }
    return overall_results, {
        'finite': results_finite,
        'inf': results_inf,
        'drkf_inf': results_drkf,
        'bcot': results_bcot,
        'risk': results_risk
    }

# --- Main Routine ---
def main(dist, num_sim, num_exp, T_total):
    seed_base = 2024
    if dist=='normal':
        robust_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]
    elif dist=='quadratic':
        robust_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]
    desired_traj = generate_desired_trajectory(T_total)
    
    # Define filters (the same keys are used for cost and state trajectories)
    filters = ['finite', 'inf', 'drkf_inf', 'bcot', 'risk']
    filter_labels = {
        'finite': "Time-varying Kalman filter",
        'inf': "Time-invariant Kalman filter",
        'bcot': "BCOT filter",
        'risk': "Risk-Sensitive filter",
        'drkf_inf': "DR Kalman filter (ours)"
    }
    
    all_results = {}
    raw_experiments_data = {}   # Store raw experiments for each robust candidate.
    for robust_val in robust_vals:
        print(f"Running experiments for robust parameter = {robust_val}")
        experiments = Parallel(n_jobs=-1)(
            delayed(run_experiment)(exp_idx, dist, num_sim, seed_base, robust_val, T_total, desired_traj)
            for exp_idx in range(num_exp)
        )
        # Unpack overall results from the tuple returned by run_experiment.
        overall_experiments = [exp[0] for exp in experiments]
        cost_keys = filters  # same as ['finite', 'inf', 'drkf_inf', 'bcot', 'risk']
        all_cost = [exp['cost'] for exp in overall_experiments]
        all_mse = {key: [] for key in cost_keys}
        for exp in overall_experiments:
            for key in cost_keys:
                all_mse[key].append(np.mean(exp[key]))
        final_cost = {key: np.mean([ec[key] for ec in all_cost]) for key in cost_keys}
        final_cost_std = {key: np.std([ec[key] for ec in all_cost]) for key in cost_keys}
        final_mse = {key: np.mean(all_mse[key]) for key in cost_keys}
        final_mse_std = {key: np.std(all_mse[key]) for key in cost_keys}
        # Store representative state trajectories from the first experiment run for this robust value
        rep_state = {filt: overall_experiments[0][f"{filt}_state"] for filt in filters}
        all_results[robust_val] = {
            'cost': final_cost,
            'cost_std': final_cost_std,
            'mse': final_mse,
            'mse_std': final_mse_std,
            'state': rep_state
        }
        # Save the raw experiments for this candidate robust parameter.
        raw_experiments_data[robust_val] = [exp[1] for exp in experiments]
        print(f"Candidate robust parameter {robust_val}: Cost = {final_cost}, Average MSE = {final_mse}")
    
    optimal_results = {}
    for f in filters:
        if f in ['finite', 'inf']:
            candidate = list(all_results.values())[0]
            optimal_results[f] = {
                'robust_val': "N/A",
                'cost': candidate['cost'][f],
                'mse': candidate['mse'][f]
            }
        else:
            best_val = None
            best_cost = np.inf
            for robust_val, res in all_results.items():
                current_cost = res['cost'][f]
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_val = robust_val
            optimal_results[f] = {
                'robust_val': best_val,
                'cost': best_cost,
                'mse': all_results[best_val]['mse'][f]
            }
        print(f"Optimal robust parameter for {f}: {optimal_results[f]['robust_val']}")
    
    sorted_optimal = sorted(optimal_results.items(), key=lambda item: item[1]['cost'])
    print("\nSummary of Optimal Results (sorted by LQR cost):")
    for filt, info in sorted_optimal:
        print(f"{filt}: Optimal robust parameter = {info['robust_val']}, LQR cost = {info['cost']:.4f}, Average MSE = {info['mse']:.4f}")
    
    results_path = "./results/estimator/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    save_data(os.path.join(results_path, f'overall_results_{dist}_{dist}.pkl'), all_results)
    save_data(os.path.join(results_path, f'optimal_results_{dist}_{dist}.pkl'), optimal_results)
    # Save raw experiments data.
    save_data(os.path.join(results_path, f'raw_experiments_{dist}_{dist}.pkl'), raw_experiments_data)
    print("LQR with state estimation experiments completed for all robust parameters.")
    
    # --- Print Readable Data for the User ---
    print("\nDetailed Experiment Results:")
    header = "{:<35} {:<25} {:<25} {:<15}".format("Method", "LQR Cost", "Average MSE", "Best theta")
    print(header)
    print("-"*100)
    for filt in filters:
        best_theta = optimal_results[filt]['robust_val']
        cost = optimal_results[filt]['cost']
        mse = optimal_results[filt]['mse']
        print("{:<35} {:<25.1f} {:<25.3f} {:<15}".format(filter_labels[filt], cost, mse, best_theta))
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', default="normal", type=str,
                        help="Uncertainty distribution (normal or quadratic)")
    parser.add_argument('--num_sim', default=1, type=int,
                        help="Number of simulation runs per experiment")
    parser.add_argument('--num_exp', default=20, type=int,
                        help="Number of independent experiments")
    parser.add_argument('--time', default=10, type=int,
                        help="Total simulation time")
    args = parser.parse_args()
    main(args.dist, args.num_sim, args.num_exp, args.time)
