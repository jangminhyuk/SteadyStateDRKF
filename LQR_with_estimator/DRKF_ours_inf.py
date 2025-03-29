#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRKF_ours_inf.py implements a distributionally robust Kalman filter (DRKF) for state estimation
in a closed-loop LQR experiment. 
"""

import numpy as np
import time
import cvxpy as cp

class DRKF_ours_inf:
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_Sigma_v,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 theta_x=None, theta_v=None):
        """
        Parameters:
          T             : Horizon length.
          dist, noise_dist : Distribution types ('normal' or 'quadratic').
          system_data   : Tuple (A, C).
          B             : Control input matrix.
          
          The following parameters are provided in two sets:
             (i) True parameters (used to simulate the system):
                 - true_x0_mean, true_x0_cov: initial state distribution.
                 - true_mu_w, true_Sigma_w: process noise.
                 - true_mu_v, true_Sigma_v: measurement noise.
             (ii) Nominal parameters (obtained via EM, used in filtering):
                 - Use known means (nominal_x0_mean, nominal_mu_w, nominal_mu_v) and
                   EM–estimated covariances (nominal_x0_cov, nominal_Sigma_w, nominal_Sigma_v).
          x0_max, x0_min, etc.: Bounds for non–normal distributions.
          theta_x, theta_v: DRKF parameters.
        """
        self.T = T
        self.dist = dist
        self.noise_dist = noise_dist
        self.A, self.C = system_data
        self.B = B
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        
        # True parameters.
        self.true_x0_mean = true_x0_mean
        self.true_x0_cov = true_x0_cov
        self.true_mu_w = true_mu_w
        self.true_Sigma_w = true_Sigma_w
        self.true_mu_v = true_mu_v
        self.true_Sigma_v = true_Sigma_v
        
        # Nominal parameters.
        self.nominal_x0_mean = nominal_x0_mean
        self.nominal_x0_cov = nominal_x0_cov
        self.nominal_mu_w = nominal_mu_w
        self.nominal_Sigma_w = nominal_Sigma_w
        self.nominal_mu_v = nominal_mu_v
        self.nominal_Sigma_v = nominal_Sigma_v
        
        # Bounds for sampling true noise.
        self.x0_max = x0_max
        self.x0_min = x0_min
        self.w_max = w_max
        self.w_min = w_min
        self.v_max = v_max
        self.v_min = v_min
        
        # DRKF parameters.
        self.theta_x = theta_x
        self.theta_v = theta_v
        
        # For simulation, generate initial state sample.
        if self.dist == "normal":
            self.x0_init = self.normal(self.true_x0_mean, self.true_x0_cov)
        elif self.dist == "quadratic":
            self.x0_init = self.quadratic(self.x0_max, self.x0_min)
        else:
            raise ValueError("Unsupported distribution for initial state.")
        
        # Similarly, generate an initial measurement sample.
        if self.noise_dist == "normal":
            self.true_v_init = self.normal(self.true_mu_v, self.true_Sigma_v)
        elif self.noise_dist == "quadratic":
            self.true_v_init = self.quadratic(self.v_max, self.v_min)
        else:
            raise ValueError("Unsupported measurement noise distribution.")
        
        # Compute worst-case measurement noise covariance via SDP.
        worst_case_Sigma_v, worst_case_Xprior, status = self.solve_sdp()
        self.wc_Sigma_v = worst_case_Sigma_v
        self.wc_Xprior = worst_case_Xprior
        
        # LQR gain will be assigned externally.
        self.K_lqr = None

    # --- Distribution Sampling Functions ---
    def normal(self, mu, Sigma, N=1):
        return np.random.multivariate_normal(mu[:, 0], Sigma, size=N).T

    def uniform(self, a, b, N=1):
        n = a.shape[0]
        return a + (b - a) * np.random.rand(n, N)

    def quad_inverse(self, x, b, a):
        row, col = x.shape
        for i in range(row):
            for j in range(col):
                beta = (a[j] + b[j]) / 2.0
                alpha = 12.0 / ((b[j] - a[j])**3)
                tmp = 3 * x[i, j] / alpha - (beta - a[j])**3
                if tmp >= 0:
                    x[i, j] = beta + tmp**(1.0/3.0)
                else:
                    x[i, j] = beta - (-tmp)**(1.0/3.0)
        return x

    def quadratic(self, max_val, min_val, N=1):
        n = min_val.shape[0]
        x = np.random.rand(N, n).T  # shape: (n, N)
        return self.quad_inverse(x, max_val, min_val)


    def sample_initial_state(self):
        if self.dist == "normal":
            return self.normal(self.true_x0_mean, self.true_x0_cov)
        elif self.dist == "quadratic":
            return self.quadratic(self.x0_max, self.x0_min)
        else:
            raise ValueError("Unsupported distribution for initial state.")

    def sample_process_noise(self):
        if self.dist == "normal":
            return self.normal(self.true_mu_w, self.true_Sigma_w)
        elif self.dist == "quadratic":
            return self.quadratic(self.w_max, self.w_min)
        else:
            raise ValueError("Unsupported distribution for process noise.")

    def sample_measurement_noise(self):
        if self.noise_dist == "normal":
            return self.normal(self.true_mu_v, self.true_Sigma_v)
        elif self.noise_dist == "quadratic":
            return self.quadratic(self.v_max, self.v_min)
        else:
            raise ValueError("Unsupported distribution for measurement noise.")

    # --- SDP Formulation and Solver for Worst-Case Measurement Covariance ---
    def create_DR_sdp(self):
        # Compute lambda_min for nominal measurement noise covariance (Sigma_v_hat)
        lambda_min_val = np.linalg.eigvalsh(self.nominal_Sigma_v).min()
        
        # Construct the SDP problem.
        # Variables
        X = cp.Variable((self.nx, self.nx), symmetric=True, name='X')
        X_pred = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred')
        Sigma_v = cp.Variable((self.ny, self.ny), symmetric=True, name='Sigma_v')
        X_pred_hat = cp.Variable((self.nx, self.nx), symmetric=True, name='X_pred_hat')
        Y = cp.Variable((self.nx, self.nx), name='Y')
        Z = cp.Variable((self.ny, self.ny), name='Z')
        
        # Parameters
        theta_x = cp.Parameter(nonneg=True, name='theta_x')
        Sigma_v_hat = cp.Parameter((self.ny, self.ny), name='Sigma_v_hat')  # nominal measurement noise covariance
        theta_v = cp.Parameter(nonneg=True, name='theta_v')
        Sigma_w_hat = cp.Parameter((self.nx, self.nx), name='Sigma_w_hat')  # nominal process noise covariance
        
        # Objective: maximize trace(X)
        obj = cp.Maximize(cp.trace(X))
        
        # Constraints using Schur complements and the additional constraint on Sigma_v.
        constraints = [
            cp.bmat([[X_pred - X, X_pred @ self.C.T],
                     [self.C @ X_pred, self.C @ X_pred @ self.C.T + Sigma_v]
                    ]) >> 0,
            cp.trace(X_pred_hat + X_pred - 2*Y) <= theta_x**2,
            cp.bmat([[X_pred_hat, Y],
                     [Y.T, X_pred]
                    ]) >> 0,
            cp.trace(Sigma_v_hat + Sigma_v - 2*Z) <= theta_v**2,
            cp.bmat([[Sigma_v_hat, Z],
                     [Z.T, Sigma_v]
                    ]) >> 0,
            X_pred_hat == self.A @ X @ self.A.T + Sigma_w_hat,                
            X >> 0,
            X_pred >> 0,
            X_pred_hat >> 0,
            Sigma_v >> 0,
            # Sigma_v is larger than lambda_min(Sigma_v_hat)*I
            Sigma_v >> lambda_min_val * np.eye(self.ny)
        ]
        
        prob = cp.Problem(obj, constraints)
        return prob
    
    def solve_sdp(self):
        prob = self.create_DR_sdp()
        params = prob.parameters()
        params[0].value = self.theta_x
        params[1].value = self.nominal_Sigma_v
        params[2].value = self.theta_v
        params[3].value = self.nominal_Sigma_w
        
        prob.solve(solver=cp.MOSEK)
        
        if prob.status in ["infeasible", "unbounded"]:
            print(prob.status, 'DRKF SDP problem')
            
        sol = prob.variables()
        worst_case_Xprior = sol[1].value
        worst_case_Sigma_v = sol[2].value 
        return worst_case_Sigma_v, worst_case_Xprior, prob.status

    # --- DR-KF Update Step ---
    def DR_kalman_filter(self, v_mean_hat, x_pred, y):
        y_pred = self.C @ x_pred + v_mean_hat
        innovation = y - y_pred
        S = self.C @ self.wc_Xprior @ self.C.T + self.wc_Sigma_v
        K = self.wc_Xprior @ self.C.T @ np.linalg.inv(S)
        x_new = x_pred + K @ innovation
        return x_new

    # --- Forward Simulation with LQR Control ---
    def forward(self):
        start_time = time.time()
        T = self.T
        nx = self.nx
        ny = self.ny
        A = self.A
        C = self.C
        B = self.B
        
        # Allocate arrays.
        x = np.zeros((T+1, nx, 1))  # True state trajectory.
        y = np.zeros((T+1, ny, 1))  # Measurement trajectory.
        x_est = np.zeros((T+1, nx, 1))  # DRKF state estimates.
        mse = np.zeros(T+1)
        
        # --- Initialization ---
        x[0] = self.sample_initial_state()
        x_est[0] = self.nominal_x0_mean.copy()
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        # Initial update using DRKF update:
        x_est[0] = self.DR_kalman_filter(self.nominal_mu_v, self.nominal_x0_mean, y[0])
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2
        
        # --- Time Update and Filtering with Control ---
        for t in range(T):
            # Compute control input: u[t] = -K_lqr * x_est[t]
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            u = -self.K_lqr @ x_est[t]
            
            # True state propagation: x[t+1] = A*x[t] + B*u + w
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w
            
            # Measurement:
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v
            
            # Prediction: include control input and nominal process noise mean.
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            # DRKF update:
            x_est[t+1] = self.DR_kalman_filter(self.nominal_mu_v, x_pred, y[t+1])
            
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
        
        comp_time = time.time() - start_time
        return {'comp_time': comp_time,
                'state_traj': x,
                'output_traj': y,
                'est_state_traj': x_est,
                'mse': mse}
    # --- Forward Trajectory Tracking Simulation with LQR Control ---
    def forward_track(self, desired_trajectory):
        """
        Performs closed-loop simulation for trajectory tracking using the DRKF.
        The control input is computed based on the tracking error:
            u[t] = -K_lqr * (x_est[t] - x_d[t]),
        where x_d[t] is the desired state at time t.
        The filter update uses the DRKF update step.
        """
        start_time = time.time()
        T = self.T
        nx = self.nx
        ny = self.ny
        A = self.A
        C = self.C
        B = self.B

        # Allocate arrays.
        x = np.zeros((T+1, nx, 1))         # true state trajectory
        y = np.zeros((T+1, ny, 1))           # measurement trajectory
        x_est = np.zeros((T+1, nx, 1))       # DRKF state estimates
        error = np.zeros((T+1, nx, 1))       # tracking error
        mse = np.zeros(T+1)                # mean squared error

        # --- Initialization ---
        x[0] = self.sample_initial_state()
        x_est[0] = self.nominal_x0_mean.copy()
        # Compute initial tracking error using the desired state at t=0.
        desired = desired_trajectory[:, 0].reshape(-1, 1)
        error[0] = x_est[0] - desired

        # First measurement.
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        # Initial DRKF update.
        x_est[0] = self.DR_kalman_filter(self.nominal_mu_v, self.nominal_x0_mean, y[0])
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2

        # --- Time Update and Filtering with Trajectory Tracking Control ---
        for t in range(T):
            # Get desired state at current time step.
            desired = desired_trajectory[:, t].reshape(-1, 1)
            # Compute tracking error.
            error[t] = x_est[t] - desired
            # Compute control input using LQR gain.
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            u = -self.K_lqr @ error[t]

            # True state propagation: x[t+1] = A*x[t] + B*u + w
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w

            # Measurement: y[t+1] = C*x[t+1] + v
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v

            # Prediction: x_pred = A*x_est[t] + B*u + nominal_mu_w
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            # DRKF update:
            x_est[t+1] = self.DR_kalman_filter(self.nominal_mu_v, x_pred, y[t+1])
            
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2

        comp_time = time.time() - start_time
        return {'comp_time': comp_time,
                'state_traj': x,
                'output_traj': y,
                'est_state_traj': x_est,
                'mse': mse,
                'tracking_error': error}
