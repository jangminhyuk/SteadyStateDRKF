#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
risk_sensitive.py

This module implements a risk–sensitive filter for state estimation
in a closed-loop LQR control experiment.
"""

import numpy as np
import time

class RiskSensitive:
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_M,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_M,
                 theta_rs,   # risk sensitivity parameter
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None):
        """
        Parameters:
          T: Time horizon.
          dist, noise_dist: 'normal' or 'quadratic'
          system_data: Tuple (A, C)
          B: Control input matrix.
          true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_M:
              True parameters for simulation.
          nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_M:
              Nominal parameters used for filtering. In this implementation the known (true) mean vectors are used,
              while the covariances come from EM.
          theta_rs: Risk sensitivity parameter.
          x0_max, x0_min, etc.: Bounds for non–normal distributions.
        """
        self.T = T
        self.dist = dist
        self.noise_dist = noise_dist
        self.A, self.C = system_data
        self.B = B
        
        # True parameters.
        self.true_x0_mean = true_x0_mean
        self.true_x0_cov = true_x0_cov
        self.true_mu_w = true_mu_w
        self.true_Sigma_w = true_Sigma_w
        self.true_mu_v = true_mu_v
        self.true_M = true_M
        
        # Nominal parameters for filtering.
        self.nominal_x0_mean = nominal_x0_mean    # known initial state mean
        self.nominal_mu_w = nominal_mu_w          # known process noise mean
        self.nominal_mu_v = nominal_mu_v          # known measurement noise mean
        self.nominal_x0_cov = nominal_x0_cov       # EM estimate for initial covariance
        self.nominal_Sigma_w = nominal_Sigma_w     # EM estimate for process noise covariance
        self.nominal_M = nominal_M                 # EM estimate for measurement noise covariance
        
        self.theta_rs = theta_rs
        
        if self.dist in ["uniform", "quadratic"]:
            self.x0_max = x0_max
            self.x0_min = x0_min
            self.w_max = w_max
            self.w_min = w_min
        if self.noise_dist in ["uniform", "quadratic"]:
            self.v_max = v_max
            self.v_min = v_min
            
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        
        # LQR gain will be assigned externally.
        self.K_lqr = None

    # --- Sampling Functions ---
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
                    x[i, j] = beta + tmp ** (1.0/3.0)
                else:
                    x[i, j] = beta - (-tmp) ** (1.0/3.0)
        return x

    def quadratic(self, max_val, min_val, N=1):
        n = min_val.shape[0]
        x = np.random.rand(N, n).T  # Generates an array of shape (n, N)
        return self.quad_inverse(x, max_val, min_val)


    def sample_initial_state(self):
        if self.dist == "normal":
            return self.normal(self.true_x0_mean, self.true_x0_cov, N=1)
        elif self.dist == "quadratic":
            return self.quadratic(self.x0_max, self.x0_min, N=1)
        else:
            raise ValueError("Unsupported distribution for initial state.")

    def sample_process_noise(self):
        if self.dist == "normal":
            return self.normal(self.true_mu_w, self.true_Sigma_w, N=1)
        elif self.dist == "quadratic":
            return self.quadratic(self.w_max, self.w_min, N=1)
        else:
            raise ValueError("Unsupported distribution for process noise.")

    def sample_measurement_noise(self):
        if self.noise_dist == "normal":
            return self.normal(self.true_mu_v, self.true_M, N=1)
        elif self.noise_dist == "quadratic":
            return self.quadratic(self.v_max, self.v_min, N=1)
        else:
            raise ValueError("Unsupported distribution for measurement noise.")

    # --- Forward Simulation Using the Risk-Sensitive Update with LQR ---
    def forward(self):
        start_time = time.time()
        T = self.T
        nx = self.nx
        ny = self.ny
        A = self.A
        C = self.C
        B = self.B
        Q = self.nominal_Sigma_w  # process noise covariance (EM estimate)
        R = self.nominal_M        # measurement noise covariance (EM estimate)
        theta_rs = self.theta_rs
        
        # Allocate arrays.
        x = np.zeros((T+1, nx, 1))  # true state trajectory
        y = np.zeros((T+1, ny, 1))  # measurement trajectory
        x_est = np.zeros((T+1, nx, 1))  # filter state estimates
        mse = np.zeros(T+1)
        
        # --- Initialization ---
        x[0] = self.sample_initial_state()
        x_est[0] = self.nominal_x0_mean.copy()
        Sigma = self.nominal_x0_cov.copy()
        
        # First measurement.
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2
        
        # Closed-loop simulation from t=0 to T-1.
        for t in range(0, T):
            # Compute control input: u[t] = -K_lqr x_est[t]
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            u = -self.K_lqr @ x_est[t]
            
            # True state propagation: x[t+1] = A x[t] + B u + w.
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w
            
            # Measurement: y[t+1] = C x[t+1] + v.
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v
            
            # Prediction step: include control input and add nominal process noise mean.
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            y_pred = C @ x_pred + self.nominal_mu_v
            
            # Risk-sensitive innovation covariance.
            S = C @ Sigma @ C.T + R + theta_rs * (C @ Sigma @ C.T)
            # Compute Kalman gain.
            K_gain = A @ Sigma @ C.T @ np.linalg.inv(S)
            # Update estimate.
            innovation = y[t+1] - y_pred
            x_est[t+1] = x_pred + K_gain @ innovation
            # Update covariance.
            Sigma = A @ Sigma @ A.T + Q - A @ Sigma @ C.T @ np.linalg.inv(S) @ C @ Sigma @ A.T
            
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
        Performs closed-loop simulation for trajectory tracking using the risk–sensitive filter.
        The control input is computed based on the tracking error:
            u[t] = -K_lqr * (x_est[t] - x_d[t]),
        where x_d[t] is the desired state at time t.
        The filter update incorporates the risk–sensitive modification.
        """
        start_time = time.time()
        T = self.T
        nx = self.nx
        ny = self.ny
        A = self.A
        C = self.C
        B = self.B
        Q = self.nominal_Sigma_w  # EM estimate for process noise covariance.
        R = self.nominal_M        # EM estimate for measurement noise covariance.
        theta_rs = self.theta_rs
        
        # Allocate arrays.
        x = np.zeros((T+1, nx, 1))           # True state trajectory.
        y = np.zeros((T+1, ny, 1))           # Measurement trajectory.
        x_est = np.zeros((T+1, nx, 1))       # Filter (risk-sensitive) state estimates.
        mse = np.zeros(T+1)
        tracking_error = np.zeros((T+1, nx, 1))
        
        # --- Initialization ---
        x[0] = self.sample_initial_state()
        x_est[0] = self.nominal_x0_mean.copy()
        Sigma = self.nominal_x0_cov.copy()
        
        # Compute initial tracking error using the desired state at t = 0.
        desired = desired_trajectory[:, 0].reshape(-1, 1)
        tracking_error[0] = x_est[0] - desired
        
        # First measurement.
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2
        
        # --- Time Update and Filtering with Trajectory Tracking ---
        for t in range(T):
            # Get desired state at current time step.
            desired = desired_trajectory[:, t].reshape(-1, 1)
            tracking_error[t] = x_est[t] - desired
            
            # Compute control input based on tracking error.
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            u = -self.K_lqr @ tracking_error[t]
            
            # True state propagation: x[t+1] = A*x[t] + B*u + w.
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w
            
            # Measurement: y[t+1] = C*x[t+1] + v.
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v
            
            # Prediction step.
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            y_pred = C @ x_pred + self.nominal_mu_v
            
            # Risk-sensitive innovation covariance.
            S = C @ Sigma @ C.T + R + theta_rs * (C @ Sigma @ C.T)
            # Compute gain.
            K_gain = A @ Sigma @ C.T @ np.linalg.inv(S)
            # Innovation.
            innovation = y[t+1] - y_pred
            # Update estimate.
            x_est[t+1] = x_pred + K_gain @ innovation
            # Update covariance.
            Sigma = A @ Sigma @ A.T + Q - A @ Sigma @ C.T @ np.linalg.inv(S) @ C @ Sigma @ A.T
            
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
        
        comp_time = time.time() - start_time
        return {'comp_time': comp_time,
                'state_traj': x,
                'output_traj': y,
                'est_state_traj': x_est,
                'mse': mse,
                'tracking_error': tracking_error}
