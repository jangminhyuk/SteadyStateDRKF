#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KF.py implements a standard Kalman filter for state estimation in a closed–loop LQR experiment.
"""

import numpy as np
import time

class KF:
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_M,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_M,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None):
        """
        Parameters:
         T: time horizon.
         dist: process noise distribution ('normal' or 'quadratic').
         noise_dist: measurement noise distribution ('normal' or 'quadratic').
         system_data: tuple (A, C).
         B: control input matrix.
         
         True parameters (for simulation):
             - true_x0_mean, true_x0_cov: initial state distribution.
             - true_mu_w, true_Sigma_w: process noise.
             - true_mu_v, true_M: measurement noise.
         Nominal parameters (for filtering):
             - Use known mean vectors (nominal_x0_mean, nominal_mu_w, nominal_mu_v) and
               EM–estimated covariances (nominal_x0_cov, nominal_Sigma_w, nominal_M).
         Bounds for quadratic/uniform distributions if used.
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
        
        # Nominal parameters.
        self.nominal_x0_mean = nominal_x0_mean
        self.nominal_x0_cov = nominal_x0_cov
        self.nominal_mu_w = nominal_mu_w
        self.nominal_Sigma_w = nominal_Sigma_w
        self.nominal_mu_v = nominal_mu_v
        self.nominal_M = nominal_M
        
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

    # --- Sampling Functions for True Noise ---
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
                alpha = 12.0 / ((b[j] - a[j]) ** 3)
                tmp = 3 * x[i, j] / alpha - (beta - a[j]) ** 3
                if tmp >= 0:
                    x[i, j] = beta + tmp ** (1.0/3.0)
                else:
                    x[i, j] = beta - (-tmp) ** (1.0/3.0)
        return x

    def quadratic(self, max_val, min_val, N=1):
        n = min_val.shape[0]
        x = np.random.rand(n, N)
        x = self.quad_inverse(x, max_val, min_val)
        return x

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
        x = np.zeros((T+1, nx, 1))         # true state trajectory
        y = np.zeros((T+1, ny, 1))           # measurement trajectory
        x_est = np.zeros((T+1, nx, 1))       # filter state estimates
        P = np.zeros((T+1, nx, nx))          # error covariance
        
        # --- Initialization ---
        x[0] = self.sample_initial_state()
        x_est[0] = self.nominal_x0_mean.copy()
        P[0] = self.nominal_x0_cov.copy()
        
        # First measurement.
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        S0 = C @ P[0] @ C.T + self.nominal_M
        K0 = P[0] @ C.T @ np.linalg.inv(S0)
        innovation0 = y[0] - (C @ x_est[0] + self.nominal_mu_v)
        x_est[0] = x_est[0] + K0 @ innovation0
        P[0] = (np.eye(nx) - K0 @ C) @ P[0]
        
        mse = np.zeros(T+1)
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2
        
        # --- Time Update and Filtering with Control ---
        for t in range(T):
            # Compute control input using LQR gain.
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            u = -self.K_lqr @ x_est[t]
            
            # True state propagation: x[t+1] = A*x[t] + B*u + w
            w = self.sample_process_noise()
            x[t+1] = A @ x[t] + B @ u + w
            
            # Measurement:
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v
            
            # Filter prediction: x_pred = A*x_est[t] + B*u + nominal_mu_w
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            P_pred = A @ P[t] @ A.T + self.nominal_Sigma_w
            
            # Update:
            S_t = C @ P_pred @ C.T + self.nominal_M
            K_t = P_pred @ C.T @ np.linalg.inv(S_t)
            innovation = y[t+1] - (C @ x_pred + self.nominal_mu_v)
            x_est[t+1] = x_pred + K_t @ innovation
            P[t+1] = (np.eye(nx) - K_t @ C) @ P_pred
            
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
        
        comp_time = time.time() - start_time
        return {'comp_time': comp_time,
                'state_traj': x,
                'output_traj': y,
                'est_state_traj': x_est,
                'mse': mse}
        
# --- Forward trajectory tracking Simulation with LQR Control ---
    def forward_track(self, desired_trajectory):
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
        x_est = np.zeros((T+1, nx, 1))       # filter state estimates
        P = np.zeros((T+1, nx, nx))          # error covariance
        error = np.zeros((self.T+1, self.nx, 1)) # Tracking error
        
        # --- Initialization ---
        x[0] = self.sample_initial_state()
        x_est[0] = self.nominal_x0_mean.copy()
        P[0] = self.nominal_x0_cov.copy()
        
        # First measurement.
        v0 = self.sample_measurement_noise()
        y[0] = C @ x[0] + v0
        S0 = C @ P[0] @ C.T + self.nominal_M
        K0 = P[0] @ C.T @ np.linalg.inv(S0)
        innovation0 = y[0] - (C @ x_est[0] + self.nominal_mu_v)
        x_est[0] = x_est[0] + K0 @ innovation0
        P[0] = (np.eye(nx) - K0 @ C) @ P[0]
        
        mse = np.zeros(T+1)
        mse[0] = np.linalg.norm(x_est[0] - x[0])**2
        
        # --- Time Update and Filtering with Control ---
        for t in range(T):
            # Compute control input using LQR gain.
            if self.K_lqr is None:
                raise ValueError("LQR gain (K_lqr) has not been assigned!")
            
            
            # True state propagation: x[t+1] = A*x[t] + B*u + w
            w = self.sample_process_noise()
            
            # Get desired trajectory at current time step (desired position and velocity)
            traj = desired_trajectory[:,t].reshape(-1, 1)
            error[t] = x_est[t] - traj  # Error as a 4D vector
            u = -self.K_lqr @ error[t]
            
            
            x[t+1] = A @ x[t] + B @ u + w
            
            # Measurement:
            v = self.sample_measurement_noise()
            y[t+1] = C @ x[t+1] + v
            
            # Filter prediction: x_pred = A*x_est[t] + B*u + nominal_mu_w
            x_pred = A @ x_est[t] + B @ u + self.nominal_mu_w
            P_pred = A @ P[t] @ A.T + self.nominal_Sigma_w
            
            # Update:
            S_t = C @ P_pred @ C.T + self.nominal_M
            K_t = P_pred @ C.T @ np.linalg.inv(S_t)
            innovation = y[t+1] - (C @ x_pred + self.nominal_mu_v)
            x_est[t+1] = x_pred + K_t @ innovation
            P[t+1] = (np.eye(nx) - K_t @ C) @ P_pred
            
            mse[t+1] = np.linalg.norm(x_est[t+1] - x[t+1])**2
        
        comp_time = time.time() - start_time
        return {'comp_time': comp_time,
                'state_traj': x,
                'output_traj': y,
                'est_state_traj': x_est,
                'mse': mse}