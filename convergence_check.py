import numpy as np
import cvxpy as cp
import mosek
import control
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

np.random.seed(42)

# -------------------------------------------------------
# Sampling and distribution functions
# -------------------------------------------------------
def uniform(a, b, N=1):
    n = a.shape[0]
    return a[:, None] + (b[:, None] - a[:, None]) * np.random.rand(n, N)

def normal(mu, Sigma, N=1):
    return np.random.multivariate_normal(mu.ravel(), Sigma, size=N).T

def quad_inverse(x, b, a):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            beta = 0.5 * (a[j] + b[j])
            alpha = 12.0 / ((b[j] - a[j]) ** 3)
            tmp = 3 * x[i, j] / alpha - (beta - a[j]) ** 3
            x[i, j] = beta + (tmp if tmp >= 0 else -(-tmp) ** (1. / 3.)) ** (1. / 3.)
    return x

def quadratic(wmax, wmin, N=1):
    x = np.random.rand(N, wmin.shape[0]).T
    return quad_inverse(x, wmax, wmin)

def gen_sample_dist_inf(dist, N_sample, mu=None, Sigma=None, w_min=None, w_max=None):
    if dist == "normal":
        w = normal(mu, Sigma, N=N_sample)
    elif dist == "uniform":
        w = uniform(w_min, w_max, N=N_sample)
    elif dist == "quadratic":
        w = quadratic(wmax=w_max, wmin=w_min, N=N_sample)
    else:
        raise ValueError("Unsupported distribution.")
    mean_ = np.mean(w, axis=1, keepdims=True)
    var_ = np.cov(w)
    return mean_, var_

# -------------------------------------------------------
# Helper function to enforce positive definiteness
# -------------------------------------------------------
def enforce_positive_definiteness(M, epsilon=1e-3):
    M = (M + M.T) / 2.0
    eigvals = np.linalg.eigvalsh(M)
    min_eig = np.min(eigvals)
    if min_eig < epsilon:
        M = M + (epsilon - min_eig) * np.eye(M.shape[0])
    return M

# -------------------------------------------------------
# (Optional) Assumption check function.
# -------------------------------------------------------
def check_assumptions(A, Sigma_w_nom, C, Sigma_v_nom, T):
    n = A.shape[0]
    m = C.shape[0]
    if np.any(np.linalg.eigvals(Sigma_w_nom) <= 0):
        raise ValueError("Sigma_w_nom is not positive definite.")
    if np.any(np.linalg.eigvals(Sigma_v_nom) <= 0):
        raise ValueError("Sigma_v_nom is not positive definite.")
    O = control.obsv(A, C)
    if np.linalg.matrix_rank(O) < n:
        raise ValueError("The pair (A, C) is not observable.")
    try:
        B = np.linalg.cholesky(Sigma_w_nom)
    except np.linalg.LinAlgError:
        raise ValueError("Sigma_w_nom is not positive definite for Cholesky.")
    CC = control.ctrb(A, B)
    if np.linalg.matrix_rank(CC) < n:
        raise ValueError("The pair (A, sqrt(Sigma_w_nom)) is not reachable.")
    O_T_blocks = [C @ np.linalg.matrix_power(A, T-1-i) for i in range(T)]
    O_T = np.vstack(O_T_blocks)
    if np.linalg.matrix_rank(O_T) < n:
        raise ValueError(f"O_T does not have full column rank; (A, C) may not be observable with T={T}")
    print("Assumptions verified.")


# -------------------------------------------------------
# DR Kalman filter measurement update (only theta_x version)
# -------------------------------------------------------
def dr_kf_solve_measurement_update(Sigma_x_minus_hat, A, C, Sigma_v_nom, theta_x, delta=1e-6):
    n = Sigma_x_minus_hat.shape[0]
    m = Sigma_v_nom.shape[0]
    Z_var = cp.Variable((m, m), PSD=True)
    Sigma_x_var = cp.Variable((n, n), PSD=True)
    Sigma_x_minus_var = cp.Variable((n, n), PSD=True)
    Y = cp.Variable((n, n))
    
    obj = cp.Maximize(cp.trace(Sigma_x_var))
    
    constraints = [
        cp.bmat([
            [Sigma_x_minus_var - Sigma_x_var, Sigma_x_minus_var @ C.T],
            [C @ Sigma_x_minus_var, C @ Sigma_x_minus_var @ C.T + Sigma_v_nom]
        ]) >> 0,
        cp.trace(Sigma_x_minus_hat + Sigma_x_minus_var - 2*Y) <= theta_x**2,
        cp.bmat([[Sigma_x_minus_hat, Y],
                 [Y.T, Sigma_x_minus_var]
                ]) >> 0,
    ]
    
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    
    return Sigma_x_var.value

# -------------------------------------------------------
# Matrix computations for phi_T and related quantities
# -------------------------------------------------------
def compute_matrices(T, A, Sigma_w_nom, C, Sigma_v_nom):
    n = A.shape[0]
    m = C.shape[0]
    Sigma_v_nom = enforce_positive_definiteness(Sigma_v_nom, epsilon=1e-3)
    B = np.linalg.cholesky(Sigma_w_nom)
    sqrt_Sigma_v_nom = np.linalg.cholesky(Sigma_v_nom)
    
    # 1. R_T: [B, A·B, A²·B, …, A^(T-1)·B]
    R_T_blocks = [np.linalg.matrix_power(A, i) @ B for i in range(T)]
    R_T = np.hstack(R_T_blocks)
    
    # 2. O_T: Vertical stacking of [C A^(T-1); C A^(T-2); ...; C]
    O_T_blocks = [C @ np.linalg.matrix_power(A, T-1-i) for i in range(T)]
    O_T = np.vstack(O_T_blocks)
    if np.linalg.matrix_rank(O_T) < n:
        raise ValueError(f"O_T does not have full column rank; (A,C) may not be observable with T={T}")
    
    # 3. O_T^R: Vertical stacking of [A^(T-1); A^(T-2); …; I]
    O_T_R_blocks = [np.linalg.matrix_power(A, T-1-i) for i in range(T)]
    O_T_R = np.vstack(O_T_R_blocks)
    
    # 4. D_T: I_T ⊗ sqrt(Sigma_v_nom)
    D_T = np.kron(np.eye(T), sqrt_Sigma_v_nom)
    
    # 5. Build block Hankel matrices L_T and H_T.
    L_blocks = [[(np.linalg.matrix_power(A, j-i-1) @ B) if (j - i >= 1) else np.zeros((n, n))
                 for j in range(T)] for i in range(T)]
    H_blocks = [[(C @ (np.linalg.matrix_power(A, j-i-1) @ B)) if (j - i >= 1) else np.zeros((m, n))
                 for j in range(T)] for i in range(T)]
    L_T = np.block(L_blocks)
    H_T = np.block(H_blocks)
    
    # 6. Compute tilde_phi_T.
    I_inner = np.eye(T * n)
    DDT = D_T @ D_T.T
    inv_DDT = np.linalg.inv(DDT)
    inner_term = I_inner + H_T.T @ inv_DDT @ H_T
    inner_inv = np.linalg.inv(inner_term)
    M = L_T @ inner_inv @ L_T.T
    eigvals = np.linalg.eigvals(M)
    lambda_max_val = np.max(np.real(eigvals))
    tilde_phi_T = 1.0 / lambda_max_val
    
    return {
        "R_T": R_T,
        "O_T": O_T,
        "O_T_R": O_T_R,
        "D_T": D_T,
        "L_T": L_T,
        "H_T": H_T,
        "tilde_phi_T": tilde_phi_T
    }

def find_phi_T(O_T, O_T_R, L_T, H_T, D_T, tilde_phi_T, tol_eig=1e-10, bisection_tol=1e-10, max_iter=1000):
    M = D_T @ D_T.T + H_T @ H_T.T
    M_inv = np.linalg.inv(M)
    J_T = O_T_R - L_T @ H_T.T @ M_inv @ O_T
    Omega_T = O_T.T @ M_inv @ O_T

    eig_vals = np.linalg.eigvals(Omega_T)
    lambda_min = np.min(np.real(eig_vals))
    if lambda_min < 0:
        raise ValueError("Omega_T is not positive definite. Check that all assumptions are met.")

    I_N = np.eye(L_T.shape[1])
    inv_DDT = np.linalg.inv(D_T @ D_T.T)
    inner_term = I_N + H_T.T @ inv_DDT @ H_T
    inner_inv = np.linalg.inv(inner_term)
    I_full = np.eye(L_T.shape[0])

    def lambda_min_Omega(phi):
        S_phi = - (1.0 / phi) * I_full + L_T @ inner_inv @ L_T.T
        try:
            S_phi_inv = np.linalg.inv(S_phi)
        except np.linalg.LinAlgError:
            return -np.inf
        Omega_phi = Omega_T + J_T.T @ S_phi_inv @ J_T
        Omega_phi = (Omega_phi + Omega_phi.T) / 2.0
        eigvals = np.linalg.eigvals(Omega_phi)
        return np.min(np.real(eigvals))

    phi_lower = 0.0
    phi_upper = tilde_phi_T
    iteration = 0
    while (phi_upper - phi_lower) > bisection_tol and iteration < max_iter:
        iteration += 1
        phi_mid = (phi_lower + phi_upper) / 2.0
        f_mid = lambda_min_Omega(phi_mid)
        if f_mid > tol_eig:
            phi_lower = phi_mid
        else:
            phi_upper = phi_mid

    phi_final = phi_lower
    return phi_final

def compute_theta_max(A, C, Sigma_w_nom, Sigma_v_nom, phi_T, q=100, T=None):
    """
    Computes theta_max that guarantees convergence of the DR Kalman filter.
    Iterates the standard Kalman Riccati update for q steps and then computes:
         theta_max = sqrt(trace(P_bar) / (1 - phi_T * lambda_max(P_bar))) - sqrt(trace(P_bar))
    """
    P_bar = Sigma_w_nom.copy()
    for q_ in range(q):
        CPCT = C @ P_bar @ C.T
        S = CPCT + Sigma_v_nom
        K = P_bar @ C.T @ np.linalg.inv(S)
        P_update = P_bar - K @ C @ P_bar
        P_bar = A @ P_update @ A.T + Sigma_w_nom

        eigvals = np.linalg.eigvals(P_bar)
        lambda_max_val = np.max(np.real(eigvals))
        trace_P = np.trace(P_bar)
            
    # If the condition (1 - phi_T * lambda_max(P_bar)) is nonpositive, theta_max cannot be computed.
    if phi_T * lambda_max_val > 1:
        return None
    
    term = np.sqrt(trace_P / (1 - phi_T * lambda_max_val))
    theta_max = term - np.sqrt(trace_P)
    return theta_max

def run_dr_kf_once(n=10, m=10, steps=200, T=20, q=100, tol=1e-4):
    try:
        A = np.array([[0.1, 1], [0, 1.2]])
        Sigma_w_nom = np.eye(2)
        C = np.array([[1, -1]])
        Sigma_v_nom = np.eye(1)
        
        # Compute matrices needed for phi_T.
        matrices = compute_matrices(T, A, Sigma_w_nom, C, Sigma_v_nom)
        tilde_phi_T = matrices["tilde_phi_T"]
        phi_T = find_phi_T(matrices["O_T"], matrices["O_T_R"], matrices["L_T"], matrices["H_T"], matrices["D_T"], tilde_phi_T)
        theta_max = compute_theta_max(A, C, Sigma_w_nom, Sigma_v_nom, phi_T, q)
        
        if theta_max is None or theta_max <= 0 or np.isnan(theta_max):
            return None

        Sigma_x_minus = np.eye(n)
        posterior_list = []
        conv_norms = []
        
        for step in range(steps):
            try:
                Sigma_x_sol = dr_kf_solve_measurement_update(Sigma_x_minus, A, C, Sigma_v_nom, theta_max)
            except cp.error.SolverError:
                return None
            posterior_list.append(Sigma_x_sol)
            if step > 0:
                diff_norm = np.linalg.norm(posterior_list[step] - posterior_list[step-1], 'fro')
                conv_norms.append(diff_norm)
                if diff_norm < tol:
                    print(f"Early stopping at iteration {step} with convergence norm {diff_norm:.4e}")
                    break
            Sigma_x_minus = A @ Sigma_x_sol @ A.T + Sigma_w_nom
            
        print("--------------------------------------------------")
        print("Calculated phi_T:", phi_T)
        print("Calculated theta_max:", theta_max)
        print("--------------------------------------------------")
        
        return {
            "A": A,
            "Sigma_w_nom": Sigma_w_nom,
            "C": C,
            "Sigma_v_nom": Sigma_v_nom,
            "phi_T": phi_T,
            "theta_max": theta_max,
            "posterior_list": posterior_list,
            "conv_norms": conv_norms
        }
    except (cp.error.SolverError, Exception):
        return None

if __name__=="__main__":
    tol = 1e-6  # convergence tolerance for final norm
    num_exp = 1  # number of valid experiments to collect
    success_count = 0
    valid_experiments = []
    
    batch_size = 1  # number of experiments to run in parallel per batch
    
    # Run with longer time horizon: steps increased to 1000
    while len(valid_experiments) < num_exp:
        results = Parallel(n_jobs=1)(
            delayed(run_dr_kf_once)(n=2, m=1, steps=1000, T=8, q=20, tol=tol)
            for _ in range(batch_size)
        )
        for res in results:
            if res is not None:
                valid_experiments.append(res)
                final_norm = res["conv_norms"][-1] if res["conv_norms"] else float('inf')
                if final_norm < tol:
                    success_count += 1
                print(f"\nValid experiment count: {len(valid_experiments)}/{num_exp}")
                if len(valid_experiments) >= num_exp:
                    break

    success_rate = (success_count / num_exp) * 100
    print("\n==================================================")
    print("Computation finished!!")

    # --------------------------
    # Convergence Rate Analysis
    # --------------------------
    res = valid_experiments[0]
    posterior_list = res["posterior_list"]

    converged_cov = posterior_list[-1]

    # Compute the 2-norm error for each time step with respect to the converged covariance
    errors_conv = [np.linalg.norm(S - converged_cov, 2) for S in posterior_list]


    # Plot the convergence error (posterior vs. converged)
    plt.figure(figsize=(10, 6))
    # Plot all but the last point with a line
    plt.semilogy(range(len(errors_conv)-1), errors_conv[:-1],
                marker='o', linestyle='-', linewidth=1, markersize=5, color='black')
    # Plot the last point
    plt.semilogy(len(errors_conv)-1, errors_conv[-1],
                marker='o', linestyle='None', markersize=5, color='black')
    plt.xlabel('Time Step', fontsize=16)
    plt.ylabel(r'$\|\Sigma_{x,t} - \Sigma_{x,\mathrm{converged}}\|_2$', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig('convergence_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


