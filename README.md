On The Steady-State Distributionally Robust Kalman Filter
====================================================

This repository contains code to run experiments on a steady–state, distributionally robust Kalman filter. 
The experiments compare the performance of different filtering methods under various uncertainty distributions.

## Requirements
- Python (>= 3.5)
- numpy (>= 1.17.4)
- scipy (>= 1.6.2)
- matplotlib (>= 3.1.2)
- control (>= 0.9.4)
- **[CVXPY](https://www.cvxpy.org/)**
- **[MOSEK (>= 9.3)](https://www.mosek.com/)** (required by CVXPY for solving optimization problems)
- (pickle5) install if you encounter compatibility issues with pickle
- joblib (>=1.4.2, Used for parallel computation)

### main.py

Runs closed–loop simulations applying an LQR controller with various filters. The experiments consider both Gaussian and U–Quadratic uncertainty distributions. After the experiments, results (including optimal robust parameter selection) are saved.

- For Gaussian uncertainties

```
python main.py --dist normal
```

- For U-Quadratic uncertainties

```
python main.py --dist quadratic
```

### convergence_check.py

```
python convergence_check.py
```