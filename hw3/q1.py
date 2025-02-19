import numpy as np
import scipy.stats as stats
import pandas as pd

def initialize_parameters():
    # Expected values and coefficients of variation
    E_z = np.array([400, 300, 10, 1.4, 20])
    delta = np.array([0.15, 0.30, 0.20, 0.10, 0.20])
    
    # Compute lambda values and standard deviations
    lambda_values = np.log(E_z / np.sqrt(1 + delta**2))
    sigma_x = np.sqrt(np.log(1 + delta**2))
    
    # Correlation matrix
    R = np.array([
        [1,  0.4,  0,  0,  0.4],
        [0.4, 1,  0,  0,  -0.2],
        [0,  0,  1,  0.3, 0.3],
        [0,  0,  0.3,  1,  -0.4],
        [0.4, -0.2,  0.3,  -0.4,  1]
    ])
    
    # Compute covariance matrix and its Cholesky decomposition
    Z = R * np.outer(sigma_x, sigma_x)
    L = np.linalg.cholesky(Z)
    
    return lambda_values.reshape(-1,1), L

def update_form_iteration(L, lambda_vec, u_k):
    """
    Performs one iteration of the First Order Reliability Method (FORM)
    
    Parameters:
    L: Lower triangular matrix from Cholesky decomposition
    lambda_vec: Vector of transformed means
    u_k: Current standard normal vector
    
    Returns:
    u_k_next: Next iteration vector
    gu_grad: Gradient vector
    gu_u: Limit state function value
    """
    # Calculate x_k and its exponential values
    x_k = np.dot(L, u_k) + lambda_vec
    z_k = np.exp(x_k)
    Fa, Fb, q, alpha, c = z_k.flatten()
    
    # Calculate gz_gradient
    gz_grad = np.array([
        [-alpha/(Fa + Fb)],
        [-alpha/(Fa + Fb)],
        [-1/q],
        [-np.log((Fa + Fb)/900)],
        [1/c]
    ])
    
    # Calculate limit state function
    gu_u = x_k[4] - x_k[2] - np.exp(x_k[3]) * np.log((np.exp(x_k[0]) + np.exp(x_k[1]))/900)
    
    # Calculate gradient
    J_zx = np.diag(z_k.flatten())
    gx_grad = np.dot(J_zx, gz_grad)
    gu_grad = np.dot(L.T, gx_grad)
    
    # Calculate next iteration vector
    numerator = (np.dot(gu_grad.T, u_k) - gu_u) * gu_grad
    denominator = np.dot(gu_grad.T, gu_grad)
    u_k_next = numerator / denominator
    
    return u_k_next, gu_grad, gu_u


def run_form_analysis(num_iterations=6):  
    # Initialize parameters
    lambda_vec, L = initialize_parameters()
    u_k = np.zeros((5, 1))
    
    # Initialize lists for each parameter
    u_values = []
    gradient_values = []
    g_values = []
    beta_values = []
    pf_values = []
    
    # Add initial values (u0)
    u_values.append(u_k.flatten())
    x_k = np.dot(L, u_k) + lambda_vec
    z_k = np.exp(x_k)
    Fa, Fb, q, alpha, c = z_k.flatten()
    
    # Calculate initial gradient
    gz_grad = np.array([
        [-alpha/(Fa + Fb)],
        [-alpha/(Fa + Fb)],
        [-1/q],
        [-np.log((Fa + Fb)/900)],
        [1/c]
    ])
    J_zx = np.diag(z_k.flatten())
    gx_grad = np.dot(J_zx, gz_grad)
    gu_grad = np.dot(L.T, gx_grad)
    gradient_values.append(gu_grad.flatten())
    
    # Calculate initial g(z), beta, and Pf
    gu_u = x_k[4] - x_k[2] - np.exp(x_k[3]) * np.log((np.exp(x_k[0]) + np.exp(x_k[1]))/900)
    g_values.append(gu_u.item())
    beta = np.linalg.norm(u_k.flatten())
    beta_values.append(beta)
    pf = stats.norm.cdf(-beta)
    pf_values.append(pf)
    
    # Perform iterations
    for i in range(num_iterations-1):
        u_k_next, gu_grad, gu_u = update_form_iteration(L, lambda_vec, u_k)
        u_k = u_k_next
        
        # Store values
        u_values.append(u_k.flatten())
        gradient_values.append(gu_grad.flatten())
        g_values.append(gu_u.item())
        beta = np.linalg.norm(u_k.flatten())
        beta_values.append(beta)
        pf = stats.norm.cdf(-beta)
        pf_values.append(pf)
    
    # Create DataFrame
    columns = [f'u{i}' for i in range(num_iterations)]
    df_results = pd.DataFrame({
        'Parameter': ['ui', '∇g(zi)', 'g(zi)', 'βi', 'Pf_i']
    })
    
    # Add data for each iteration
    for i in range(num_iterations):
        df_results[f'u{i}'] = [
            np.round(u_values[i], 4),
            np.round(gradient_values[i], 4),
            np.round(g_values[i], 4),
            np.round(beta_values[i], 4),
            f'{pf_values[i]:.2e}'
        ]
    
    return df_results

def compute_design_point(L, lambda_vec, u_star):
    """
    Compute the design point in the physical space.
    
    Parameters:
    L: Lower triangular matrix from Cholesky decomposition
    lambda_vec: Vector of transformed means
    u_star: Optimal design point in standard normal space
    
    Returns:
    z_star: Design point in the physical space
    """
    # Compute x* in physical space
    x_star = np.dot(L, u_star) + lambda_vec
    
    # Convert to original space
    z_star = np.exp(x_star)
    
    return z_star

def compute_importance_measures(L, grad_u_star):
    """
    Computes the importance measure in both standard normal space (x) and physical space (z).

    Parameters:
    - L: Cholesky decomposition matrix (transforms from standard normal space to physical space)
    - grad_u_star: Gradient of the limit state function at the design point u* (∇g(u*))

    Returns:
    - gamma_x: Importance measure in x-space (normalized)
    - gamma_z: Importance measure in z-space (normalized)
    """
    # Compute alpha (importance measure in standard normal space)
    alpha = grad_u_star / np.linalg.norm(grad_u_star)

    # Compute importance measure in x-space
    S_x = np.diag(np.std(L, axis=1))  # Approximate standard deviation in x-space
    eta_x = S_x @ np.linalg.inv(L.T) @ alpha
    gamma_x = eta_x / np.linalg.norm(eta_x)  # Normalize

    # Compute importance measure in z-space
    S_z = np.diag(np.std(np.exp(L), axis=1))  # Approximate standard deviation in z-space
    J_xu = L.T  # Transformation from standard normal space
    J_uz = np.linalg.inv(J_xu)  # Inverse transformation
    eta_z = S_z @ J_uz.T @ alpha
    gamma_z = eta_z / np.linalg.norm(eta_z)  # Normalize

    return gamma_x, gamma_z

if __name__ == "__main__":
    results_df = run_form_analysis()
    results_df.to_csv('./results/form_analysis_results.csv', index=False)

    # Run FORM analysis to get u*
    results_df = run_form_analysis()
    u_star = np.array(results_df.iloc[0, -1]).reshape(-1,1)  # Extract last iteration result

    # Compute the design point in physical space
    lambda_vec, L = initialize_parameters()
    z_star = compute_design_point(L, lambda_vec, u_star)

    # Print the results
    print("Design Point z* in Physical Space:")
    print(np.round(z_star, 3))

    # (i)
    grad_u_star = [-0.1266, -0.3182, -0.146 ,-0.1253,0.1141]
    gamma_x, gamma_z = compute_importance_measures(L, grad_u_star)

    print("Importance Measure in x-space:", np.round(gamma_x, 4))
    print("Importance Measure in z-space:", np.round(gamma_z, 4))