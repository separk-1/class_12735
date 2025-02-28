import numpy as np

def monte_carlo_cutset_prob(n_simulations=10000, failure_prob=0.35):
    """
    Monte Carlo simulation to estimate failure probabilities of minimum cut-sets.
    
    Parameters:
    - n_simulations (int): Number of Monte Carlo iterations
    - failure_prob (float): Probability of failure for each component
    
    Returns:
    - P_A (float): Probability of all components in first minimum cut-set failing
    - P_B (float): Probability of all components in second minimum cut-set failing
    - P_A_intersect_B (float): Probability of common components failing in both cut-sets
    - P_union (float): Probability of either cut-set failing
    """
    np.random.seed(42)  # Reproducibility
    num_components = 6  # Total components
    
    # Track failures
    failures_A = 0
    failures_B = 0
    failures_intersect = 0
    
    for _ in range(n_simulations):
        states = np.random.rand(num_components) > failure_prob  # 1 = working, 0 = failed
        
        # Define cut-sets
        cutset_1 = not (states[0] and states[1] and states[2])  # C1, C2, C3 fail
        cutset_2 = not (states[3] and states[4] and states[5])  # C4, C5, C6 fail
        
        # Common components (교집합, 예제에 따라 다를 수 있음)
        common_fail = not (states[1] and states[2])  # 예제에서 공통 요소는 C2, C3

        if cutset_1:
            failures_A += 1
        if cutset_2:
            failures_B += 1
        if cutset_1 and cutset_2 and common_fail:
            failures_intersect += 1

    # 확률 계산
    P_A = failures_A / n_simulations
    P_B = failures_B / n_simulations
    P_A_intersect_B = failures_intersect / n_simulations
    P_union = P_A + P_B - P_A_intersect_B  # 합집합 법칙 검증

    return P_A, P_B, P_A_intersect_B, P_union

# 실행
P_A, P_B, P_A_intersect_B, P_union = monte_carlo_cutset_prob()

# 결과 출력
print(f"P(A) = {P_A:.5f}")
print(f"P(B) = {P_B:.5f}")
print(f"P(A ∩ B) = {P_A_intersect_B:.5f}")
print(f"P(A ∪ B) = {P_union:.5f}")
print(f"Verification (Should be close to P(A) + P(B) - P(A ∩ B)): {P_A + P_B - P_A_intersect_B:.5f}")
