import numpy as np

def lognormal_params_from_median_cov(median, cov):
    """
    로그정규 분포의 중앙값(median)과 변동계수(COV)가 주어졌을 때,
    내부 정규분포의 mu, sigma를 근사적으로 구한다.
    """
    mu = np.log(median)
    
    # 이진 탐색으로 sigma^2를 찾는 함수
    def objective(sigma_sq):
        # 평균 = exp(mu + sigma^2/2)
        mean_approx = np.exp(mu + sigma_sq/2)
        # 분산 = (exp(sigma^2)-1)*exp(2*mu + sigma^2)
        var_approx  = (np.exp(sigma_sq)-1)*np.exp(2*mu + sigma_sq)
        cov_approx  = np.sqrt(var_approx)/mean_approx
        return cov_approx - cov
    
    left, right = 1e-9, 5.0
    for _ in range(100):
        mid = 0.5*(left + right)
        val = objective(mid)
        if abs(val) < 1e-10:
            sigma_sq = mid
            break
        if val > 0:
            right = mid
        else:
            left = mid
    sigma_sq = 0.5*(left + right)
    return mu, np.sqrt(sigma_sq)

def mean_of_lognormal(median, cov):
    """
    중앙값과 COV로부터 로그정규 분포의 평균값을 구한다.
    """
    mu, sigma = lognormal_params_from_median_cov(median, cov)
    # 로그정규분포의 평균 = exp(mu + sigma^2/2)
    return np.exp(mu + 0.5*sigma**2)

def construction_cost(num_roads):
    """
    간단한 예시로, 도로 개수에 따른 건설비용(단위: M$).
    실제 문제의 공식을 적용해도 됨.
    """
    # 예: 기본비용 2M$, 도로 1개당 추가비용 4M$
    return 2.0 + 4.0 * num_roads

def sample_lognormal(median, cov, size, rng):
    """
    (median, cov)를 만족하는 로그정규분포에서 샘플 size개 추출.
    """
    mu, sigma = lognormal_params_from_median_cov(median, cov)
    return rng.lognormal(mean=mu, sigma=sigma, size=size)

def system_failure_probability(num_roads, budget=20.0, n_sims=10_000, seed=42):
    """
    병렬로 도로 num_roads개를 건설했을 때, 
    예산 이내이면 Monte Carlo로 시스템 실패확률 추정, 
    예산 초과면 None 반환.
    
    - 도로 용량: 중앙값=30, COV=0.40
    - 수요(Demand): 중앙값=5, COV=0.50 (예시)
    """
    cost = construction_cost(num_roads)
    if cost > budget:
        return None
    
    rng = np.random.default_rng(seed)
    
    # 용량, 수요 분포 파라미터
    median_capacity = 30.0
    cov_capacity    = 0.40
    
    median_demand = 5.0
    cov_demand    = 0.50  # 문제에서 구체값이 없으니 예시
    
    n_fail = 0
    for _ in range(n_sims):
        # 도로별 capacity 샘플
        caps = sample_lognormal(median_capacity, cov_capacity, num_roads, rng)
        # 도로별 demand 샘플
        demands = sample_lognormal(median_demand, cov_demand, num_roads, rng)
        
        # 병렬 시스템: 하나라도 cap >= demand이면 연결 유지
        # => 모든 도로가 (cap < demand)면 실패
        if np.all(caps < demands):
            n_fail += 1
    
    return n_fail / n_sims

def main():
    budget = 20.0   # M$
    max_roads = 5   # 예: 1~5개 도로
    n_sims = 10000
    
    # 각 도로의 평균 용량 (단일 도로 기준)
    mean_cap_single = mean_of_lognormal(median=30, cov=0.40)
    
    results = []
    for n in range(1, max_roads+1):
        p_fail = system_failure_probability(n, budget=budget, n_sims=n_sims, seed=42)
        cost   = construction_cost(n)
        if p_fail is not None:
            # n개 도로 모두 동일 설계(동일 분포)라고 가정 => 단일 도로 평균 용량은 동일
            # '시스템 전체'의 평균 용량은 단순 합이 아니라 병렬이라 정의가 애매하지만,
            # 문제에서 "mean capacity"를 보고하라고 했으므로
            # 여기서는 "각 도로의 평균 용량"만 표기하거나, n * mean_cap_single 도 가능
            results.append((n, cost, p_fail, mean_cap_single))
    
    # 실패확률 기준 정렬 (낮은 순)
    results.sort(key=lambda x: x[2])
    
    print(" n  |  Cost(M$) | FailureProb | MeanCapacity(per road)")
    for (n, c, pf, mc) in results:
        print(f"{n:2d}  |  {c:8.2f} | {pf:12.6f} | {mc:12.4f}")
    
    if results:
        best_n, best_cost, best_pfail, best_mc = results[0]
        print("\n--- Best design (lowest failure probability) ---")
        print(f"  # of roads    = {best_n}")
        print(f"  Cost          = {best_cost:.2f} M$")
        print(f"  P(failure)    = {best_pfail:.6f}")
        print(f"  Mean capacity = {best_mc:.4f} (per road)")
    else:
        print("No feasible design within the budget.")

if __name__ == "__main__":
    main()
