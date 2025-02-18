import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# === CSV 파일 로드 ===
df_mc = pd.read_csv("monte_carlo_results.csv")
df_is = pd.read_csv("importance_sampling_results.csv")
df_is_s = pd.read_csv("importance_sampling_s_values.csv")

# === 그래프 설정 ===
plt.figure(figsize=(12, 5))

# === (1) Monte Carlo: 샘플 개수 vs. 실패 확률 ===
plt.subplot(1, 2, 1)
plt.plot(df_mc["Samples"], df_mc["Failure Probability"], marker='o', linestyle='-', label="MC Simulation")
plt.xscale("log")  # 샘플 개수가 크기 때문에 로그 스케일 적용
plt.xlabel("Number of Samples (log scale)")
plt.ylabel("Failure Probability")
plt.title("Monte Carlo Simulation: Failure Probability")
plt.grid(True)
plt.legend()

# === (2) Monte Carlo: 샘플 개수 vs. 신뢰도 지수 ===
plt.subplot(1, 2, 2)
plt.plot(df_mc["Samples"], df_mc["Reliability Index"], marker='s', linestyle='-', label="MC Simulation")
plt.xscale("log")
plt.xlabel("Number of Samples (log scale)")
plt.ylabel("Reliability Index (Beta)")
plt.title("Monte Carlo Simulation: Reliability Index")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# === Importance Sampling 결과 그래프 ===
plt.figure(figsize=(12, 5))

# === (3) Importance Sampling: 샘플 개수 vs. 실패 확률 ===
plt.subplot(1, 2, 1)
plt.plot(df_is["Samples"], df_is["Failure Probability"], marker='o', linestyle='-', label="Importance Sampling")
plt.xscale("log")
plt.xlabel("Number of Samples (log scale)")
plt.ylabel("Failure Probability")
plt.title("Importance Sampling: Failure Probability")
plt.grid(True)
plt.legend()

# === (4) Importance Sampling: 샘플 개수 vs. 신뢰도 지수 ===
plt.subplot(1, 2, 2)
plt.plot(df_is["Samples"], df_is["Reliability Index"], marker='s', linestyle='-', label="Importance Sampling")
plt.xscale("log")
plt.xlabel("Number of Samples (log scale)")
plt.ylabel("Reliability Index (Beta)")
plt.title("Importance Sampling: Reliability Index")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# === Importance Sampling의 s 값 변화 그래프 ===
plt.figure(figsize=(6, 5))
plt.plot(df_is_s["s Value"], df_is_s["Failure Probability"], marker='o', linestyle='-', label="Failure Probability")
plt.xscale("log")
plt.xlabel("s Value (log scale)")
plt.ylabel("Failure Probability")
plt.title("Effect of s on Importance Sampling")
plt.grid(True)
plt.legend()

plt.show()
