"""
Run All ML Models
==================
Executes all 6 model scripts in dependency order and reports results.

Usage:
    python notebooks/run_all_models.py
"""
import subprocess
import sys
import os
import time

BASE = os.path.dirname(__file__)

SCRIPTS = [
    ("K-Means Developer Clustering",    "developer_clustering.py"),
    ("Isolation Forest Burnout",         "burnout_isolation_forest.py"),
    ("XGBoost + SHAP Explainability",   "xgboost_shap.py"),
    ("Transformer Multi-Step Forecast", "transformer_forecast.py"),
    ("VAE Anomaly Detection",           "anomaly_detection_vae.py"),
    ("RL Bandit Sprint Optimizer",      "bandit_sprint_optimizer.py"),
]

results = []
print("=" * 60)
print("DEVINSIGHT — Running All ML Models")
print("=" * 60)

for title, script in SCRIPTS:
    path = os.path.join(BASE, script)
    print(f"\n▶  {title}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=300
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            print(f"   ✅  Done in {elapsed:.1f}s")
            results.append((title, "✅ OK", f"{elapsed:.1f}s"))
        else:
            print(f"   ❌  Failed (rc={proc.returncode})")
            print(proc.stderr[-800:] if proc.stderr else "")
            results.append((title, "❌ FAIL", proc.stderr[-120:] if proc.stderr else ""))
    except subprocess.TimeoutExpired:
        results.append((title, "⏱ TIMEOUT", ">300s"))
        print("   ⏱  Timeout after 300s")
    except Exception as e:
        results.append((title, "❌ ERROR", str(e)))
        print(f"   ❌  Error: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, status, detail in results:
    print(f"  {status}  {name:<42} {detail}")
print()
