import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/ultimate-ufc/ufc-master.csv")

print(f"Shape: {df.shape}")
print(f"\nMissing values (cols with > 0):")
missing = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
summary = pd.DataFrame({"count": missing, "pct": missing_pct})
print(summary[summary["count"] > 0].sort_values("count", ascending=False))

# Individual plots for better readability
msno.matrix(df, figsize=(16, 8), fontsize=8, sparkline=True)
plt.tight_layout()
plt.savefig("plots/missing/missing_matrix.png", dpi=150)
plt.close()

msno.bar(df, figsize=(16, 8), fontsize=8)
plt.tight_layout()
plt.savefig("plots/missing/missing_bar.png", dpi=150)
plt.close()

msno.heatmap(df, figsize=(14, 10), fontsize=8)
plt.tight_layout()
plt.savefig("plots/missing/missing_heatmap.png", dpi=150)
plt.close()

msno.dendrogram(df, figsize=(14, 8), fontsize=8)
plt.tight_layout()
plt.savefig("plots/missing/missing_dendrogram.png", dpi=150)
plt.close()

print("\nSaved: plots/missing/missing_matrix.png, plots/missing/missing_bar.png, plots/missing/missing_heatmap.png, plots/missing/missing_dendrogram.png")
