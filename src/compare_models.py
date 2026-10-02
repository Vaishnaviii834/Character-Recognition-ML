import pandas as pd
import matplotlib.pyplot as plt


results = {
    "Model": ["KNN", "SVM", "Neural Network"],
    "Accuracy": [0.9455, 0.9451, 0.9760],
"Precision": [0.9458, 0.9455, 0.9760],
"Recall": [0.9455, 0.9451, 0.9760],
"F1-Score": [0.9454, 0.9451, 0.9760]
}

df = pd.DataFrame(results)

print("\nModel Comparison")
print("================")
print(df.to_string(index=False))

# Plot comparison
df.set_index("Model").plot(kind="bar", figsize=(10, 6))

plt.title("Character Recognition Model Comparison")
plt.ylabel("Score")
plt.ylim(0.90, 1.00)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("results/model_comparison.png")
plt.show()