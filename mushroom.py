import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "uciml/mushroom-classification",
    "mushrooms.csv",
)

# Remove duplicates
df = df.drop_duplicates()

# Remove rows with NaN values
df = df.dropna()

odor = pd.get_dummies(df["odor"]).to_numpy()
gill_color = pd.get_dummies(df["gill-color"]).to_numpy()
spore_color = pd.get_dummies(df["spore-print-color"]).to_numpy()
gill_size = df["gill-size"].map({"b": 1, "n": 0}).to_numpy()

y = df["class"].map({"e": 1, "p": 0}).to_numpy()

# Combine features into a single array
X = np.hstack([odor, gill_color, spore_color, gill_size.reshape(-1, 1)])

# Split data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize KNN classifier (you can tune n_neighbors)
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")

# Train the classifier
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Poisonous", "Edible"]))


# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X)

# Fit KNN on 2D-projected features for visualization
knn_vis = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn_vis.fit(X_vis, y)

# Create a meshgrid for decision boundaries
x_min, x_max = X_vis[:, 0].min() - 0.5, X_vis[:, 0].max() + 0.5
y_min, y_max = X_vis[:, 1].min() - 0.5, X_vis[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

# Predict over mesh
Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and points
plt.figure(figsize=(8, 6))
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
cmap_bold = ListedColormap(["#FF0000", "#00AA00"])

plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=30)
plt.title("KNN Decision Boundary (2D PCA projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(["Poisonous", "Edible"], loc="upper right")
plt.savefig("mushroom-classification.png", dpi=300)
plt.close()
