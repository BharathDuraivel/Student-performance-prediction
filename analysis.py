import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("student.csv")

# -----------------------------
# GPA Distribution
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df["GPA"], bins=20)

plt.title("GPA Distribution")
plt.xlabel("GPA")
plt.ylabel("Number of Students")

plt.savefig("static/gpa_distribution.png")
plt.close()

# -----------------------------
# Study Time vs GPA
# -----------------------------
plt.figure(figsize=(8,5))

plt.scatter(
    df["StudyTimeWeekly"],
    df["GPA"]
)

plt.title("Study Time vs GPA")
plt.xlabel("Study Time Weekly")
plt.ylabel("GPA")

plt.savefig("static/study_vs_gpa.png")
plt.close()

# -----------------------------
# Feature Importance
# -----------------------------
X = df.drop(["GradeClass", "StudentID"], axis=1)
y = df["GradeClass"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

importance = model.feature_importances_

features = X.columns

plt.figure(figsize=(10,6))

plt.barh(features, importance)

plt.title("Feature Importance")

plt.savefig("static/feature_importance.png")
plt.close()

print("Graphs generated successfully!")