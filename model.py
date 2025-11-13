import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
# Load dataset (example: diabetes)
data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.3, random_state=42)
# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_probs = log_model.predict_proba(X_test)[:, 1]
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
# Evaluate
print("Logistic Regression ROC-AUC:", roc_auc_score(y_test,
log_probs))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, rf_probs))
print("Log Loss (Logistic):", log_loss(y_test, log_probs))
print("Log Loss (RF):", log_loss(y_test, rf_probs))