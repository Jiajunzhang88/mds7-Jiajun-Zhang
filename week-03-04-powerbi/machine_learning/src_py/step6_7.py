## step 5 - 6
# Install machine learning libraries
# !pip install xgboost scikit-learn -q

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Prepare data using the top 5 features selected in the previous step
X = df[top_5_features]
y = df['Survived']

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model 1: Logistic Regression ---
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_f1 = f1_score(y_test, lr_pred)

# --- Model 2: XGBoost ---
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_f1 = f1_score(y_test, xgb_pred)

# Print performance results
print(f"Logistic Regression F1 Score: {lr_f1:.4f}")
print(f"XGBoost F1 Score: {xgb_f1:.4f}")

# Compare models and select the one with the higher F1 score
best_model_name = "XGBoost" if xgb_f1 > lr_f1 else "Logistic Regression"
best_pred = xgb_pred if xgb_f1 > lr_f1 else lr_pred

print(f"The better performing model is: {best_model_name}")

# Generate and plot the Confusion Matrix for the best model
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f"Confusion Matrix: {best_model_name}")
plt.show()
