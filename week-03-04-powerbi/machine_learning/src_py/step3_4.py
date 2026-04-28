## step 3 - 4
# !pip install seaborn matplotlib -q

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Correlation Analysis
# Select only numeric columns for correlation calculation
numeric_df = df.select_dtypes(include=['number'])

# Calculate the absolute correlation values between all numeric variables and 'Survived'
# Using .abs() ensures we capture strong negative correlations (like Gender) as important features
correlations = numeric_df.corr()['Survived'].abs().sort_values(ascending=False)

# Select the top 5 features (excluding 'Survived' itself at index 0)
top_5_features = correlations.iloc[1:6].index.tolist()
print(f"Selected Top 5 Features: {top_5_features}")

# 2. Visualization (Multivariate Analysis)
# Create a heatmap to visualize the relationship between the top 5 features and the target
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df[top_5_features + ['Survived']].corr(), annot=True, cmap='RdBu_r', center=0)
plt.title("Correlation Heatmap: Top 5 Features vs Survived")
plt.show()