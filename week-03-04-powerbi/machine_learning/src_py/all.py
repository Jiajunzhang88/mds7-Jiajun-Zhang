
# !pip install boto3

import boto3
import pandas as pd
import urllib.request
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt




import joblib
from github import Github, Auth
from datetime import datetime




AWS_ACCESS_KEY_ID ='AKIASFKCF2FR4UJJLYR3'
AWS_SECRET_ACCESS_KEY ='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'
REGION_NAME = 'eu-north-1'
BUCKET_NAME = 'jiajun-zhang'

# Initialize the Boto3 S3 Client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=REGION_NAME
)

 # Fetch the raw Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
raw_filename = "titanic_raw.csv"
urllib.request.urlretrieve(url, raw_filename)

# Push the raw dataset to S3
print(f"Uploading '{raw_filename}' to S3 bucket '{BUCKET_NAME}'...")
s3_client.upload_file(raw_filename, BUCKET_NAME, raw_filename)
print("✅ Raw upload complete!")



# import pandas as pd

# 1. 读取原始数据
raw_filename = "titanic_raw.csv"
df = pd.read_csv(raw_filename)

# 2. 处理缺失值
# Age 用中位数填充
df['Age'].fillna(df['Age'].median(), inplace=True)

# Embarked 用众数填充
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 删除 Cabin 列（缺失率太高）
df.drop('Cabin', axis=1, inplace=True)

# 3. 类别变量编码
# Sex 映射为数值
df['Sex_male'] = (df['Sex'] == 'male').astype(int)  # 1=男, 0=女
df.drop('Sex', axis=1, inplace=True)

# Embarked 独热编码
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 4. 保存清洗后的数据
clean_filename = "titanic_clean.csv"
df.to_csv(clean_filename, index=False)

print(f"数据清洗完成，已保存为 {clean_filename}")



# 读取清洗后的数据
df_clean = pd.read_csv("titanic_clean.csv")
print("清洗后数据形状:", df_clean.shape)

# 注意：这里要用 df_clean，而不是 df
numeric_df = df_clean.select_dtypes(include=['number'])
correlations = numeric_df.corr()['Survived'].abs().sort_values(ascending=False)
top_5_features = correlations.iloc[1:6].index.tolist()

print(f"Selected Top 5 Features: {top_5_features}")


# 2. Visualization - Histogram/Bar Chart
# plt.figure(figsize=(10, 6))
# # 只取前5个特征的相关系数（不包含Survived自己）
# top_corr = correlations.iloc[1:6]
# sns.barplot(x=top_corr.values, y=top_corr.index, palette='RdBu_r')
# plt.xlabel('Absolute Correlation with Survived')
# plt.ylabel('Features')
# plt.title('Top 5 Features Correlation with Survived')
# plt.show()




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
print(f"The better performing model is: {best_model_name}")

# --- Plot as Histogram/Bar Chart ---
# models = ['Logistic Regression', 'XGBoost']
# f1_scores = [lr_f1, xgb_f1]

# plt.figure(figsize=(8, 6))
# plt.bar(models, f1_scores, color=['#3498db', '#e74c3c'])
# plt.xlabel('Model')
# plt.ylabel('F1 Score')
# plt.title('Model Performance Comparison (F1 Score)')
# plt.ylim(0, 1)  # F1 score range is 0 to 1

# # Add value labels on top of bars
# for i, v in enumerate(f1_scores):
#     plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

# plt.show()

# 1. Save the best model as a .pkl file
MODEL_FILE = 'best_titanic_model.pkl'
# Select the model with the higher F1 score (Logic from Step 6)
best_model = xgb_model if xgb_f1 > lr_f1 else lr_model
joblib.dump(best_model, MODEL_FILE)
print(f"Model successfully saved as {MODEL_FILE}")

# 2. Upload the model to AWS S3
# Ensure the path aligns with the machine_learning folder requirement
s3_path = f"machine_learning/{MODEL_FILE}"
s3.upload_file(MODEL_FILE, BUCKET_NAME, s3_path)
print(f"Model successfully uploaded to S3 bucket: {s3_path}")

# 3. Upload the model to GitHub and update the Audit Trail
# --- GitHub Configuration ---
STUDENT_TOKEN = ''
REPO_NAME = 'Jiajunzhang88/mds7-Jiajun-Zhang'
GITHUB_TARGET_PATH = f"week-03-04-powerbi/machine_learning/{MODEL_FILE}"

auth = Auth.Token(STUDENT_TOKEN)
g = Github(auth=auth)
repo = g.get_repo(REPO_NAME)

# Read the binary content of the saved model
with open(MODEL_FILE, 'rb') as f:
    content = f.read()

try:
    # Attempt to update the file if it already exists in the repository
    contents = repo.get_contents(GITHUB_TARGET_PATH, ref="main")
    repo.update_file(
        contents.path, 
        "Update trained model for Lecture 4", 
        content, 
        contents.sha, 
        branch="main"
    )
    print("GitHub: Model file updated successfully")
except:
    # Create the file if it does not exist
    repo.create_file(
        GITHUB_TARGET_PATH, 
        "Initial upload of trained model for Lecture 4", 
        content, 
        branch="main"
    )
    print("GitHub: Model file created successfully")

# Update AUDIT_TRAIL.md
print("Updating AUDIT_TRAIL.md...")
try:
    audit_file = repo.get_contents("AUDIT_TRAIL.md", ref="main")
    current_content = audit_file.decoded_content.decode('utf-8')
    date_str = datetime.now().strftime("%Y-%m-%d")
 
    new_entry = f"""
    ## Week 4: Titanic ML Project
    * **Date:** {date_str}
    * **Milestone:** Lecture 4 ML Pipeline Task Completed. BestModel: {best_model_name}.
    * **Notes:** Trained {best_model_name} using top 5 features ({features_txt}). Achieved F1 score of {f1_val:.4f}. Model artifacts (.pkl) and notebook pushed to S3 and GitHub 'machine_learning' folder.
    """
    updated_content = current_content + new_entry
    repo.update_file(
        path="AUDIT_TRAIL.md",
        message="Update Audit Trail for Week 4 pipeline",
        content=updated_content,
        sha=audit_file.sha,
        branch="main"
    )
    print("AUDIT_TRAIL.md updated successfully.")
except Exception as e:
    print(f"Error updating audit trail: {e}")
