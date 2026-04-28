## step 7 - 8
# Install GitHub integration and model serialization libraries
# !pip install PyGithub joblib -q

import joblib
from github import Github, Auth
from datetime import datetime

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


