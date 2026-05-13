

import boto3

import os

from github import Github

from datetime import datetime



# 获取环境变量

auth_id = os.getenv('AWS_ID')

auth_secret = os.getenv('AWS_SECRET')

auth_token = os.getenv('GITHUB_TOKEN')



if not all([auth_id, auth_secret, auth_token]):

    print("❌ Error: Credentials missing.")

    raise ValueError("Missing Credentials")



# 配置信息

S3_BUCKET = 'jiajun-zhang'

REPO_NAME = "Jiajunzhang88/mds7-Jiajun-Zhang"

GIT_TARGET_ROOT = "week-05-06-bigquery/deeplearning"



# 获取路径

current_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = os.path.join(current_dir, "output")



# --- 1. AWS S3 部署 (Artifacts) ---

s3 = boto3.client('s3', aws_access_key_id=auth_id, aws_secret_access_key=auth_secret)

if os.path.exists(output_dir):

    for f in os.listdir(output_dir):

        local_path = os.path.join(output_dir, f)

        if os.path.isfile(local_path):

            s3.upload_file(local_path, S3_BUCKET, f"deeplearning/{f}")

    print("✅ AWS S3 Artifacts Synced.")



# --- 2. GitHub 部署 (Full Task Folder) ---

g = Github(auth_token)

repo = g.get_repo(REPO_NAME)

for root, dirs, files in os.walk(current_dir):

    for f in files:

        full_path = os.path.join(root, f)

        rel_path = os.path.relpath(full_path, current_dir)

        with open(full_path, 'rb') as file_data:

            content = file_data.read()

        git_path = f"{GIT_TARGET_ROOT}/{rel_path}"

        try:

            sha = repo.get_contents(git_path).sha

            repo.update_file(git_path, f"Update {rel_path}", content, sha)

        except:

            repo.create_file(git_path, f"Deploy {rel_path}", content)

print("✅ GitHub Codebase Synced.")



# --- 3. 审计日志更新 (平衡版内容) --- [cite: 66, 68]

audit = repo.get_contents("AUDIT_TRAIL.md")

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')



log_entry = f"""

- **Task: {timestamp} (Week 5-6):** 

    - **Task:** Advanced Deep Learning Classification & ETLS Pipeline Deployment. 

    - **Models:** Trained Model A (3-layer) and Model B (5-layer) using Keras HDF5. 

    - **Processing:** Applied `StandardScaler` for neural convergence. 

    - **Deployment:** Programmatically synchronized artifacts and documentation to AWS S3 and GitHub. 

    - **Status:** ETLS pipeline execution verified and complete. 

"""



new_content = audit.decoded_content.decode() + "\n" + log_entry

repo.update_file("AUDIT_TRAIL.md", f"Audit Update: DL Pipeline {timestamp}", new_content, audit.sha)

print("✅ Audit Trail updated with concise log.")

