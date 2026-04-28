## Download titanic_clean.csv
# !pip install boto3
## step 1 - 2
import boto3
import pandas as pd

AWS_ACCESS_KEY_ID = ''
AWS_SECRET_ACCESS_KEY = ''
BUCKET_NAME = 'jiajun-zhang'

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

s3.download_file(BUCKET_NAME, 'titanic_clean.csv', 'titanic_clean.csv')
df = pd.read_csv('titanic_clean.csv')

print("titanic_clean.csv len:",len(df))