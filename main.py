import os
import boto3
from dotenv import load_dotenv

load_dotenv()
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')

s3 = boto3.client('s3',
                  aws_access_key_id='AWS_ACCESS_KEY',
                  aws_secret_access_key='AWS_SECRET_KEY',
                  region_name='AWS_REGION')

