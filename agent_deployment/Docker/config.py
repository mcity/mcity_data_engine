import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    REGION = os.getenv('REGION', 'us-east-1')
    OPENAIAPIKEY = os.getenv('OPENAIAPIKEY')
    HFTOKEN = os.getenv('HFTOKEN')
    ROLE_ARN = os.getenv('ROLE_ARN', 'arn:aws:iam::1234567800000:role/mcity-data-engine-agent-cf-role')
    KEY_PAIR_NAME = os.getenv('KEY_PAIR_NAME', 'mcity-data-engine-key')
    INSTANCE_TYPE = os.getenv('INSTANCE_TYPE', 'r5.large')
    STACK_NAME = "mcity-data-engine-agent"
    WHITELIST = set([
        "rpatnaik@umich.edu",
        "admin@umich.edu"
        # Add more whitelisted emails here.
    ])
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    MAX_USERS_PER_DAY = 20
    MAX_USER_REQUESTS_PER_DAY = 1
    STACK_DELETE_DELAY = 3600  # seconds