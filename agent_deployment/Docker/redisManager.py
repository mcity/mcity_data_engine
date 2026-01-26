import redis
from datetime import datetime
from config import Config
import json 

class RedisManager:
    def __init__(self, config: Config):
        self.client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )

    def get_today(self):
        return datetime.utcnow().strftime('%Y-%m-%d')

    def user_key(self, email):
        return f"email:{email.lower()}"

    def user_daily_key(self, email, today):
        return f"{self.user_key(email)}:date:{today}"

    def daily_count_key(self, today):
        return f"requests:{today}:count"

    def increment_user_request(self, email):
        today = self.get_today()
        key = self.user_daily_key(email, today)
        self.client.incr(key)
        self.client.expire(key, 86400)

    def increment_global_count(self):
        today = self.get_today()
        key = self.daily_count_key(today)
        self.client.incr(key)
        self.client.expire(key, 86400)

    def get_user_daily_count(self, email):
        today = self.get_today()
        key = self.user_daily_key(email, today)
        return int(self.client.get(key) or 0)

    def get_global_daily_count(self):
        today = self.get_today()
        key = self.daily_count_key(today)
        return int(self.client.get(key) or 0)

    def store_user_data(self, email, data: dict):
        key = f"{self.user_key(email)}:last_submission"
        self.client.set(key, json.dumps(data))