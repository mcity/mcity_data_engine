from flask import Flask, render_template, request
from redisManager import RedisManager
from cloudFormationManager import CloudFormationManager
from config import Config
from datetime import datetime

class App:
    def __init__(self):
        self.config = Config()
        self.redis_manager = RedisManager(self.config)
        self.cf_manager = CloudFormationManager(self.config)
        self.flask_app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        @self.flask_app.route('/', methods=['GET'])
        def show_form():
            return render_template('form.html')

        @self.flask_app.route('/submit', methods=['POST'])
        def submit():
            first_name = request.form['first_name'].strip()
            last_name = request.form['last_name'].strip()
            email = request.form['email'].strip().lower()
            organization = request.form['organization'].strip()

            today = self.redis_manager.get_today()

            # Whitelisting
            if email not in self.config.WHITELIST:
                # Per-user daily limit
                if self.redis_manager.get_user_daily_count(email) >= self.config.MAX_USER_REQUESTS_PER_DAY:
                    return "There is a usage limit per user. Please try again tomorrow."
                # Global daily limit
                if self.redis_manager.get_global_daily_count() >= self.config.MAX_USERS_PER_DAY:
                    return "Limited number of users allowed per day. Please try again tomorrow."

            # Save in Redis
            form_data = {
                "first_name": first_name,
                "last_name": last_name,
                "email": email,
                "organization": organization,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.redis_manager.store_user_data(email, form_data)
            self.redis_manager.increment_user_request(email)
            self.redis_manager.increment_global_count()

            try:
                # AWS CloudFormation
                stack_name = self.cf_manager.create_stack(email)
                public_ip = self.cf_manager.wait_for_stack(stack_name)
                url = f"http://{public_ip}:5225"
                # Schedule deletion
                self.cf_manager.delete_stack_later(stack_name, self.config.STACK_DELETE_DELAY)
                return render_template(
                        'submit_response.html',
                        public_url=url,
                        now=datetime.now()
                    )
            except Exception as e:
                return f"<h3>Error deploying stack:</h3><pre>{e}</pre>"

    def run(self):
        self.flask_app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    # Make sure templates directory exists and has form.html
    App().run()