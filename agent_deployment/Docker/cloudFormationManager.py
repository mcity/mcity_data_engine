import boto3
from config import Config
import time
import threading

class CloudFormationManager:
    def __init__(self, config: Config):
        self.config = config
        self.cf_client = boto3.client(
            'cloudformation',
            aws_access_key_id=config.AWS_ACCESS_KEY,
            aws_secret_access_key=config.AWS_SECRET_KEY,
            region_name=config.REGION
        )

    def create_stack(self, email):
        with open('deploy-agent-AWS-AMI.yml', 'r') as f:
            template_body = f.read()
        stack_name = f"{self.config.STACK_NAME}-{email.replace('@', '-').replace('.', '-')}-{int(time.time())}"
        keypair_name=f"{self.config.KEY_PAIR_NAME}-{email.replace('@', '-').replace('.', '-')}-{int(time.time())}"
        response = self.cf_client.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            RoleARN=self.config.ROLE_ARN,
            Parameters=[
                {'ParameterKey': 'KeyPairName', 'ParameterValue': keypair_name},
                {'ParameterKey': 'OPENAIAPIKEY', 'ParameterValue': self.config.OPENAIAPIKEY},
                {'ParameterKey': 'HFTOKEN', 'ParameterValue': self.config.HFTOKEN},
                {'ParameterKey': 'InstanceType', 'ParameterValue': self.config.INSTANCE_TYPE}
            ],
            Capabilities=['CAPABILITY_NAMED_IAM']
        )
        return stack_name
    def wait_for_stack(self, stack_name):
        waiter = self.cf_client.get_waiter('stack_create_complete')
        waiter.wait(StackName=stack_name)
        stack_info = self.cf_client.describe_stacks(StackName=stack_name)
        outputs = stack_info['Stacks'][0].get('Outputs', [])
        public_ip = next(o['OutputValue'] for o in outputs if o['OutputKey'] == 'PublicIP')
        return public_ip

    def delete_stack_later(self, stack_name, delay):
        def delete_stack():
            time.sleep(delay)
            try:
                self.cf_client.delete_stack(StackName=stack_name)
                print(f"Stack {stack_name} deleted after {delay} seconds.")
            except Exception as e:
                print(f"Error deleting stack {stack_name}: {e}")
        threading.Thread(target=delete_stack, daemon=True).start()    