import wandb
from wandb.sdk.launch import launch_add

# https://docs.wandb.ai/ref/python/launch-library/launch_add/

name = "API Test"
project_uri = "https://github.com/daniel-bogdoll/mcity_data_engine"
config = {"alpha": 0.5, "l1_ratio": 0.01}
# Run W&B project and create a reproducible docker environment
# on a local host
api = wandb.apis.internal.Api()
queue = "data-engine"
docker_image = "Dockerfile.wandb"
launch_add(
    name=name,
    uri=project_uri,
    config=config,
    docker_image=docker_image,
    queue_name=queue,
)
