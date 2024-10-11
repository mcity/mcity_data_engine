import wandb
from wandb.sdk.launch import launch_add, launch

# https://docs.wandb.ai/ref/python/launch-library/launch_add/


def job_to_queue(config):

    entity = "mcity"
    name = "API Test"
    project_uri = "https://github.com/daniel-bogdoll/mcity_data_engine"
    # Run W&B project and create a reproducible docker environment
    # on a local host
    # api = wandb.apis.internal.Api()
    queue = "data-engine"
    docker_image = "dbogdollresearch/mcity_data_engine:latest"
    project = "mcity-data-engine"
    launch_add(
        name=name,
        entity=entity,
        uri=project_uri,
        config=config,
        docker_image=docker_image,
        queue_name=queue,
        project=project,
    )


job_to_queue(config={"alpha": 0.5, "l1_ratio": 0.01})
job_to_queue(config={"alpha": 1, "l1_ratio": 0.05})
