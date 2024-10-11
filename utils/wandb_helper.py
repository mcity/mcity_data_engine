from config.config import WANDB_CONFIG
from wandb.sdk.launch import launch_add


def get_wandb_conf(config, value):
    """
    Retrieve a specific configuration value from a nested dictionary structure.

    Args:
        config (dict): The configuration dictionary containing nested 'overrides' and 'run_config' keys.
        value (str): The key whose corresponding value needs to be retrieved from the 'run_config' dictionary.

    Returns:
        The value associated with the specified key in the 'run_config' dictionary.

    Raises:
        KeyError: If the specified key is not found in the 'run_config' dictionary.
    """
    return config["overrides"]["run_config"][value]


def launch_to_queue(name, project, config):
    """
    Adds a launch configuration to the queue for execution. More info:
    - https://docs.wandb.ai/ref/python/launch-library/launch_add/
    - https://docs.wandb.ai/guides/launch/add-job-to-queue/

    Parameters:
    name (str): The name of the launch.
    project (str): The project name associated with the launch.
    config (dict): The configuration dictionary for the launch.

    Returns:
    None
    """
    launch_add(
        name=name,
        entity=WANDB_CONFIG["entity"],
        uri=WANDB_CONFIG["github"],
        config=config,
        docker_image=WANDB_CONFIG["docker"],
        queue_name=WANDB_CONFIG["queue"],
        project=project,
        entry_point=["python", "wandb_runs/anomalib_run.py"],
    )
