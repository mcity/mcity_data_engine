from config.config import WANDB_CONFIG

# from wandb.sdk.launch import launch_add

import subprocess


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


#
#
# def launch_to_queue_python(name, project, config):
#    """
#    Adds a launch configuration to the queue for execution. More info:
#    - https://docs.wandb.ai/ref/python/launch-library/launch_add/
#    - https://docs.wandb.ai/guides/launch/add-job-to-queue/
#
#    Parameters:
#    name (str): The name of the launch.
#    project (str): The project name associated with the launch.
#    config (dict): The configuration dictionary for the launch.
#
#    Returns:
#    None
#    """
#    launch_add(
#        name=name,
#        entity=WANDB_CONFIG["entity"],
#        uri=WANDB_CONFIG["github"],
#        config=config,
#        docker_image=WANDB_CONFIG[
#            "docker_image"
#        ],  # "Dockerfile.wandb" # BUG --> build Docker Image on Docker Hub has the whole repository in it hardcopied
#        queue_name=WANDB_CONFIG["queue"],
#        project=project,
#        entry_point=["python", "wandb_runs/anomalib_run.py"],
#        # build=True, # FIXME Throws error "To build an image on queue a URI must be set."
#    )
#
#
def launch_to_queue_terminal(name, project, config_file):
    uri = WANDB_CONFIG["github"]
    entity = WANDB_CONFIG["entity"]
    queue_name = WANDB_CONFIG["queue"]
    docker = WANDB_CONFIG["docker_file"]
    entry_point = "python wandb_runs/anomalib_run.py"

    command = (
        f"wandb launch "
        f"--uri {uri} "
        f"--entity {entity} "
        f"--name {name} "
        f'--project "{project}" '
        f'--entry-point "{entry_point}" '
        f"--dockerfile {docker} "
        f"--queue {queue_name} "
        f"--config {config_file}"
    )

    subprocess.run(command, shell=True, check=True)
