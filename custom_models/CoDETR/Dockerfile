FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Set the working directory to the mounted git repository
WORKDIR /launch

# Nvidia Keys
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key A4B469963BF863CC
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key F60F4B3D7FA2AF80

# Installations
RUN apt-get update && \
    apt-get install -y \
    git \
    nano

# Upgrade pip
RUN pip install --upgrade pip

# Install package requirements
COPY Co-DETR/requirements.txt /launch/requirements.txt
COPY Co-DETR/requirements /launch/requirements
RUN pip install -r /launch/requirements.txt
RUN rm /launch/requirements.txt
RUN rm -rf /launch/requirements

# Copy the entrypoint script to a different directory
COPY entrypoint_run.sh /usr/local/bin/entrypoint_run.sh
COPY entrypoint_compose.sh /usr/local/bin/entrypoint_compose.sh
RUN chmod +x /usr/local/bin/entrypoint_run.sh
RUN chmod +x /usr/local/bin/entrypoint_compose.sh


# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint_run.sh"]