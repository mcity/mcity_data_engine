# Test if the Data Engine builds on a UofM Lighthouse SLURM compute node
# https://its.umich.edu/advanced-research-computing/high-performance-computing/lighthouse
# Use the Red Hat Universal Base Image (UBI) as the base image
FROM redhat/ubi8:8.8

# Install Python 3.11.5
RUN yum -y update && \
    yum -y install gcc openssl-devel bzip2-devel libffi-devel zlib-devel make && \
    cd /usr/src && \
    curl -O https://www.python.org/ftp/python/3.11.5/Python-3.11.5.tgz && \
    tar xzf Python-3.11.5.tgz && \
    cd Python-3.11.5 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    ln -s /usr/local/bin/python3.11 /usr/bin/python3.11 && \
    yum clean all && \
    rm -rf /usr/src/Python-3.11.5*

# Add Python 3.11 to alternatives and set it as the default
RUN alternatives --install /usr/bin/unversioned-python python /usr/local/bin/python3.11 1 && \
    alternatives --set python /usr/local/bin/python3.11

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip