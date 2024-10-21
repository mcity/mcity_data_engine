# Use the Red Hat Universal Base Image (UBI) as the base image
FROM redhat/ubi8:8.8

# Install dependencies and Python 3.11.5
RUN yum -y update && \
    yum -y install gcc openssl-devel bzip2-devel libffi-devel zlib-devel make wget unzip tar && \
    # Install dependencies for Git
    yum -y install libcurl-devel expat-devel tcl gettext && \
    # Download and install Git from source
    cd /usr/src && \
    wget https://github.com/git/git/archive/refs/tags/v2.39.3.zip && \
    unzip v2.39.3.zip && \
    cd git-2.39.3 && \
    make prefix=/usr/local all && \
    make prefix=/usr/local install && \
    # Install Python 3.11.5
    cd /usr/src && \
    curl -O https://www.python.org/ftp/python/3.11.5/Python-3.11.5.tgz && \
    tar xzf Python-3.11.5.tgz && \
    cd Python-3.11.5 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    ln -s /usr/local/bin/python3.11 /usr/bin/python3.11 && \
    yum clean all && \
    rm -rf /usr/src/Python-3.11.5* /usr/src/git-2.39.3* /usr/src/v2.39.3.zip

# Add Python 3.11 to alternatives and set it as the default
RUN alternatives --install /usr/bin/unversioned-python python /usr/local/bin/python3.11 1 && \
    alternatives --set python /usr/local/bin/python3.11

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip