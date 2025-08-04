FROM ubuntu:22.04

# Install minimal required packages
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/mlc

# Download and unpack Intel MLC tool
RUN wget https://downloadmirror.intel.com/834254/mlc_v3.11b.tgz && \
    tar -xzf mlc_v3.11b.tgz && \
    chmod +x ./Linux/mlc && \
    cp ./Linux/mlc /usr/local/bin/mlc
