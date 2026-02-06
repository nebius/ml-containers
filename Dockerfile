# https://console.eu.nebius.com/project-e00managed-schedulers/registry/registry-e00hrt9na9xsn2px9f
FROM cr.eu-north1.nebius.cloud/ml-containers/ubuntu@sha256:8a48136281fe35ee40426bf9933cfff1b2fa9bdfbb82cb7a77a62a2544aa072f AS neubuntu

######
# Base Ubuntu image (Neubuntu as Nebius + Ubuntu) with modified repositories. It uses Nebius mirrors of the public Ubuntu
# repositories to address network accessibility issues, and it also resolves the snapshot problem for
# ARM packages (at the time of writing, Ubuntu does not support snapshots). The image additionally includes
# public repositories for Docker, NVIDIA CUDA, nvtop, and OFED.
######

LABEL org.opencontainers.image.authors="Pavel Sofronii pavel.sofrony@nebius.com"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=en_US.UTF-8

# Install reqirements for Nebius Ubuntu mirror
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      apt-transport-https \
      ca-certificates  \
      curl \
      gnupg \
      gnupg2 && \
    install -m 0755 -d /etc/apt/keyrings && \
    rm -f /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add Nebius Ubuntu mirrors
COPY ansible/roles/repos/files/nebius-ubuntu.sources /etc/apt/sources.list.d/nebius-ubuntu.sources
COPY ansible/roles/repos/files/nebius-ubuntu-security.sources /etc/apt/sources.list.d/nebius-ubuntu-security.sources
COPY ansible/roles/repos/files/nebius-ubuntu-updates.sources /etc/apt/sources.list.d/nebius-ubuntu-updates.sources

RUN apt-get update &&  \
    apt-get install -y --no-install-recommends \
      locales \
      tzdata \
      wget && \
    ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    locale-gen en_US.UTF-8 && \
    dpkg-reconfigure locales tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.utf8 \
    LC_ALL=en_US.utf8

# Install minimal python packages for Ansible
RUN apt-get update &&  \
    apt-get install -y \
        python3.12="3.12.3-1ubuntu0.10" \
        python3.12-venv="3.12.3-1ubuntu0.10"

# Install Ansible and base configs
COPY ansible/ansible.cfg /opt/ansible/ansible.cfg
COPY ansible/requirements.txt /opt/ansible/requirements.txt
COPY ansible/inventory/ /opt/ansible/inventory/
RUN cd /opt/ansible && /usr/bin/python3.12 -m venv .venv && \
    . .venv/bin/activate && pip install -r requirements.txt

ENV PATH="/opt/ansible/.venv/bin:${PATH}"

# Install python
COPY ansible/python.yml /opt/ansible/python.yml
COPY ansible/roles/python /opt/ansible/roles/python
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local python.yml

# Manage repositories
COPY ansible/repos.yml /opt/ansible/repos.yml
COPY ansible/roles/repos /opt/ansible/roles/repos
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local repos.yml

#######################################################################################################################
FROM neubuntu AS base

# Install common packages
COPY ansible/common-packages.yml /opt/ansible/common-packages.yml
COPY ansible/roles/common-packages /opt/ansible/roles/common-packages
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local common-packages.yml

# Install useful packages
RUN apt-get update && \
    apt -y install \
        iputils-ping \
        dnsutils \
        telnet \
        strace \
        vim \
        tree \
        lsof \
        tar

#######################################################################################################################
FROM base AS ansible_roles

######
# Image with all ansible roles from this repository
######

COPY ansible/ /opt/ansible/

#######################################################################################################################
FROM base AS slurm

######
# Image with preinstalled all slurm packages
######

ARG SLURM_VERSION
ENV SLURM_VERSION=$SLURM_VERSION

# Install slurm client and divert files
COPY ansible/slurm.yml /opt/ansible/slurm.yml
COPY ansible/roles/slurm /opt/ansible/roles/slurm
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local slurm.yml -e "slurm_version=${SLURM_VERSION}"

# Update linker cache
RUN ldconfig

#######################################################################################################################
FROM base AS cuda

######
# Base CUDA image with CUDA and mock packages for drivers.
#
# CUDA driver packages are required by various meta-packages, such as CUDA. However, in containerized
# environments, drivers are not installed via traditional packages but through tools like
# nvidia-container-toolkit or the k8s GPU Operator. When we mocking packages apt/dpkg thinking that
# the driver is installed via standard packages. As a result, other packages that depend on the driver
# as a dependency can be installed and used without issues.
######

ENV PATH=/usr/local/cuda/bin:${PATH}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install mock packages for NVIDIA drivers
COPY cuda/scripts/install_driver_mocks.sh /opt/bin/
RUN chmod +x /opt/bin/install_driver_mocks.sh && \
    /opt/bin/install_driver_mocks.sh && \
    rm /opt/bin/install_driver_mocks.sh

ARG CUDA_VERSION

# Install, hold and pin CUDA packages
# About CUDA packages https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages
COPY ansible/cuda.yml /opt/ansible/cuda.yml
COPY ansible/roles/cuda /opt/ansible/roles/cuda
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local cuda.yml -e "cuda_version=${CUDA_VERSION}"

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

#######################################################################################################################
FROM cuda AS training

######
# Minimal image for working with GPUs in distributed training. Essential tools (curl, wget, sudo, etc.),
# NCCL (without tests), rdma-core, ibverbs-utils, numactl, OpenMPI & UCX, Python
######

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      rdma-core="2404mlnx51-1.2404066" \
      ibverbs-utils="2404mlnx51-1.2404066" \
      libibverbs1 librdmacm1 libmlx5-1 libpci3 \
      libibumad3 ibverbs-providers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install OpenMPI, UCX, and related config
COPY ansible/openmpi.yml /opt/ansible/openmpi.yml
COPY ansible/roles/openmpi /opt/ansible/roles/openmpi
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local openmpi.yml

#######################################################################################################################
FROM cuda AS fryer

######
# fryer binary build for different CUDA versions
######

ENV REPO_URL="https://github.com/huggingface/gpu-fryer"
ENV TAG="v1.1.0"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN git clone --depth 1 --branch "${TAG}" "${REPO_URL}" /gpu-fryer

WORKDIR /gpu-fryer

RUN cargo build --release

#######################################################################################################################
FROM training AS training_diag

######
# Image for training and diagnostics with a specific tools
######

# Install dcgmi tools
# https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html
COPY ansible/dcgmi.yml /opt/ansible/dcgmi.yml
COPY ansible/roles/dcgmi /opt/ansible/roles/dcgmi
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local dcgmi.yml

# Install Intel MLC binary
COPY ansible/mlc.yml /opt/ansible/mlc.yml
COPY ansible/roles/mlc /opt/ansible/roles/mlc
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local mlc.yml

# Download NCCL tests executables
ARG CUDA_VERSION
ARG NCCL_TESTS_VERSION
COPY ansible/nccl-tests.yml /opt/ansible/nccl-tests.yml
COPY ansible/roles/nccl-tests /opt/ansible/roles/nccl-tests
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local nccl-tests.yml -e "nccl_tests_cuda_version=${CUDA_VERSION}" \
    -e "nccl_tests_version=${NCCL_TESTS_VERSION}"

# Download cuda-samples executables
COPY ansible/cuda-samples.yml /opt/ansible/cuda-samples.yml
COPY ansible/roles/cuda-samples /opt/ansible/roles/cuda-samples
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local cuda-samples.yml -e "cuda_samples_cuda_version=${CUDA_VERSION}"

# Download perftest executables
COPY ansible/perftest.yml /opt/ansible/perftest.yml
COPY ansible/roles/perftest /opt/ansible/roles/perftest
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local perftest.yml -e "perftest_cuda_version=${CUDA_VERSION}"

COPY --from=fryer /gpu-fryer/target/release/gpu-fryer /usr/bin/gpu-fryer

#######################################################################################################################
FROM training_diag AS slurm_training_diag

######
# CUDA Image for training and diagnostics with a slurm client
######

# Install slurm client and divert files
COPY ansible/slurm-client.yml /opt/ansible/slurm-client.yml
COPY ansible/roles/slurm-client /opt/ansible/roles/slurm-client
COPY ansible/roles/slurm-divert /opt/ansible/roles/slurm-divert
RUN cd /opt/ansible && \
    ansible-playbook -i inventory/ -c local slurm-client.yml
