# cr.eu-north1.nebius.cloud/ml-containers/ubuntu:noble
FROM cr.eu-north1.nebius.cloud/ml-containers/ubuntu@sha256:8a48136281fe35ee40426bf9933cfff1b2fa9bdfbb82cb7a77a62a2544aa072f AS neubuntu

LABEL org.opencontainers.image.authors="Pavel Sofronii <pavel.sofrony@nebius.com>"

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

# Add Nebius Ubuntu mirror
COPY repos/nebius-ubuntu.sources /etc/apt/sources.list.d/nebius-ubuntu.sources

RUN apt-get update &&  \
    apt-get install -y --no-install-recommends \
      locales \
      tzdata \
      wget \
      curl && \
    ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    locale-gen en_US.UTF-8 && \
    dpkg-reconfigure locales tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.utf8 \
    LC_ALL=en_US.utf8

# Add Nebius public registry
RUN curl -fsSL https://dr.nebius.cloud/public.gpg -o /usr/share/keyrings/nebius.gpg.pub && \
    codename="$(. /etc/os-release && echo $VERSION_CODENAME)" && \
    echo "deb [signed-by=/usr/share/keyrings/nebius.gpg.pub] https://dr.nebius.cloud/ $codename main" > /etc/apt/sources.list.d/nebius.list && \
    echo "deb [signed-by=/usr/share/keyrings/nebius.gpg.pub] https://dr.nebius.cloud/ stable main" >> /etc/apt/sources.list.d/nebius.list

#######################################################################################################################
FROM neubuntu AS cuda

RUN ARCH=$(uname -m) && \
        case "$ARCH" in \
          x86_64) ARCH_DEB=x86_64 ;; \
          aarch64) ARCH_DEB=sbsa ;; \
          *) echo "Unsupported architecture: ${ARCH}" && exit 1 ;; \
        esac && \
        echo "Using architecture: ${ARCH_DEB}" && \
    UBUNTU_VERSION_ID=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2 | tr -d .) && \
        echo "Using architecture: ${ARCH_DEB}, ubuntu version: ubuntu${UBUNTU_VERSION_ID}" && \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION_ID}/${ARCH_DEB}/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm -rf cuda-keyring_1.1-1_all.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/cuda/bin:${PATH}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install mock packages for NVIDIA drivers
COPY cuda/scripts/install_driver_mocks.sh /opt/bin/
RUN chmod +x /opt/bin/install_driver_mocks.sh && \
    /opt/bin/install_driver_mocks.sh && \
    rm /opt/bin/install_driver_mocks.sh

ARG CUDA_MAJOR
ARG CUDA_MINOR
ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG LIBNCCL_VERSION

# About CUDA packages https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda=${CUDA_VERSION} \
        libcublas-dev-${CUDA_MAJOR}-${CUDA_MINOR} \
        libcudnn9-cuda-${CUDA_MAJOR}=${CUDNN_VERSION} \
        libcudnn9-dev-cuda-${CUDA_MAJOR}=${CUDNN_VERSION} \
        libcudnn9-headers-cuda-${CUDA_MAJOR}=${CUDNN_VERSION} \
        libnccl-dev=${LIBNCCL_VERSION}+cuda${CUDA_MAJOR}.${CUDA_MINOR} \
        libnccl2=${LIBNCCL_VERSION}+cuda${CUDA_MAJOR}.${CUDA_MINOR} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Disable automatic upgrades for CUDA packages
RUN apt-mark hold \
        cuda \
        libcublas-dev-${CUDA_MAJOR}-${CUDA_MINOR} \
        libcudnn9-cuda-${CUDA_MAJOR} \
        libcudnn9-dev-cuda-${CUDA_MAJOR} \
        libcudnn9-headers-cuda-${CUDA_MAJOR} \
        libnccl-dev \
        libnccl2

COPY cuda/pin_packages/ /etc/apt/preferences.d/

RUN echo "export PATH=\$PATH:/usr/local/cuda/bin" > /etc/profile.d/path_cuda.sh && \
    . /etc/profile.d/path_cuda.sh

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

#######################################################################################################################
FROM cuda AS training

ARG CUDA_MAJOR
ARG DCGMI_VERSION

# Install dcgmi tools
# https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        datacenter-gpu-manager-4-cuda${CUDA_MAJOR}=${DCGMI_VERSION} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#######################################################################################################################
FROM cuda AS fryer

ENV REPO_URL="https://github.com/huggingface/gpu-fryer"
ENV TAG="v1.1.0"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libssl-dev \
        pkg-config \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN git clone --depth 1 --branch "${TAG}" "${REPO_URL}" /gpu-fryer

WORKDIR /gpu-fryer

RUN cargo build --release

#######################################################################################################################
FROM training AS training_diag

ARG CUDA_VERSION
ARG NCCL_TESTS_VERSION
ARG PACKAGES_REPO_URL="https://github.com/nebius/slurm-deb-packages/releases/download"
ARG MLC_TOOL_URL="https://downloadmirror.intel.com/866182/mlc_v3.12.tgz"
ARG OPENMPI_VERSION
ARG OPENMPI_VERSION_SHORT
ARG OFED_VERSION
ARG UCX_VERSION

# Install OpenMPI, UCX, and related config
RUN DISTRO="$(. /etc/os-release && echo "${ID}${VERSION_ID}")"; \
    ALT_ARCH="$(uname -m)"; \
    cd /etc/apt/sources.list.d; \
    wget https://linux.mellanox.com/public/repo/mlnx_ofed/${OFED_VERSION}/${DISTRO}/mellanox_mlnx_ofed.list; \
    wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add -; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        openmpi=${OPENMPI_VERSION} \
        ucx=${UCX_VERSION}; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*; \
    echo "export PATH=\$PATH:/usr/mpi/gcc/openmpi-${OPENMPI_VERSION_SHORT}/bin" > /etc/profile.d/path_openmpi.sh; \
    chmod +x /etc/profile.d/path_openmpi.sh; \
    printf "/lib/${ALT_ARCH}-linux-gnu\n/usr/lib/${ALT_ARCH}-linux-gnu\n/usr/local/cuda/targets/${ALT_ARCH}-linux/lib\n/usr/mpi/gcc/openmpi-${OPENMPI_VERSION_SHORT}/lib\n" > /etc/ld.so.conf.d/openmpi.conf; \
    ldconfig


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      rdma-core="2404mlnx51-1.2404066" \
      ibverbs-utils="2404mlnx51-1.2404066" \
      tar sudo \
      libibverbs1 librdmacm1 libmlx5-1 libpci3 \
      libibumad3 ibverbs-providers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Intel MLC binary
RUN mkdir -p /tmp/mlc && \
    wget "$MLC_TOOL_URL" -O /tmp/mlc/mlc.tgz && \
    tar -xzf /tmp/mlc/mlc.tgz -C /tmp/mlc && \
    chmod +x /tmp/mlc/Linux/mlc && \
    cp /tmp/mlc/Linux/mlc /usr/local/bin/mlc && \
    rm -rf /tmp/mlc

# Download NCCL tests, CUDA samples, and perftest executables
RUN ARCH=$(uname -m) && \
    echo "Using architecture: $ARCH" && \
    echo "Downloading NCCL tests" && \
    wget -P /tmp "${PACKAGES_REPO_URL}/nccl_tests_${CUDA_VERSION}_ubuntu24.04/nccl-tests-perf-${ARCH}.tar.gz" && \
    tar -xvzf /tmp/nccl-tests-perf-${ARCH}.tar.gz -C /usr/bin && \
    rm -rf /tmp/nccl-tests-perf-${ARCH}.tar.gz && \
    echo "Downloading CUDA samples" && \
    wget -P /tmp "${PACKAGES_REPO_URL}/cuda_samples_${CUDA_VERSION}_ubuntu24.04/cuda-samples-${ARCH}.tar.gz" && \
    tar -xvzf /tmp/cuda-samples-${ARCH}.tar.gz -C /usr/bin --strip-components=1 && \
    rm -rf /tmp/cuda-samples-${ARCH}.tar.gz && \
    echo "Downloading perftest" && \
    wget -P /tmp "${PACKAGES_REPO_URL}/perftest_${CUDA_VERSION}_ubuntu24.04/perftest-${ARCH}.tar.gz" && \
    tar -xvzf /tmp/perftest-${ARCH}.tar.gz -C /usr/bin && \
    chmod +x /usr/bin/ib_* && \
    rm -rf /tmp/perftest-${ARCH}.tar.gz

# Install numactl
RUN apt update && \
    apt install -y \
        numactl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=fryer /usr/local/bin/gpu-fryer /usr/bin/gpu-fryer
