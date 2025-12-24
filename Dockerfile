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

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        datacenter-gpu-manager-4-cuda${CUDA_MAJOR}=${DCGMI_VERSION} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
