FROM cr.eu-north1.nebius.cloud/soperator/ubuntu:noble

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=en_US.UTF-8
ENV LIBNCCL_VERSION=2.28.7-1
ENV DCGMI_VERSION=1:4.4.2-1

RUN apt-get update &&  \
    apt-get install -y --no-install-recommends \
      gnupg2  \
      ca-certificates \
      locales \
      tzdata \
      wget \
      curl && \
    ARCH=$(uname -m) && \
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
    ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    locale-gen en_US.UTF-8 && \
    dpkg-reconfigure locales tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG="en_US.UTF-8" \
	LC_CTYPE="en_US.UTF-8" \
	LC_NUMERIC="en_US.UTF-8" \
	LC_TIME="en_US.UTF-8" \
	LC_COLLATE="en_US.UTF-8" \
	LC_MONETARY="en_US.UTF-8" \
	LC_MESSAGES="en_US.UTF-8" \
	LC_PAPER="en_US.UTF-8" \
	LC_NAME="en_US.UTF-8" \
	LC_ADDRESS="en_US.UTF-8" \
	LC_TELEPHONE="en_US.UTF-8" \
	LC_MEASUREMENT="en_US.UTF-8" \
	LC_IDENTIFICATION="en_US.UTF-8"

ENV PATH=/usr/local/cuda/bin:${PATH}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Add Nebius public registry
RUN curl -fsSL https://dr.nebius.cloud/public.gpg -o /usr/share/keyrings/nebius.gpg.pub && \
    codename="$(. /etc/os-release && echo $VERSION_CODENAME)" && \
    echo "deb [signed-by=/usr/share/keyrings/nebius.gpg.pub] https://dr.nebius.cloud/ $codename main" > /etc/apt/sources.list.d/nebius.list && \
    echo "deb [signed-by=/usr/share/keyrings/nebius.gpg.pub] https://dr.nebius.cloud/ stable main" >> /etc/apt/sources.list.d/nebius.list


# Install mock packages for NVIDIA drivers
COPY cuda/scripts/install_driver_mocks.sh /opt/bin/
RUN chmod +x /opt/bin/install_driver_mocks.sh && \
    /opt/bin/install_driver_mocks.sh && \
    rm /opt/bin/install_driver_mocks.sh

# Install dcgmi (DCGM for CUDA 13)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        datacenter-gpu-manager-4-cuda13=${DCGMI_VERSION} && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# About CUDA packages https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages
RUN apt update && \
    apt install -y \
        cuda=13.0.2-1 \
        libcublas-dev-13-0 \
        libcudnn9-cuda-13=9.15.0.58-1 \
        libcudnn9-dev-cuda-13=9.15.0.58-1 \
        libcudnn9-headers-cuda-13=9.15.0.58-1 \
        libnccl-dev=${LIBNCCL_VERSION}+cuda13.0 \
        libnccl2=${LIBNCCL_VERSION}+cuda13.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Disable automatic upgrades for CUDA packages
RUN apt-mark hold \
    cuda=13.0.2-1 \
    libcublas-dev-13-0 \
    libcudnn9-cuda-13=9.15.0.58-1 \
    libcudnn9-dev-cuda-13=9.15.0.58-1 \
    libcudnn9-headers-cuda-13=9.15.0.58-1 \
    libnccl-dev=${LIBNCCL_VERSION}+cuda13.0 \
    libnccl2=${LIBNCCL_VERSION}+cuda13.0 \
    datacenter-gpu-manager-4-cuda13

COPY cuda/pin_packages/ /etc/apt/preferences.d/
RUN apt update

RUN echo "export PATH=\$PATH:/usr/local/cuda/bin" > /etc/profile.d/path_cuda.sh && \
    . /etc/profile.d/path_cuda.sh

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
