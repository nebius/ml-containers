FROM cr.eu-north1.nebius.cloud/soperator/cuda_base:12.9.0-ubuntu24.04-nccl2.26.5-1-1ba1d76

# Download NCCL tests executables
ARG CUDA_VERSION
ARG NCCL_TESTS_VERSION
ARG PACKAGES_REPO_URL="https://github.com/nebius/slurm-deb-packages/releases/download"

RUN ARCH=$(uname -m) && \
    case "$ARCH" in \
      x86_64) ARCH_DEB=x64 ;; \
      aarch64) ARCH_DEB=arm64 ;; \
      *) echo "Unsupported architecture: $ARCH" && exit 1 ;; \
    esac && \
    echo "Using architecture: $ARCH_DEB" && \
    wget -P /tmp "${PACKAGES_REPO_URL}/nccl_tests_${CUDA_VERSION}_ubuntu24.04/nccl-tests-perf-${ARCH_DEB}.tar.gz" && \
    tar -xvzf /tmp/nccl-tests-perf-${ARCH_DEB}.tar.gz -C /usr/bin && \
    rm -rf /tmp/nccl-tests-perf-${ARCH_DEB}.tar.gz
