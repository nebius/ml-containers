FROM cr.eu-north1.nebius.cloud/soperator/cuda_base:13.0.2-ubuntu24.04-nccl2.28.7-1-14542c2 AS fryer

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

################################################

FROM cr.eu-north1.nebius.cloud/soperator/cuda_base:13.0.2-ubuntu24.04-nccl2.28.7-1-14542c2

ARG CUDA_VERSION
ARG NCCL_TESTS_VERSION
ARG PACKAGES_REPO_URL="https://github.com/nebius/slurm-deb-packages/releases/download"
ARG MLC_TOOL_URL="https://downloadmirror.intel.com/866182/mlc_v3.12.tgz"
ARG OPENMPI_VERSION=4.1.7a1-1.2404066
ARG OPENMPI_VERSION_SHORT=4.1.7a1
ARG OFED_VERSION=24.04-0.7.0.0
ARG UCX_VERSION=1.17.0-1.2404066

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      rdma-core ibverbs-utils wget tar sudo \
      libibverbs1 librdmacm1 libmlx5-1 libpci3 \
      libibumad3 ibverbs-providers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

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
    # NCCL tests
    wget -P /tmp "${PACKAGES_REPO_URL}/nccl_tests_${CUDA_VERSION}_ubuntu24.04/nccl-tests-perf-${ARCH}.tar.gz" && \
    tar -xvzf /tmp/nccl-tests-perf-${ARCH}.tar.gz -C /usr/bin && \
    rm -rf /tmp/nccl-tests-perf-${ARCH}.tar.gz && \
    # CUDA samples
    wget -P /tmp "${PACKAGES_REPO_URL}/cuda_samples_${CUDA_VERSION}_ubuntu24.04/cuda-samples-${ARCH}.tar.gz" && \
    tar -xvzf /tmp/cuda-samples-${ARCH}.tar.gz -C /usr/bin --strip-components=1 && \
    rm -rf /tmp/cuda-samples-${ARCH}.tar.gz && \
    # perftest
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

# Install dcgmi tools
# https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html
RUN apt-get update && \
    apt install -y datacenter-gpu-manager-4-cuda13 && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=fryer /gpu-fryer/target/release/gpu-fryer /usr/bin/gpu-fryer
