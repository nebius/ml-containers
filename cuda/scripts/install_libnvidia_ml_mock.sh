#!/bin/bash

set -e

# Install deb package mocking tool
apt update -y && apt install -y equivs

# Build mock packages
mkdir -p /tmp/mocks
pushd /tmp/mocks
  echo "Generate mock package definitions"
  cat > "mock_package.ctl" <<EOF
Package: libnvidia-ml1-fake
Version: 99999999-fake
Architecture: all
Provides: libnvidia-ml1, libnvidia-ml.so.1
Description: Fake NVML provider (GPU operator handles real libs)
EOF
  MAKEFLAGS="-j$(nproc)" equivs-build mock_package.ctl

  echo "Install mock packages"
  dpkg -i ./libnvidia-ml1-fake*.deb
if ! popd; then
  echo "Warning: popd failed, but continuing execution."
fi

# Cleanup
rm -rf /mocks
apt-get clean && rm -rf /var/lib/apt/lists/*
