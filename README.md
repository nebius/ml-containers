# Docker Containers for Soperator and Related Projects

This repository contains Docker container definitions used in [Soperator](https://github.com/nebius/soperator) and other related projects. The goal is to provide reproducible, isolated environments for components required in CI/CD pipelines, production deployments, and hardware validation.

## Containers

The list of containers will be expanded over time as new images are added to the repository.

## DOCA kernel module builds

The Slurm and training image stages install the full `doca-ofed` profile. The
shared `doca_ofed` role installs headers for `6.11.0-1016-nvidia` (Debian package
version `6.11.0-1016.16`) before installing DOCA. These headers satisfy the DKMS
packages' recommended `linux-headers` dependency and select the target kernel in a clean
image, independently of the kernel running on the build host.

DOCA 3.3's `mlnx-ofed-kernel` DKMS configuration passes the header directory to
`configure` but omits the kernel version. Without an override, `configure` uses
`uname -r` and can disable features based on the build host's kernel while
compiling against different headers. The role installs a version-specific DKMS
override that passes `--kernel-version=$kernelver` alongside the target headers.

The target kernel, header package version and OFED module version are pinned in
`ansible/roles/doca_ofed/defaults/main.yml`. Keep the module version aligned with
the selected DOCA release. `6.11.0-1016-nvidia-64k` is a separate kernel target;
it is not interchangeable with `6.11.0-1016-nvidia`.

Building these modules does not install or load drivers on Kubernetes nodes.
Nodes must have compatible drivers provisioned separately. Image build checks
must verify `dkms status` for the target kernel and `dpkg --audit`; hardware RDMA
and NCCL checks are still required on the target nodes.

## License

Copyright 2025 Nebius B.V.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
