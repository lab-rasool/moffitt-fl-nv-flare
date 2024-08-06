# moffitt-fl-nv-flare
## This repository contains federated learning experiments with data from the National Lung Cancer Screening Trials as well as insallation instrucitons for NVFLARE

# Introduction
This document will include various usages for NVFLARE, and will be updated as knowledge on the flare environment is acquired. It will cover installation, setup, CLI usage, and more. This document is designed to ease the learning curve and speed up deployments. The official docs and GitHub repo for NVFLARE can be found at [this link](https://developer.nvidia.com/flare).

# Installation
NVFLARE installs using pip. It is recommended to set up a virtual environment.
```sh
pip install nvflare

If an error is thrown, upgrade pip and setuptools and retry:

pip install --upgrade pip
pip install --upgrade setuptools

nvflare --help
nvflare --version

This document is using version 2.4.0. For additional information, refer to the docs here.

