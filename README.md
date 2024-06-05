<h1 align="center">Spacemesh-cuda is a library for plot acceleration using CUDA-enabled GPUs.</h1>

[![license](https://img.shields.io/packagist/l/doctrine/orm.svg)](https://github.com/ZakuroLab/spacemesh-cuda/blob/master/LICENSE)
[![release](https://img.shields.io/github/v/release/ZakuroLab/spacemesh-cuda?include_prereleases)](https://github.com/ZakuroLab/spacemesh-cuda/releases)
![platform](https://img.shields.io/badge/platform-%20linux--64-lightgrey.svg)
[![open help wanted issues](https://img.shields.io/github/issues-raw/ZakuroLab/spacemesh-cuda/help%20wanted?logo=github)](https://github.com/ZakuroLab/spacemesh-cuda/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
[![made by](https://img.shields.io/badge/madeby-ZakuroLab-blue.svg)](https://github.com/ZakuroLab)

## <a name='TableofContents'></a>Table of Contents

* [1. Overview](#1-overview)
* [2. Performance](#2-performance)
* [3. Build & Integration Guide](#3-build-&-integration-guide)
  * [3.1 From source](#31-From-source)
  * [3.2 From binary](#32-From-binary)
* [4. License](#4-license)

## 1. Overview

__spacemesh-cuda__ is a cuda library for plot acceleration for [spacemesh](https://github.com/spacemeshos/go-spacemesh). 
This library optimizes memory access, calculation parallelism, etc. Compared with the official program, the library improved by **86.6%**.

## 2. Performance
| GPU\Library | Official | spacemesh-cuda |
| ---- |  --- | ---- |
| RTX3080 | 3.2MB/s | 5.97MB/s |

## 3. Build & Integration Guide

### 3.1 From source
```shell
# build libpost.so
git clone https://github.com/ZakuroLab/post-rs.git && cd post-rs/ffi
cargo build --release
cd ../../

# get postcli
wget https://github.com/spacemeshos/post/releases/download/v0.12.5/postcli-Linux.zip
unzip -d postcli ./postcli-Linux.zip
cd postcli && mv ../post-rs/target/release/libpost.so ./
```

### 3.2 From binary
```shell
mkdir postcli && cd postcli
wget https://github.com/ZakuroLab/spacemesh-cuda/releases/download/v0.0.1/libpost.so
wget https://github.com/ZakuroLab/spacemesh-cuda/releases/download/v0.0.1/postcli
```

## 4. License

We welcome all contributions to `spacemesh-cuda`. Please refer to the [license](#4-license) for the terms of contributions.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE.md)
