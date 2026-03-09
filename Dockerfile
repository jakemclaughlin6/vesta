FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Core build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    # Ceres dependencies
    libceres-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    # Eigen
    libeigen3-dev \
    # Boost serialization
    libboost-serialization-dev \
    # SuiteSparse (CCOLAMD)
    libsuitesparse-dev \
    # Testing
    libgtest-dev \
    libgmock-dev \
    # Optional: benchmarks
    libbenchmark-dev \
    # Dev conveniences
    clang-format \
    clang-tidy \
    gdb \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
