FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    gnupg \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 3.13
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
  apt-get update && \
  apt-get install -y python3.13 python3.13-venv python3.13-dev && \
  rm -rf /var/lib/apt/lists/*

# GHC
RUN apt-get update && apt-get install -y ghc cabal-install && \
  rm -rf /var/lib/apt/lists/*

# Golang
RUN add-apt-repository ppa:longsleep/golang-backports -y
RUN apt update
RUN apt install -y golang-go

# venv
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && \
  pip install .

FROM base AS test
RUN pip install -e .[dev]
CMD ["pytest", "-vvs"]

FROM base AS lango
ENTRYPOINT ["lango"]
