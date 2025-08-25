FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv

RUN apt-get update && apt-get install -y \
  software-properties-common \
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

# OCaml
RUN apt-get update && apt-get install -y ocaml opam && \
  rm -rf /var/lib/apt/lists/*

# GHC
RUN apt-get update && apt-get install -y ghc cabal-install && \
  rm -rf /var/lib/apt/lists/*

# Set default python3 to python3.13
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

WORKDIR /app
COPY . /app


RUN pip install --upgrade pip && \
  pip install .

ENTRYPOINT ["lango"]
