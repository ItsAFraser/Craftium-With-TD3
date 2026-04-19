# to build the image
#docker build -t craftium-with-td3 .

# to run the container
#docker run --rm --name app craftium-with-td3

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

ARG CRAFTIUM_REPO=https://github.com/mikelma/craftium.git
ARG CRAFTIUM_REF=main

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    git \
    g++ \
    gettext \
    libcurl4-gnutls-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libgmp-dev \
    libjpeg-dev \
    libjsoncpp-dev \
    libluajit-5.1-dev \
    libogg-dev \
    libopenal-dev \
    libpng-dev \
    libsqlite3-dev \
    libvorbis-dev \
    libx11-dev \
    libxext-dev \
    libzstd-dev \
    make \
    pkg-config \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${CRAFTIUM_REF}" --recurse-submodules "${CRAFTIUM_REPO}" /tmp/craftium \
    && cd /tmp/craftium \
    && if [ ! -f craftium-envs/minetest_game/game.conf ]; then \
        rm -rf craftium-envs/minetest_game \
        && git clone --depth 1 https://github.com/luanti-org/minetest_game.git craftium-envs/minetest_game; \
    fi \
    && bash build_sdl2.sh \
    && bash build_craftium.sh \
    && pip install --no-cache-dir . \
    && rm -rf /tmp/craftium

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py main.py
COPY experiment.py experiment.py
COPY gumbel_softmax.py gumbel_softmax.py
COPY sb3_train_td3.py sb3_train_td3.py
COPY td3_joint_policy.py td3_joint_policy.py
COPY td3_gumbel_policy.py td3_gumbel_policy.py

RUN mkdir -p /app/results /app/logs

#CMD ["python", "./sb3_train_td3.py", "--method", "td3", "--total-timesteps", "50_000", "--runs-dir", "./run-logs/td3-gumbel", "--run-name", "new_run", "--num-envs", "1", "--env-id", "Craftium/Speleo-v0"]
CMD ["python", "./experiment.py", "--runs-dir", "./run-logs/SpidersAttack-50k", "--env-id", "Craftium/SpidersAttack-v0"]
