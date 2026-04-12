# to build the image
#docker build -t craftium-with-td3 .

# to run the container
#docker run --rm --name app craftium-with-td3

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

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
    libzstd-dev \
    make \
    pkg-config \
    wget \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY craftium /tmp/craftium
RUN cd /tmp/craftium \
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

RUN mkdir -p /app/results /app/logs

CMD ["python", "./main.py"]
