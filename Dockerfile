# to build the image
#docker build -t craftium-manylinux .
#  -t tags the image with a name (craftium-manylinux)

# to run the container
#docker run --rm -v .:/app --name craft craftium-manylinux
#  --rm automatically removes the container when it exits (docker will freak out if you try to run it again without this since the container already exists)
#  -v mounts the current directory (.) to /app in the container, allowing access to files inside the container (how we view saved observations)
#  --name gives the container a name (craft) for easier reference (like checking logs with "docker logs craft")

FROM quay.io/pypa/manylinux_2_28_x86_64:latest

ENV PIP=/opt/python/cp312-cp312/bin/pip
ENV PYTHON=/opt/python/cp312-cp312/bin/python

WORKDIR /app

RUN yum install -y mesa-dri-drivers mesa-libEGL mesa-libGL

RUN ln -s $PYTHON /usr/local/bin/python && \
    ln -s $PIP /usr/local/bin/pip

RUN pip install --upgrade pip

RUN pip install https://github.com/mikelma/craftium/releases/download/v0.0.1/craftium-0.0.1-cp312-cp312-manylinux_2_28_x86_64.whl

# copy first to make use of Docker cache for pip install
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "./main.py"]
