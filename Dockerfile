# docker builds containers from top to bottom in these files
# docker caches every command, and only rebuilds if the command itself or
# one above it changes. this is why code is copied last

# official python image that we build on top of
FROM python:3.11-slim

# Ensure TLS root certificates are present and current
RUN apt-get update \
	&& apt-get install -y --no-install-recommends ca-certificates \
	&& update-ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

# copying first allows docker to cache this layer for quicker start times
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# we copy the code last since it is changed the most
COPY . .

# this commands runs in the directory given by WORKDIR earlier
CMD ["python", "./main.py"]