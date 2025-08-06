# Use a base image with essential build tools
FROM ubuntu:22.04

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for ns-3, Python, and other tools
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    python3 \
    python3-dev \
    python3-pip \
    cmake \
    git \
    curl \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy and install Python requirements
COPY scratch/requirements.txt scratch/requirements.txt
RUN pip3 install --no-cache-dir -r scratch/requirements.txt

#
# REMOVED: The following steps are no longer needed here.
# The build will be triggered from docker-compose using host volumes.
#
# COPY . .
# RUN ./ns3 configure
# RUN ./ns3 build
#

# The default command to run when the container starts
CMD ["bash"]