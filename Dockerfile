# https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.10.2-cuda11.3-ubuntu20.04/Dockerfile
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# install ubuntu dependencies
ENV DEBIAN_FRONTEND=noninteractive 
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    xvfb \ 
    # https://stackoverflow.com/questions/32151043/xvfb-docker-cannot-open-display
    # build-essential \
    # python-opengl \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=$HOME/miniconda/bin:$PATH
COPY environment.yml /app/environment.yml
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda env update -n base -f /app/environment.yml \
 && rm /app/environment.yml \
 && conda clean -ya

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

# install python dependencies
# https://stackoverflow.com/questions/31528384/conditional-copy-add-in-dockerfile
COPY pyproject.toml poetry.loc[k] /app/
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi \
 && poetry install -E atari \
 && poetry install -E pybullet

# install mujoco
# RUN apt-get -y install wget unzip software-properties-common \
#     libgl1-mesa-dev \
#     libgl1-mesa-glx \
#     libglew-dev \
#     libosmesa6-dev patchelf
# RUN poetry install -E mujoco
# RUN poetry run python -c "import mujoco_py"

# https://stackoverflow.com/questions/32151043/xvfb-docker-cannot-open-display
# COPY entrypoint.sh /usr/local/bin/
# RUN chmod 777 /usr/local/bin/entrypoint.sh
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set the default command to python3
CMD ["python3"]
