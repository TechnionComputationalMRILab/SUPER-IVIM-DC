FROM rockylinux/rockylinux:8

# Install any necessary packages
RUN dnf update -y

# Install miniconda
RUN dnf install -y wget bzip2

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda
RUN rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# install the required packages
RUN conda create --name venv python=3.10
RUN echo "conda activate venv" >> ~/.bashrc
ENV PATH /opt/conda/envs/venv/bin:$PATH

# install the required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /code
COPY . /code/
RUN pip install .
