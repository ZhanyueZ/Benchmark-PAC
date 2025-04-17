FROM ubuntu:20.04
RUN apt update && \
DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common && \
add-apt-repository ppa:deadsnakes/ppa && \
DEBIAN_FRONTEND=noninteractive apt install -y wget libedit-dev libxml2 python3.8-dev libelf-dev libnuma-dev python3-pip pkg-config python3.7-dev && \
pip3 install six && \
wget http://sdk-releases.upmem.com/2021.4.0/ubuntu_20.04/upmem-2021.4.0-Linux-x86_64.tar.gz && \
tar xf upmem-2021.4.0-Linux-x86_64.tar.gz && \
rm upmem-2021.4.0-Linux-x86_64.tar.gz && \
mv upmem-2021.4.0-Linux-x86_64/ upmem-sdk-2021.4.0
CMD bash -c "source /upmem-sdk-2021.4.0/upmem_env.sh simulator && bash"
