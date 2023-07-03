#Our image will start on tensorflow with GPU support
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
#Updating packages
RUN apt update
#Installing APT utils
RUN apt install -y apt-utils
# Installing NANO
RUN apt install -y nano
#Installing CMAKE
RUN apt install -y cmake
#Updating pip
RUN pip install --upgrade pip
RUN pip install 'gudhi~=3.5.0'
RUN pip install 'numpy~=1.22.4'
RUN pip install 'imageio~=2.19.5'
RUN pip install 'matplotlib~=3.5.2'
RUN pip install 'seaborn~=0.12.2'
RUN pip install 'mediapy~=1.0.3'
RUN pip install 'tabulate~=0.8.10'
RUN pip install 'giotto-ph~=0.2.2''
RUN pip install scikit-posthocs
RUN pip install networkx

## added:
