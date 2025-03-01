FROM --platform=linux/amd64 ubuntu:18.04

#
# Python3.8 install for Ubuntu
#
# ref: https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.8 python3-pip python3.8-dev
# Update symlink to point to latest
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3
RUN python3 --version
RUN pip3 --version

WORKDIR /fortap

COPY fortap/ .
RUN pip3 install setuptools==58 numpy==1.19.4 Cython==0.29.36 
RUN pip3 install torch==1.7.0 openpyxl==3.0.7 icecream==2.1.3 
RUN pip3 install torch_scatter==2.0.8 json-lines==0.5.0