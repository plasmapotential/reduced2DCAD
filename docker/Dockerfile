#Dockerfile
#Engineer: T Looby
#Date: 06/28/2022
#Description:  Dockerfile for building reduced2DCAD docker image
#
#this script should only be run on dev machine
#script is not run directly, but is used when buildDocker bash
#script is run.  buildDocker script calls this file using syntax
#similar to this:
# docker build -t plasmapotential/reduced2dcad -f ./Dockerfile ./github/source

# start from base
FROM ubuntu:22.04

MAINTAINER Tom Looby <tlooby@cfs.energy>

# environment variables
ENV runMode docker
ENV AppDir /root
ENV APPDIR /root
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONPATH $PYTHONPATH:/usr/lib/freecad-python3/lib:/root/source/reduced2DCAD/github:/root/source/HEAT
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/netgen

# install system-wide deps for HEAT
RUN apt-get -yqq update
RUN apt-get -yqq install python3
RUN apt-get -yqq install python3-pip
RUN apt-get -yqq install python3-pkg-resources
RUN apt-get -yqq install python3-distutils
#RUN apt-get -yqq install libfreecad-python3-0.18
RUN apt-get -yqq install libfreecad-python3-0.19
RUN apt-get -yqq install libnglib-6.2
RUN apt-get -yqq install coreutils
RUN apt-get -yqq install nano
RUN apt-get -yqq install git

# copy context
WORKDIR /root/docker/
COPY . .

# fetch app specific deps
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN pip3 install dash --upgrade

# expose port
EXPOSE 8050

# start app
CMD [ "python3", "/root/source/reduced2DCAD/reducedCADlauncher.py", "--a", "0.0.0.0", "--p", "8050"]
