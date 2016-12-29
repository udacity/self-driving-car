FROM ubuntu:xenial
MAINTAINER pavel.cernocky@artin.cz
MAINTAINER tomas.cernik@artin.cz

ENV rosVersion kinetic

RUN apt-get update && \
  apt-get install -y git wget software-properties-common readline-common net-tools && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Add repositories
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
  wget http://packages.ros.org/ros.key -O - | apt-key add - && \
  add-apt-repository -y ppa:webupd8team/java && \
  add-apt-repository --yes ppa:xqms/opencv-nonfree && \
  apt-add-repository multiverse

# Install ROS
RUN apt-get update && \
  apt-get install -y ros-${rosVersion}-desktop-full \
    gazebo7 \
    ros-kinetic-lms1xx \
    libeigen3-dev \
    libgps-dev && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*


# wstool
RUN apt-get update && \
  apt-get install -y python-wstool && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV LANGUAGE=en_US:en

# set root password
RUN echo "root:robo.auto" | chpasswd

# create user robo
RUN useradd --create-home --shell=/bin/bash robo
RUN adduser robo video
RUN echo "robo:robo.auto" | chpasswd
RUN usermod -aG sudo robo
RUN echo "robo ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# set file system permitions
RUN chown -R robo:robo /home/robo

ENV HOME=/home/robo
ENV USER=robo
ENV DISPLAY=:0
ENV QT_GRAPHICSSYSTEM=native

RUN apt-get update && \
	apt-get install -y \
		terminator \
		vim \
		valgrind \
	        nano \
        	mc \
	        htop \
	        dstat && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN . /opt/ros/${rosVersion}/setup.sh &&\  
  rosdep init

USER robo

RUN . /opt/ros/${rosVersion}/setup.sh &&\  
  rosdep update

WORKDIR /home/robo
CMD /usr/bin/terminator
