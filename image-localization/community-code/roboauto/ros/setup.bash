#!/bin/bash
# Usage: source setup.bash
# Author: Vojtech Smejkal

# Sample blacklist
# BLACKLIST=`cat <<EOT | sed -e 's/[\t ]*//g'
#     gamepad_controller;
# EOT`

ROS_SETUP=/opt/ros/kinetic/setup.bash
ROSJAVA_SETUP=/opt/rosjava/devel/setup.bash
LOCAL_SETUP=$(dirname ${BASH_SOURCE[0]})/devel/setup.bash

function __catkin_make() {
    BUILDTYPE=$1
    shift

    catkin_make -DCATKIN_BLACKLIST_PACKAGES=$BLACKLIST -DCMAKE_BUILD_TYPE=$BUILDTYPE $@

    if [ -e $LOCAL_SETUP ]; then
        source $LOCAL_SETUP
    fi
}

alias catkin_make_debug="__catkin_make Debug"
alias catkin_make_release="__catkin_make Release"
alias catkin_make_release_debug="__catkin_make RelWithDebInfo"

if [ -e $LOCAL_SETUP ]; then
    source $LOCAL_SETUP
else
    source $ROS_SETUP
fi
