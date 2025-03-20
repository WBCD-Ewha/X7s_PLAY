#!/bin/bash

workspace=$(dirname "$(pwd)")

clean="rm -rf build devel .catkin_workspace src/CMakeLists.txt"

check_path() {
    if [ ! -d "$1" ]; then
        echo "Directory not found: $1"

        exit 1
    fi
}

check_path "${workspace}/X7s/00-sh/ROS/"
check_path "${workspace}/X7s_PLAY/realsense_camera"
check_path "${workspace}/X7s_PLAY/mobile_aloha"

gnome-terminal --title="x7" -- bash -c "cd ${workspace}/X7s/00-sh/ROS/; bash 01make.sh; bash 02make.sh; exec bash"
sleep 1
gnome-terminal --title="realsense" -- bash -c "cd ${workspace}/X7s_PLAY/realsense_camera; $clean; catkin_make; exec bash"
sleep 1

if [ ! -d "${workspace}/X7s_PLAY/mobile_aloha/venv/" ];then
    gnome-terminal -t "venv" -x  bash -c "cd ${workspace}/X7s_PLAY/mobile_aloha; bash venv.sh; exec bash; "

else
    echo "python venv already exist"
fi
