#!/bin/bash

workspace=$(dirname "$(pwd)")

max_publish_step=10000
ckpt_dir=weights
ckpt_name=policy_best.ckpt

check_path() {
    if [ ! -d "$1" ]; then
        echo "Directory not found: $1"

        exit 1
    fi
}

check_executable() {
    if [ ! -x "$1" ]; then
        echo "Script not executable: $1"
        exit 1
    fi
}

check_path "${workspace}/ARX_X7s/ARX_CAN"
check_path "${workspace}/ARX_X7s/X7s_Body_SDK"
check_path "${workspace}/ARX_X7s/ARX_X7s_SDK"
check_path "${workspace}/X7s_PLAY/realsense_camera"
check_path "${workspace}/X7s_PLAY/mobile_aloha"

check_executable "${workspace}/ARX_X7s/ARX_CAN/can.sh"
check_executable "${workspace}/ARX_X7s/X7s_Body_SDK/LIFT.sh"
check_executable "${workspace}/ARX_X7s/ARX_X7s_SDK/play.sh"
check_executable "${workspace}/X7s_PLAY/realsense_camera/realsense.sh"

gnome-terminal --title="can" -- bash -c "cd ${workspace}/ARX_X7s/ARX_CAN; bash can.sh; exec bash"
sleep 1
gnome-terminal --title="x7" -- bash -c "cd ${workspace}/ARX_X7s/ARX_X7s_SDK; bash play.sh; exec bash"
sleep 1
gnome-terminal --title="lift" -- bash -c "cd ${workspace}/ARX_X7s/X7s_Body_SDK; bash LIFT.sh; exec bash"
sleep 1
gnome-terminal --title="realsense" -- bash -c "cd ${workspace}/X7s_PLAY/realsense_camera; bash realsense.sh; exec bash"
sleep 1
gnome-terminal --title="inference" -- bash -c "cd ${workspace}/X7s_PLAY/mobile_aloha; source ./venv/bin/activate; \
python inference.py --max_publish_step $max_publish_step --ckpt $ckpt_dir --ckpt_name $ckpt_name; exec bash"
sleep 1