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

check_path "${workspace}/X7s/ARX_CAN/arx_can"
check_path "${workspace}/X7s/body/ROS"
check_path "${workspace}/X7s/x7s/ROS/x7s_ws"
check_path "${workspace}/X7s_PLAY/realsense_camera"
check_path "${workspace}/X7s_PLAY/mobile_aloha"

check_executable "${workspace}/X7s/ARX_CAN/arx_can/arx_can1.sh"
check_executable "${workspace}/X7s/ARX_CAN/arx_can/arx_can3.sh"
check_executable "${workspace}/X7s/ARX_CAN/arx_can/arx_can5.sh"
check_executable "${workspace}/X7s_PLAY/realsense_camera/realsense.sh"

gnome-terminal --title="can1" -x bash -c "cd ${workspace}/X7s/ARX_CAN/arx_can; bash arx_can1.sh; exec bash"
sleep 1
gnome-terminal --title="can3" -x bash -c "cd ${workspace}/X7s/ARX_CAN/arx_can; bash arx_can3.sh; exec bash"
sleep 1
gnome-terminal --title="can5" -x bash -c "cd ${workspace}/X7s/ARX_CAN/arx_can; bash arx_can5.sh; exec bash"
sleep 1
gnome-terminal --title="body" -- bash -c "cd ${workspace}/X7s/body/ROS; source ./devel/setup.bash && roslaunch arx_lift_controller x7s.launch; exec bash"
sleep 1
gnome-terminal --title="left_arm" -- bash -c "cd ${workspace}/X7s/x7s/ROS/x7s_ws; source ./devel/setup.bash && roslaunch arx_x7_controller left_arm_inference.launch; exec bash"
sleep 1
gnome-terminal --title="right_arm" -- bash -c "cd ${workspace}/X7s/x7s/ROS/x7s_ws; source ./devel/setup.bash && roslaunch arx_x7_controller right_arm_inference.launch; exec bash"
sleep 1
gnome-terminal --title="realsense" -- bash -c "cd ${workspace}/X7s_PLAY/realsense_camera; bash realsense.sh; exec bash"
sleep 1
gnome-terminal --title="inference" -- bash -c "cd ${workspace}/X7s_PLAY/mobile_aloha; source ./venv/bin/activate; \
python inference.py --max_publish_step $max_publish_step --ckpt $ckpt_dir --ckpt_name $ckpt_name; exec bash"
sleep 1