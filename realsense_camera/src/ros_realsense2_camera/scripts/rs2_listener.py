#!/usr/bin/env python3
import sys
import time
import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CompressedImage as msg_CompressedImage
from sensor_msgs.msg import PointCloud2 as msg_PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Imu as msg_Imu
from sensor_msgs.msg import CameraInfo as msg_CameraInfo
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
import inspect
import ctypes
import struct
import open3d as o3d
from datetime import datetime
import tf
import imageio.v2 as imageio
import json
from pathlib import Path


try:
    from theora_image_transport.msg import Packet as msg_theora
except Exception:
    pass


def pc2_to_xyzrgb(point):
	# Thanks to Panos for his code used in this function.
    x, y, z = point[:3]
    rgb = point[3]

    # cast float32 to int so that bitwise operations are possible
    s = struct.pack('>f', rgb)
    i = struct.unpack('>l', s)[0]
    # you can get back the float value by the inverse operations
    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)
    return x, y, z, r, g, b


class CWaitForMessage:
    def __init__(self, params={}):
        self.result = None

        self.break_timeout = False
        self.timeout = params.get('timeout_secs', -1) * 1e-3
        self.seq = params.get('seq', -1)
        self.time = params.get('time', None)
        self.node_name = params.get('node_name', 'rs2_listener')
        self.bridge = CvBridge()
        self.listener = None
        self.prev_msg_time = 0
        self.fout = None
        self.camera = params.get('camera_name')
        self.file_dir = params.get('file_directory')
        print(self.camera)

        self.themes = {'depthStream': {'topic': f'/{self.camera}/depth/image_rect_raw', 'callback': self.imageDepthCallback, 'msg_type': msg_Image},
                       'colorStream': {'topic': f'/{self.camera}/color/image_raw', 'callback': self.imageColorCallback, 'msg_type': msg_Image},
                       'pointscloud': {'topic': f'/{self.camera}/depth/color/points', 'callback': self.pointscloudCallback, 'msg_type': msg_PointCloud2},
                       'alignedDepthInfra1': {'topic': f'/{self.camera}/aligned_depth_to_infra1/image_raw', 'callback': self.imageColorCallback, 'msg_type': msg_Image},
                       'alignedDepthColor': {'topic': f'/{self.camera}/aligned_depth_to_color/image_raw', 'callback': self.imageColorCallback, 'msg_type': msg_Image},
                       'static_tf': {'topic': f'/{self.camera}/color/image_raw', 'callback': self.imageColorCallback, 'msg_type': msg_Image},
                       'accelStream': {'topic': f'/{self.camera}/accel/sample', 'callback': self.imuCallback, 'msg_type': msg_Imu},
                       'cameraInfo': {'topic': f'/{self.camera}/color/camera_info','callback': self.cameraInfoCallback,'msg_type': msg_CameraInfo,}
                       }

        self.func_data = dict()

    def cameraInfoCallback(self, theme_name):
        def _cameraInfoCallback(data: msg_CameraInfo):
            if self.func_data[theme_name].get('saved_once'):
                return

            # Intrinsic 파라미터 추출
            intr_dict = {
                "image_resolution": {
                    "width": data.width,
                    "height": data.height
                },
                "focal_lengths_in_pixels": {
                    "fx": data.K[0],
                    "fy": data.K[4]
                },
                "principal_point_in_pixels": {
                    "cx": data.K[2],
                    "cy": data.K[5]
                }
            }

            # 저장 경로
            json_path = Path(self.file_dir) / "camera_intrinsics.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)

            with open(json_path, "w") as f:
                json.dump(intr_dict, f, indent=4)

            rospy.loginfo("Saved camera intrinsics to %s", json_path)

            # 플래그 & 구독 해제
            self.func_data[theme_name]['saved_once'] = True

        return _cameraInfoCallback

    def imuCallback(self, theme_name):
        def _imuCallback(data):
            if self.listener is None:
                self.listener = tf.TransformListener()
            self.prev_time = time.time()
            self.func_data[theme_name].setdefault('value', [])
            self.func_data[theme_name].setdefault('ros_value', [])
            try:
                frame_id = data.header.frame_id
                value = data.linear_acceleration

                (trans,rot) = self.listener.lookupTransform('/camera_link', frame_id, rospy.Time(0))
                quat = tf.transformations.quaternion_matrix(rot)
                point = np.matrix([value.x, value.y, value.z, 1], dtype='float32')
                point.resize((4, 1))
                rotated = quat*point
                rotated.resize(1,4)
                rotated = np.array(rotated)[0][:3]
            except Exception as e:
                print(e)
                return
            self.func_data[theme_name]['value'].append(value)
            self.func_data[theme_name]['ros_value'].append(rotated)
        return _imuCallback            

    def imageColorCallback(self, theme_name):
        def _imageColorCallback(data):
            if self.func_data[theme_name].get('saved_once'):
                return

            self.prev_time = time.time()
            self.func_data[theme_name].setdefault('avg', [])
            self.func_data[theme_name].setdefault('ok_percent', [])
            self.func_data[theme_name].setdefault('num_channels', [])
            self.func_data[theme_name].setdefault('shape', [])
            self.func_data[theme_name].setdefault('reported_size', [])

            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            except CvBridgeError as e:
                print(e)
                return
            channels = cv_image.shape[2] if len(cv_image.shape) > 2 else 1
            pyimg = np.asarray(cv_image)

            ok_number = (pyimg != 0).sum()

            self.func_data[theme_name]['avg'].append(pyimg.sum() / ok_number)
            self.func_data[theme_name]['ok_percent'].append(float(ok_number) / (pyimg.shape[0] * pyimg.shape[1]) / channels)
            self.func_data[theme_name]['num_channels'].append(channels)
            self.func_data[theme_name]['shape'].append(cv_image.shape)
            self.func_data[theme_name]['reported_size'].append((data.width, data.height, data.step))

            ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
            npimage_filename = f"{self.file_dir}/image_left.png"
            cvimage_filename = f"{self.file_dir}/image_left_cv.png"

            if data.encoding.lower().startswith("bgr"):
                rgb_uint8 = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_uint8 = cv_image.copy()

            imageio.imwrite(npimage_filename, rgb_uint8)

            # cv_image 는 data.encoding 에 따라 RGB 또는 BGR
            if data.encoding.lower().startswith("rgb"):
                # RGB → BGR 변환 후 OpenCV 저장
                bgr_for_cv = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            else:  # 이미 BGR
                bgr_for_cv = cv_image
            cv2.imwrite(cvimage_filename, bgr_for_cv)

            rospy.loginfo("Saved RGB images to %s  (NumPy)  and  %s  (OpenCV)",
                          npimage_filename, cvimage_filename)

            self.func_data[theme_name]['saved_once'] = True

        return _imageColorCallback

    def imageDepthCallback(self, theme_name):
        def _imageDepthCallback(data):
            if self.func_data[theme_name].get('saved_once'):
                return

            self.prev_time = time.time()
            self.func_data[theme_name].setdefault('avg', [])
            self.func_data[theme_name].setdefault('ok_percent', [])
            self.func_data[theme_name].setdefault('num_channels', [])
            self.func_data[theme_name].setdefault('shape', [])
            self.func_data[theme_name].setdefault('reported_size', [])

            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            except CvBridgeError as e:
                print(e)
                return
            channels = cv_image.shape[2] if len(cv_image.shape) > 2 else 1
            pyimg = np.asarray(cv_image)

            ok_number = (pyimg != 0).sum()

            self.func_data[theme_name]['avg'].append(pyimg.sum() / ok_number)
            self.func_data[theme_name]['ok_percent'].append(
                float(ok_number) / (pyimg.shape[0] * pyimg.shape[1]) / channels)
            self.func_data[theme_name]['num_channels'].append(channels)
            self.func_data[theme_name]['shape'].append(cv_image.shape)
            self.func_data[theme_name]['reported_size'].append((data.width, data.height, data.step))

            ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
            npimage_filename = f"{self.file_dir}/depth_image.jpg"
            cvimage_filename = f"{self.file_dir}/depth_image_cv.jpg"
            raw16_filename = f"{self.file_dir}/depth_image_raw16.png"  # 추가: 16bit 저장 파일 이름

            # 1) 16‑bit/float depth → 8‑bit 그레이스케일
            depth_uint8 = cv2.normalize(pyimg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            imageio.imwrite(npimage_filename, depth_uint8)
            cv2.imwrite(cvimage_filename, depth_uint8)

            # 2) 원본 16bit 그대로 저장 (추가)
            if pyimg.dtype == np.float32 or pyimg.dtype == np.float64:
                # float 타입이면 16bit integer로 변환해서 저장
                pyimg_to_save = (pyimg * 1000).astype(np.uint16)  # 보통 meter -> mm 변환
            elif pyimg.dtype == np.uint16:
                # 이미 16bit integer면 그대로 저장
                pyimg_to_save = pyimg
            else:
                # 다른 타입이면 에러
                rospy.logwarn("Unexpected depth image dtype: {}".format(pyimg.dtype))
                pyimg_to_save = pyimg.astype(np.uint16)

            # 16bit depth 저장 (PNG가 16bit를 지원함)
            cv2.imwrite(raw16_filename, pyimg_to_save)

            rospy.loginfo("Saved depth images to %s  (NumPy)  and  %s  (OpenCV) and %s (Raw 16bit)",
                          npimage_filename, cvimage_filename, raw16_filename)

            self.func_data[theme_name]['saved_once'] = True

        return _imageDepthCallback


    def pointscloudCallback(self, theme_name):
        def _pointscloudCallback(data):
            if self.func_data[theme_name].get('saved_once'):
                return
            self.prev_time = time.time()
            print ('Got pointcloud: %d, %d' % (data.width, data.height))

            self.func_data[theme_name].setdefault('frame_counter', 0)
            self.func_data[theme_name].setdefault('avg', [])
            self.func_data[theme_name].setdefault('size', [])
            self.func_data[theme_name].setdefault('width', [])
            self.func_data[theme_name].setdefault('height', [])
            # until parsing pointcloud is done in real time, I'll use only the first frame.
            self.func_data[theme_name]['frame_counter'] += 1

            if self.func_data[theme_name]['frame_counter'] == 1:
                # Known issue - 1st pointcloud published has invalid texture. Skip 1st frame.
                return

            try:
                points = np.array([pc2_to_xyzrgb(pp) for pp in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z", "rgb"))])
            except Exception as e:
                print(e)
                return
            self.func_data[theme_name]['avg'].append(points.mean(0))
            self.func_data[theme_name]['size'].append(len(points))
            self.func_data[theme_name]['width'].append(data.width)
            self.func_data[theme_name]['height'].append(data.height)

            # save pointcloud in .ply file using open3d
            # points 배열: [x, y, z, r, g, b]  ― r g b는 0‑255
            xyz = points[:, :3]
            rgb = points[:, 3:] / 255.0  # Open3D는 0‑1 실수 색상

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)

            # 고유한 파일이름(테마 이름 + 타임스탬프)
            ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")[:-3]
            output_ply_path = f"{self.file_dir}/point_cloud.ply"

            o3d.io.write_point_cloud(output_ply_path, pcd, write_ascii=True)
            rospy.loginfo("Saved point cloud to %s (%d points)", output_ply_path, len(pcd.points))
            self.func_data[theme_name]['saved_once'] = True

        return _pointscloudCallback

    def wait_for_message(self, params, msg_type=msg_Image):
        topic = params['topic']
        print ('connect to ROS with name: %s' % self.node_name)
        rospy.init_node(self.node_name, anonymous=True)

        out_filename = params.get('filename', None)
        if (out_filename):
            self.fout = open(out_filename, 'w')
            if msg_type is msg_Imu:
                col_w = 20
                print ('Writing to file: %s' % out_filename)
                columns = ['frame_number', 'frame_time(sec)', 'accel.x', 'accel.y', 'accel.z', 'gyro.x', 'gyro.y', 'gyro.z']
                line = ('{:<%d}'*len(columns) % (col_w, col_w, col_w, col_w, col_w, col_w, col_w, col_w)).format(*columns) + '\n'
                sys.stdout.write(line)
                self.fout.write(line)

        rospy.loginfo('Subscribing on topic: %s' % topic)
        self.sub = rospy.Subscriber(topic, msg_type, self.callback)

        self.prev_time = time.time()
        break_timeout = False
        while not any([rospy.core.is_shutdown(), break_timeout, self.result]):
            rospy.rostime.wallsleep(0.5)
            if self.timeout > 0 and time.time() - self.prev_time > self.timeout:
                break_timeout = True
                self.sub.unregister()

        return self.result

    @staticmethod
    def unregister_all(registers):
        for test_name in registers:
            rospy.loginfo('Un-Subscribing test %s' % test_name)
            registers[test_name]['sub'].unregister()

    def wait_for_messages(self, themes):
        # tests_params = {<name>: {'callback', 'topic', 'msg_type', 'internal_params'}}
        self.func_data = dict([[theme_name, {}] for theme_name in themes])

        print ('connect to ROS with name: %s' % self.node_name)
        rospy.init_node(self.node_name, anonymous=True)
        for theme_name in themes:
            theme = self.themes[theme_name]
            rospy.loginfo('Subscribing %s on topic: %s' % (theme_name, theme['topic']))
            self.func_data[theme_name]['sub'] = rospy.Subscriber(theme['topic'], theme['msg_type'], theme['callback'](theme_name))

        self.prev_time = time.time()
        break_timeout = False
        while not any([rospy.core.is_shutdown(), break_timeout]):
            rospy.rostime.wallsleep(0.5)
            if self.timeout > 0 and time.time() - self.prev_time > self.timeout:
                break_timeout = True
                self.unregister_all(self.func_data)

        return self.func_data

    def callback(self, data):
        msg_time = data.header.stamp.secs + 1e-9 * data.header.stamp.nsecs

        if (self.prev_msg_time > msg_time):
            rospy.loginfo('Out of order: %.9f > %.9f' % (self.prev_msg_time, msg_time))
        if type(data) == msg_Imu:
            col_w = 20
            frame_number = data.header.seq
            accel = data.linear_acceleration
            gyro = data.angular_velocity
            line = ('\n{:<%d}{:<%d.6f}{:<%d.4f}{:<%d.4f}{:<%d.4f}{:<%d.4f}{:<%d.4f}{:<%d.4f}' % (col_w, col_w, col_w, col_w, col_w, col_w, col_w, col_w)).format(frame_number, msg_time, accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z)
            sys.stdout.write(line)
            if self.fout:
                self.fout.write(line)

        self.prev_msg_time = msg_time
        self.prev_msg_data = data

        self.prev_time = time.time()
        if any([self.seq < 0 and self.time is None, 
                self.seq > 0 and data.header.seq >= self.seq,
                self.time and data.header.stamp.secs == self.time['secs'] and data.header.stamp.nsecs == self.time['nsecs']]):
            self.result = data
            self.sub.unregister()



def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '/?' in sys.argv:
        print ('USAGE:')
        print ('------')
        print ('rs2_listener.py <topic | theme> [Options]')
        print ('example: rs2_listener.py /camera/color/image_raw --time 1532423022.044515610 --timeout 3')
        print ('example: rs2_listener.py pointscloud')
        print ('')
        print ('Application subscribes on <topic>, wait for the first message matching [Options].')
        print ('When found, prints the timestamp.')
        print
        print ('[Options:]')
        print ('--camera_name <camera_name>: choose among camera_h, camera_r, camera_l')
        print ('--file_directory')
        print ('-s <sequential number>')
        print ('--time <secs.nsecs>')
        print ('--timeout <secs>')
        print ('--filename <filename> : write output to file')
        exit(-1)

    # wanted_topic = '/device_0/sensor_0/Depth_0/image/data'
    # wanted_seq = 58250

    wanted_topic = sys.argv[1]
    msg_params = {}
    if 'points' in wanted_topic:
        msg_type = msg_PointCloud2
    elif ('imu' in wanted_topic) or ('gyro' in wanted_topic) or ('accel' in wanted_topic):
        msg_type = msg_Imu
    elif 'theora' in wanted_topic:
        try:
            msg_type = msg_theora
        except NameError as e:
            print ('theora_image_transport is not installed. \nType "sudo apt-get install ros-kinetic-theora-image-transport" to enable registering on messages of type theora.')
            raise
    elif 'compressed' in wanted_topic:
        msg_type = msg_CompressedImage
    else:
        msg_type = msg_Image

    for idx in range(2, len(sys.argv)):
        if sys.argv[idx] == '--camera_name':
            msg_params['camera_name'] = sys.argv[idx + 1]
        if sys.argv[idx] == '--file_directory':
            msg_params['file_directory'] = sys.argv[idx + 1]
        if sys.argv[idx] == '-s':
            msg_params['seq'] = int(sys.argv[idx + 1])
        if sys.argv[idx] == '--time':
            msg_params['time'] = dict(zip(['secs', 'nsecs'], [int(part) for part in sys.argv[idx + 1].split('.')]))
        if sys.argv[idx] == '--timeout':
            msg_params['timeout_secs'] = int(sys.argv[idx + 1])
        if sys.argv[idx] == '--filename':
            msg_params['filename'] = sys.argv[idx+1]

    msg_retriever = CWaitForMessage(msg_params)
    if '/' in wanted_topic:
        msg_params.setdefault('topic', wanted_topic)
        res = msg_retriever.wait_for_message(msg_params, msg_type)
        rospy.loginfo('Got message: %s' % res.header)
        if (hasattr(res, 'encoding')):
            print ('res.encoding:', res.encoding)
        if (hasattr(res, 'format')):
            print ('res.format:', res.format)
    else:
        themes = [wanted_topic]
        res = msg_retriever.wait_for_messages(themes)
        print (res)

def visualize_pointcloud(ply_path: str) -> None:
    print(f"Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        sys.exit("Loaded point cloud is empty or file not found.")

    # 알아보기 쉬운 좌표축 추가(선택사항)
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # 카메라(뷰어) 파라미터
    lookat = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    front = np.array([0.0, -1.0, 0.0])  # 뒤에서 앞을 보도록 설정 예시
    zoom = 0.5

    # 시각화
    o3d.visualization.draw_geometries(
        [pcd, coord],
        window_name="Open3D – Point Cloud Viewer",
        width=1280,
        height=720,
        mesh_show_back_face=True,
        lookat=lookat,
        up=up,
        front=front,
        zoom=zoom,
    )


if __name__ == '__main__':
    main()

    # visualize_pointcloud("./pointscloud_pointcloud_20250419T184513567.ply")

