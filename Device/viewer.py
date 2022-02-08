import os
import socket
import struct
import subprocess
from queue import Queue
from time import sleep

import av
import numpy as np
from loguru import logger

from .control import ControlMixin


class AndroidViewer(ControlMixin):
    video_socket = None
    is_video_socket_close = False
    control_socket = None
    resolution = None
    Subp = [None]

    received = 0
    video_data_queue = Queue()

    def __init__(self, max_width=3840, bitrate=8000000, max_fps=8, adb_path='', ip='127.0.0.1', port=8081, multipleDevices=False, deviceADBName=None):
        self.ip = ip
        self.port = port

        self.device_name = ""

        self.adb_path = adb_path

        self.multipleDevices = multipleDevices
        if multipleDevices:
            self.deviceADBName = deviceADBName

        assert self.deploy_server(max_width, bitrate, max_fps)

        self.codec = av.codec.CodecContext.create('h264', 'r')

        self.init_server_connection()

    def receiver(self):
        while True:

            raw_h264 = self.video_socket.recv(0x10000)

            if not raw_h264:
                continue

            self.video_data_queue.put(raw_h264)

    def init_server_connection(self):
        logger.info("连接视频流套接字...")
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_socket.connect((self.ip, self.port))

        dummy_byte = self.video_socket.recv(1)
        if not len(dummy_byte):
            raise ConnectionError("Did not receive Dummy Byte!")

        logger.info("连接控制套接字")

        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.connect((self.ip, self.port))

        device_name = self.video_socket.recv(64).decode("utf-8")

        if not len(device_name):
            raise ConnectionError("未能成功获取设备名!")
        logger.info("设备名: " + device_name)
        self.device_name = device_name

        res = self.video_socket.recv(4)
        self.resolution = struct.unpack(">HH", res)
        logger.info("屏幕分辨率: " + str(self.resolution))
        self.video_socket.setblocking(0)

    def deploy_server(self, max_width=1024, bitrate=8000000, max_fps=10):
        try:
            logger.info("Upload JAR...")

            server_root = os.path.abspath(os.path.dirname(__file__))
            server_file_path = server_root + '/scrcpy-server.jar'
            if not os.path.exists(server_file_path):
                logger.error(
                    "scrcpy-server.jar 丢失, 请检查目录:" + server_file_path)
                raise FileNotFoundError(
                    "crcpy-server.jar 丢失, 请检查目录:" + str(server_file_path))
            device_args = []
            if self.multipleDevices:
                device_args = ["-s", self.deviceADBName]

            adb_push = subprocess.Popen([self.adb_path] + device_args + ['push', server_file_path, '/data/local/tmp/'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=server_root)
            adb_push_comm = ''.join([x.decode("utf-8") for x in adb_push.communicate() if x is not None])
            if "error" in adb_push_comm:
                logger.critical("Is your device/emulator visible to ADB?")
                raise Exception(adb_push_comm)

            logger.info("开始运行服务端...")
            self.Subp[0] = subprocess.Popen([self.adb_path] + device_args +
                                            ['shell', 'CLASSPATH=/data/local/tmp/scrcpy-server.jar', 'app_process', '/',
                                             'com.genymobile.scrcpy.Server 1.12.1 {} {} {} true - false true'.format(max_width, bitrate, max_fps)],
                                            cwd=server_root)
            logger.info("转发端口...")
            subprocess.Popen([self.adb_path] + device_args + ['forward', 'tcp:8081', 'localabstract:scrcpy'], cwd=server_root).wait()
            sleep(2)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Couldn't find ADB or jar at path ADB_bin: " + str(self.adb_path))

        return True

    def get_next_frames(self) -> np.ndarray:
        packets = []
        try:
            raw_h264 = self.video_socket.recv(0x10000)
            # sleep(0.048)
            if raw_h264 == b'':
                self.is_video_socket_close = True
                return None
            self.received = self.received + len(raw_h264)
            packets = self.codec.parse(raw_h264)

            if not packets:
                return None

        except socket.error as e:
            return None

        if not packets:
            return None

        result_frames = []

        for packet in packets:
            frames = self.codec.decode(packet)
            for frame in frames:
                result_frames.append(frame.to_ndarray(format="rgb24"))

        return result_frames or None

    def Dispose(self):
        try:
            self.video_socket.shutdown(socket.SHUT_RDWR)
            self.control_socket.shutdown(socket.SHUT_RDWR)
            self.video_socket.close()
            self.control_socket.close()
        except Exception as e:
            logger.warning("A error happend at close socket: " + str(e))
        server_root = os.path.abspath(os.path.dirname(__file__))
        subprocess.Popen([self.adb_path, 'forward', '--remove', 'tcp:8081'], cwd=server_root).wait()
        self.Subp[0].kill()
