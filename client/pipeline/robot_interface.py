"""
ロボットへの把持姿勢送信インターフェース

生成された把持姿勢 (23関節座標) をロボットに送信し、
実際に把持動作を実行させるためのモジュール。

対応プロトコル:
    - ROSトピック (ROS1 / ROS2)
    - TCP/IPソケット通信
    - シリアル通信

使用例:
    interface = RobotInterface(mode="ros", topic="/grasp_pose")
    interface.connect()
    interface.send_grasp_pose(left_hand, right_hand, object_scale=1.0)
    result = interface.wait_for_result()
    interface.disconnect()
"""

import json
import time
import numpy as np
from typing import Optional, Tuple


class GraspPose:
    """
    把持姿勢データクラス

    Attributes:
        left_hand:    (23, 3) 左手関節座標 [正規化座標系]
        right_hand:   (23, 3) 右手関節座標 [正規化座標系]
        object_scale: 物体の実スケール [メートル] (座標変換に使用)
        object_center: 物体中心の実座標 [メートル]
    """

    def __init__(
        self,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        object_scale: float = 1.0,
        object_center: Optional[np.ndarray] = None,
    ):
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.object_scale = object_scale
        self.object_center = object_center if object_center is not None else np.zeros(3)

    def to_world_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        正規化座標系から実世界座標系に変換する

        Returns:
            left_world:  (23, 3) 左手関節座標 [メートル]
            right_world: (23, 3) 右手関節座標 [メートル]
        """
        left_world = self.left_hand * self.object_scale + self.object_center
        right_world = self.right_hand * self.object_scale + self.object_center
        return left_world, right_world

    def to_dict(self) -> dict:
        """JSON送信用の辞書形式に変換"""
        left_world, right_world = self.to_world_coordinates()
        return {
            "left_hand": left_world.tolist(),
            "right_hand": right_world.tolist(),
            "object_scale": float(self.object_scale),
            "object_center": self.object_center.tolist(),
        }


class RobotInterface:
    """
    ロボット制御インターフェース基底クラス

    実際のロボット環境に合わせて、以下のサブクラスを使用:
        - ROSRobotInterface:    ROS (Robot Operating System) 経由
        - TCPRobotInterface:    TCP/IPソケット通信
        - SerialRobotInterface: シリアル通信
    """

    def __init__(self, mode: str = "tcp", **kwargs):
        """
        Args:
            mode: 通信モード ("ros", "tcp", "serial", "mock")
            **kwargs: 各モード固有の設定パラメータ
        """
        self.mode = mode
        self.kwargs = kwargs
        self._connected = False

        mode_map = {
            "ros": ROSRobotInterface,
            "tcp": TCPRobotInterface,
            "serial": SerialRobotInterface,
            "mock": MockRobotInterface,
        }

        if mode not in mode_map:
            raise ValueError(f"未対応のモード: {mode}. 対応モード: {list(mode_map.keys())}")

        self._impl = mode_map[mode](**kwargs)

    def connect(self):
        """ロボットに接続する"""
        self._impl.connect()
        self._connected = True
        print(f"[RobotInterface] 接続完了 (mode={self.mode})")

    def send_grasp_pose(
        self,
        grasp_pose: GraspPose,
        execute: bool = True,
    ) -> bool:
        """
        把持姿勢をロボットに送信する

        Args:
            grasp_pose: GraspPose オブジェクト
            execute:    True の場合、把持動作を実行させる

        Returns:
            送信成功: True
        """
        if not self._connected:
            raise RuntimeError("ロボットに接続されていません。connect() を先に呼んでください。")

        pose_dict = grasp_pose.to_dict()
        pose_dict["execute"] = execute

        print(f"[RobotInterface] 把持姿勢を送信中...")
        print(f"  左手首位置: {grasp_pose.left_hand[0].tolist()}")
        print(f"  右手首位置: {grasp_pose.right_hand[0].tolist()}")

        return self._impl.send(pose_dict)

    def wait_for_result(self, timeout: float = 30.0) -> dict:
        """
        ロボットの動作完了を待ち、結果を受け取る

        Args:
            timeout: タイムアウト秒数

        Returns:
            結果辞書 {"success": bool, "message": str}
        """
        return self._impl.wait_for_result(timeout)

    def disconnect(self):
        """ロボットから切断する"""
        if self._connected:
            self._impl.disconnect()
            self._connected = False
            print("[RobotInterface] 切断完了")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class ROSRobotInterface:
    """
    ROS (Robot Operating System) 経由のロボット制御

    必要: rospy または rclpy (ROS1/ROS2)
    """

    def __init__(
        self,
        topic: str = "/grasp_pose",
        result_topic: str = "/grasp_result",
        ros_version: int = 1,
    ):
        self.topic = topic
        self.result_topic = result_topic
        self.ros_version = ros_version
        self._publisher = None
        self._result = None

    def connect(self):
        try:
            if self.ros_version == 1:
                import rospy
                from std_msgs.msg import String
                rospy.init_node("real_world_demo", anonymous=True)
                self._publisher = rospy.Publisher(self.topic, String, queue_size=1)
                rospy.Subscriber(self.result_topic, String, self._result_callback)
            else:
                import rclpy
                from std_msgs.msg import String
                rclpy.init()
                self._node = rclpy.create_node("real_world_demo")
                self._publisher = self._node.create_publisher(String, self.topic, 1)
        except ImportError as e:
            raise ImportError(f"ROSライブラリが見つかりません: {e}")

    def _result_callback(self, msg):
        self._result = json.loads(msg.data)

    def send(self, pose_dict: dict) -> bool:
        if self.ros_version == 1:
            from std_msgs.msg import String
            msg = String()
            msg.data = json.dumps(pose_dict)
            self._publisher.publish(msg)
        else:
            from std_msgs.msg import String
            msg = String()
            msg.data = json.dumps(pose_dict)
            self._publisher.publish(msg)
        return True

    def wait_for_result(self, timeout: float = 30.0) -> dict:
        start = time.time()
        while self._result is None and (time.time() - start) < timeout:
            time.sleep(0.1)
        if self._result is None:
            return {"success": False, "message": "タイムアウト"}
        result = self._result
        self._result = None
        return result

    def disconnect(self):
        pass


class TCPRobotInterface:
    """
    TCP/IPソケット通信によるロボット制御
    """

    def __init__(self, host: str = "192.168.1.100", port: int = 5005):
        self.host = host
        self.port = port
        self._socket = None

    def connect(self):
        import socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self.host, self.port))
        self._socket.settimeout(30.0)
        print(f"[TCP] 接続: {self.host}:{self.port}")

    def send(self, pose_dict: dict) -> bool:
        data = json.dumps(pose_dict).encode("utf-8")
        length = len(data).to_bytes(4, byteorder="big")
        self._socket.sendall(length + data)
        return True

    def wait_for_result(self, timeout: float = 30.0) -> dict:
        self._socket.settimeout(timeout)
        try:
            length_bytes = self._socket.recv(4)
            length = int.from_bytes(length_bytes, byteorder="big")
            data = b""
            while len(data) < length:
                chunk = self._socket.recv(min(4096, length - len(data)))
                if not chunk:
                    break
                data += chunk
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            return {"success": False, "message": str(e)}

    def disconnect(self):
        if self._socket:
            self._socket.close()
            self._socket = None


class SerialRobotInterface:
    """
    シリアル通信によるロボット制御
    """

    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._serial = None

    def connect(self):
        try:
            import serial
            self._serial = serial.Serial(self.port, self.baudrate, timeout=30.0)
            time.sleep(2.0)  # デバイス初期化待ち
            print(f"[Serial] 接続: {self.port} @ {self.baudrate}bps")
        except ImportError:
            raise ImportError("pyserial が見つかりません。pip install pyserial")

    def send(self, pose_dict: dict) -> bool:
        data = json.dumps(pose_dict) + "\n"
        self._serial.write(data.encode("utf-8"))
        return True

    def wait_for_result(self, timeout: float = 30.0) -> dict:
        self._serial.timeout = timeout
        try:
            line = self._serial.readline().decode("utf-8").strip()
            return json.loads(line)
        except Exception as e:
            return {"success": False, "message": str(e)}

    def disconnect(self):
        if self._serial and self._serial.is_open:
            self._serial.close()


class MockRobotInterface:
    """
    モック実装 (実ロボットなしでのデバッグ用)
    """

    def __init__(self, delay: float = 2.0):
        self.delay = delay

    def connect(self):
        print("[Mock] ロボット接続 (モック)")

    def send(self, pose_dict: dict) -> bool:
        print("[Mock] 把持姿勢送信:")
        print(f"  左手首: {pose_dict['left_hand'][0]}")
        print(f"  右手首: {pose_dict['right_hand'][0]}")
        return True

    def wait_for_result(self, timeout: float = 30.0) -> dict:
        print(f"[Mock] {self.delay}秒後に把持成功を返します...")
        time.sleep(self.delay)
        return {"success": True, "message": "把持成功 (モック)"}

    def disconnect(self):
        print("[Mock] ロボット切断 (モック)")
