import cv2
import time
import math
import numpy as np
from dazu.CPS import CPSClient
import yaml
from scipy.spatial.transform import Rotation as R


def rpy_to_transformation_matrix(x, y, z, roll, pitch, yaw):
    # Create rotation matrix from RPY angles
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)  # Specify the sequence of axes 'xyz'
    rotation_matrix = r.as_matrix()

    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)  # Start with identity matrix

    # Set the rotation part (3x3) of the transformation matrix
    transformation_matrix[:3, :3] = rotation_matrix

    # Set the translation part (position) in the last column
    transformation_matrix[:3, 3] = [x, y, z]

    return transformation_matrix


# Function to convert a 4x4 transformation matrix to position and RPY
def transformation_matrix_to_rpy(matrix):
    # Extract the translation (x, y, z) from the last column of the matrix
    translation = matrix[:3, 3]
    x, y, z = translation

    # Extract the rotation matrix (top-left 3x3 part of the transformation matrix)
    rotation_matrix = matrix[:3, :3]

    # Convert the rotation matrix to RPY (roll, pitch, yaw) in radians
    r = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = r.as_euler('xyz', degrees=True)

    return x, y, z, roll, pitch, yaw


def update_pose(current_pose, x, y, z, roll, pitch, yaw):
    # Extract the current position and rotation (RPY) from the current pose
    current_position = current_pose[:3]  # Assuming current_pose = [x, y, z, roll, pitch, yaw]
    current_rotation = current_pose[3:]

    # Update translation (position)
    updated_position = np.array(current_position) + np.array([x, y, z])

    # Create the rotation from RPY (roll, pitch, yaw)
    new_rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)

    # Convert the current rotation (RPY) to a Rotation object
    current_rotation_r = R.from_euler('xyz', current_rotation, degrees=True)

    # Combine the rotations (multiplying quaternion or matrix form)
    updated_rotation = current_rotation_r * new_rotation

    # Extract updated RPY from the new combined rotation
    updated_rpy = updated_rotation.as_euler('xyz', degrees=True)

    # Return the updated pose (position + RPY)
    updated_pose = np.concatenate([updated_position, updated_rpy])

    return updated_pose


class RobotController:
    def __init__(self, box_id=0, rbt_id=0, position_file='./config/position.yaml',
                 calibration_file='./config/hand_eye_config.yaml',
                 step_size=0.2, rotation_step_size=0.1):
        """
        初始化机器人控制器
        :param step_size: 控制机器人每次移动的步进量（默认值为1）
        :param rotation_step_size: 控制机器人每次旋转的步进量（默认值为5）
        """
        self.box_id = box_id
        self.rbt_id = rbt_id
        self.position_file = position_file
        self.calibration_file = calibration_file
        self.client = CPSClient()
        self.step_size = step_size  # 步进量
        self.rotation_step_size = rotation_step_size  # 旋转步进量

        # 连接到电箱和控制器
        self.client.HRIF_Connect(self.box_id, '192.168.11.7', 10003)
        self.client.HRIF_Connect2Controller(self.box_id)

        # 获取当前位置
        self.current_pose = self.client.read_pos()
        print(f"Initial Current Pose: {self.current_pose}")

        # 读取目标位置
        self.positions = self.read_position()

    def read_position(self, tag='positions'):
        with open(self.position_file, 'r') as file:
            positions = yaml.safe_load(file)
        return positions[tag]

    def move_arm(self, pose, dServoTime):
        """
        控制机械臂移动到指定的pose，pose包含位置和姿态的6个参数（x, y, z, Rx, Ry, Rz）。
        """
        ucs = [0, 0, 0, 0, 0, 0]  # 运动坐标系的占位符
        tcp = [0, 0, 0, 0, 0, 0]  # 末端执行器坐标系的占位符
        res = self.client.HRIF_PushServoP(self.box_id, self.rbt_id, pose, ucs, tcp)
        print(res)
        time.sleep(0.001)  # 延迟，防止命令重叠

    def control_arm_with_cv2(self):
        self.current_pose = self.client.read_pos()
        matrix = rpy_to_transformation_matrix(self.current_pose[0], self.current_pose[1], self.current_pose[2],
                                              self.current_pose[3], self.current_pose[4], self.current_pose[5])
        bat_length = rpy_to_transformation_matrix(0, 0, 258, 0, 0, 0)
        bat_matrix = matrix @ bat_length
        bat_pose = np.array(transformation_matrix_to_rpy(bat_matrix), dtype=float)
        print(bat_pose)

def main():
    controller = RobotController()
    controller.control_arm_with_cv2()  # 启动cv2控制


# 启动程序
if __name__ == "__main__":
    main()
