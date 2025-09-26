#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[ROS1 노드] ArUco 상대 위치/각도 퍼블리셔 (VideoCapture, Python3, no-tf, Overlay)
기능:
- id 1..12의 (x,y,yaw)을 기준 id=0 좌표계로 계산
- id 0과 1..12가 모두 보일 때만 `/aruco_relative/xyyaw_1_12` (Float64MultiArray, 길이36) 퍼블리시
- 각 id(1..12) 검출 여부를 `/aruco_relative/detected_flags_1_12` (UInt8MultiArray)로 퍼블리시
- 12개 모두 검출 여부를 `/aruco_relative/all_detected` (Bool)로 퍼블리시
- `_show_window:=true`면: 바운딩 박스, 좌표축, 상태 텍스트 오버레이 표기

파라미터:
- ~camera_device (기본 0), ~camera_width (1280), ~camera_height (720), ~camera_fps (30)
- ~camera_info_yaml (cam.yml), ~dictionary_id (DICT_6X6_250), ~marker_length_m (0.04), ~show_window (false)
"""

import os
import sys
import math
import numpy as np

import rospy
from std_msgs.msg import Float64MultiArray, UInt8MultiArray, MultiArrayDimension, Bool

try:
    import cv2
    aruco = cv2.aruco
except Exception as e:
    print("ERROR: OpenCV(cv2) 또는 aruco 모듈 로드 실패. opencv-contrib-python 설치 확인. err={}".format(e), file=sys.stderr)
    raise

# ----------------- 유틸 -----------------

def load_opencv_yaml(yaml_path):
    if not yaml_path or not os.path.exists(yaml_path):
        raise FileNotFoundError("camera_info_yaml 파일을 찾을 수 없습니다: {}".format(yaml_path))
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError("OpenCV FileStorage로 YAML을 열 수 없습니다: {}".format(yaml_path))
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("distortion_coefficients").mat()
    fs.release()
    if K is None or D is None:
        raise ValueError("YAML에 'camera_matrix' 또는 'distortion_coefficients'가 없습니다.")
    return K, D

def rotvec_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec); return R

def yaw_from_R(R):
    return math.atan2(R[1,0], R[0,0])

def relative_pose_from_camera_poses(rvec_base, tvec_base, rvec_i, tvec_i):
    R0 = rotvec_to_R(rvec_base); Ri = rotvec_to_R(rvec_i)
    t0 = tvec_base.reshape(3,);   ti = tvec_i.reshape(3,)
    R_rel = R0.T.dot(Ri)
    t_rel = R0.T.dot(ti - t0)
    yaw = yaw_from_R(R_rel)
    return t_rel, yaw

def get_aruco_dictionary(dict_id):
    return aruco.getPredefinedDictionary(dict_id) if hasattr(aruco, "getPredefinedDictionary") else aruco.Dictionary_get(dict_id)

def get_detector_parameters():
    return aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()

# ----------------- 메인 -----------------

class ArucoXYYawPublisher(object):
    def __init__(self):
        self.device = rospy.get_param("~camera_device", 2)
        # self.w = int(rospy.get_param("~camera_width", 640))
        # self.h = int(rospy.get_param("~camera_height", 480))
        self.w = int(rospy.get_param("~camera_width", 1920))
        self.h = int(rospy.get_param("~camera_height", 1080))
        self.fps = int(rospy.get_param("~camera_fps", 30))
        self.show = bool(rospy.get_param("~show_window", True))

        # self.yaml_path = rospy.get_param("~camera_info_yaml", "cam.yml")
        self.yaml_path = rospy.get_param("~camera_info_yaml", "cam_1920.yml")
        # self.yaml_path = rospy.get_param("~camera_info_yaml", "cam_realsense.yml")
        self.dict_id = int(rospy.get_param("~dictionary_id", int(aruco.DICT_4X4_50)))
        self.marker_len = float(rospy.get_param("~marker_length_m", 0.16))

        self.base_id = 0
        self.ids_target = list(range(1, 7))

        if not self.yaml_path:
            rospy.logerr("~camera_info_yaml 미설정: pose 추정 불가")
            raise RuntimeError("camera intrinsics required")
        self.K, self.D = load_opencv_yaml(self.yaml_path)

        self.dictionary = get_aruco_dictionary(self.dict_id)
        self.params = get_detector_parameters()
        self.detector = aruco.ArucoDetector(self.dictionary, self.params) if hasattr(aruco, "ArucoDetector") else None

        self.pub_xyyaw = rospy.Publisher("/aruco_relative/poses", Float64MultiArray, queue_size=1)
        self.pub_flags = rospy.Publisher("/aruco_relative/detected_flags_1_12", UInt8MultiArray, queue_size=1)
        self.pub_all   = rospy.Publisher("/aruco_relative/all_detected", Bool, queue_size=1)

        self.cap = cv2.VideoCapture(self.device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cap.set(cv2.CAP_PROP_FPS,          self.fps)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("카메라 열기 실패: {}".format(self.device))

        self._first_detect_logged = False
        rospy.loginfo("ArucoXYYawPublisher: device=%s, size=%dx%d@%dfps, dict=%d, base_id=0, targets=1..12",
                      str(self.device), self.w, self.h, self.fps, self.dict_id)

    def spin(self):
        rate = rospy.Rate(self.fps if self.fps > 0 else 30)
        while not rospy.is_shutdown():
            ok, frame = self.cap.read()
            if not ok:
                rospy.logwarn_throttle(2.0, "프레임 읽기 실패"); rate.sleep(); continue

            self.process_frame(frame)

            if self.show:
                cv2.imshow("vcap", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    rospy.signal_shutdown("ESC pressed"); break

            rate.sleep()

        try: self.cap.release()
        except Exception: pass
        if self.show: cv2.destroyAllWindows()

    def process_frame(self, img):
        # 검출
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(img)
        else:
            corners, ids, _ = aruco.detectMarkers(img, self.dictionary, parameters=self.params)

        # 미리보기: 감지된 마커 표시
        if self.show and ids is not None and len(ids) > 0:
            try: aruco.drawDetectedMarkers(img, corners, ids)
            except Exception: pass

        flags = np.zeros(len(self.ids_target), dtype=np.uint8)

        if ids is None or len(ids) == 0:
            self._publish_flags(flags)
            self.pub_all.publish(Bool(data=False))
            return

        ids = ids.flatten()

        if not self._first_detect_logged:
            rospy.loginfo("마커 검출 시작: 감지된 IDs=%s", list(map(int, ids)))
            self._first_detect_logged = True

        if 0 not in ids:
            for j, mid in enumerate(self.ids_target):
                flags[j] = 1 if mid in ids else 0
            self._publish_flags(flags)
            self.pub_all.publish(Bool(data=False))
            return

        try:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_len, self.K, self.D)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "estimatePoseSingleMarkers 실패: %s", e)
            self._publish_flags(flags)
            self.pub_all.publish(Bool(data=False))
            return

        id_to_pose = {}
        for i, mid in enumerate(ids):
            id_to_pose[int(mid)] = (rvecs[i].reshape(3,), tvecs[i].reshape(3,))

        # 좌표축 그리기 & 라벨 (미리보기)
        if self.show:
            for i, mid in enumerate(ids):
                try:
                    rvec_i, tvec_i = id_to_pose[int(mid)]
                except KeyError:
                    continue
                try:
                    if hasattr(aruco, "drawAxis"):
                        aruco.drawAxis(img, self.K, self.D, rvec_i, tvec_i, self.marker_len*0.5)
                    else:
                        cv2.drawFrameAxes(img, self.K, self.D, rvec_i, tvec_i, self.marker_len*0.5)
                except Exception:
                    pass
                c = corners[i][0] if i < len(corners) else None
                if c is not None:
                    cx, cy = int(c[:,0].mean()), int(c[:,1].mean())
                    color = (0,255,0) if int(mid) != self.base_id else (0,128,255)
                    cv2.putText(img, f"ID {int(mid)}", (cx-20, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        for j, mid in enumerate(self.ids_target):
            flags[j] = 1 if mid in id_to_pose else 0

        all_detected = bool(np.all(flags == 1) and (self.base_id in id_to_pose))
        self._publish_flags(flags)
        self.pub_all.publish(Bool(data=all_detected))

        if not all_detected:
            if self.show:
                y0 = 30
                cv2.rectangle(img, (5, 5), (265, 5 + 18*(len(self.ids_target)+3)), (0,0,0), -1)
                cv2.putText(img, "Detected (ID 1..12):", (10, y0-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                for k, mid in enumerate(self.ids_target, start=1):
                    ok = (flags[k-1] == 1)
                    color = (0,255,0) if ok else (0,0,255)
                    cv2.putText(img, f"{mid}: {'OK' if ok else '---'}", (10, y0+18*k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                cv2.putText(img, "Need ALL detected to publish xyyaw", (10, y0+18*(len(self.ids_target)+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            return

        rvec_b, tvec_b = id_to_pose[self.base_id]

        out = []
        rel_texts = []
        for mid in self.ids_target:
            rvec_i, tvec_i = id_to_pose[mid]
            t_rel, yaw_rel = relative_pose_from_camera_poses(rvec_b, tvec_b, rvec_i, tvec_i)
            # Mapping change: publish/display as (y, -x, yaw)
            y_mapped = float(t_rel[1])
            nx_mapped = float(-t_rel[0])
            out.extend([y_mapped, nx_mapped, float(yaw_rel)])
            rel_texts.append((mid, y_mapped, nx_mapped, yaw_rel))

        # 미리보기: 상대 y,-x,yaw 요약 표시
        if self.show:
            y0 = 30
            cv2.rectangle(img, (5, 5), (375, 5 + 18*(len(rel_texts)+3)), (0,0,0), -1)
            cv2.putText(img, "Base ID 0: ALL 1..12 DETECTED", (10, y0-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            for k,(mid, ry, rnx, ryaw) in enumerate(rel_texts, start=1):
                cv2.putText(img, f"ID {mid}: y={ry:.3f}  -x={rnx:.3f}  yaw={ryaw:.3f} rad",
                            (10, y0+18*k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,255), 1, cv2.LINE_AA)

        msg = Float64MultiArray()
        if len(out) == 36:
            dim = MultiArrayDimension(); dim.label="xyyaw_by_id_1_to_12"; dim.size=36; dim.stride=36
            msg.layout.dim = [dim]
        msg.data = out
        self.pub_xyyaw.publish(msg)

    def _publish_flags(self, flags_np):
        m = UInt8MultiArray()
        dim = MultiArrayDimension(); dim.label="detected_flags_id_1_to_12"
        dim.size = flags_np.size; dim.stride = flags_np.size
        m.layout.dim = [dim]
        m.data = flags_np.astype(np.uint8).tolist()
        self.pub_flags.publish(m)

def main():
    rospy.init_node("aruco_xyyaw_vcap", anonymous=False)
    node = ArucoXYYawPublisher()
    rospy.loginfo("aruco_xyyaw_vcap 노드가 시작되었습니다.")
    node.spin()

if __name__ == "__main__":
    main()
