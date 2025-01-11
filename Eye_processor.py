# Eye_processor.py
import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import CubicSpline

class ExtendedEyeLandmarkDrawer:
    def __init__(self, static_image_mode=True, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.9):
        # MediaPipe FaceMesh の初期化（必要ない場合もあります）
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence
        )
        self.img = None
        self.img_rgb = None
        self.h = 0
        self.w = 0
        self.shift_value = -8  # Y軸のシフト値
        
    def set_image(self, img):
        self.img = img
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.h, self.w, _ = self.img.shape

    def draw_landmarks_with_extension(self, face_landmarks_list):
        """
        既に取得済みのランドマークを使用して、延長線を描画します。
        
        :param face_landmarks_list: ランドマークのリスト（各顔ごとのランドマーク）
        """
        if not face_landmarks_list:
            print("ランドマークが提供されていません。")
            return

        # マスク作成（延長線のみ描画用）
        mask = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        for face_landmarks in face_landmarks_list:
            # 左目と右目のランドマークインデックス
            left_eye_indices = [159, 158, 157, 173, 133, 246, 161, 160]
            right_eye_indices = [386, 385, 384, 398, 362, 263, 466]

            # 左目と右目の線を描画
            self._draw_lines(mask, face_landmarks, left_eye_indices)
            self._draw_lines(mask, face_landmarks, right_eye_indices)

        # マスクをぼかす（延長部分のみ）
        blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # 元の画像とぼかしたマスクを合成
        self.img = cv2.addWeighted(self.img, 1.0, blurred_mask, 0.5, 0)

    def _draw_lines(self, img, face_landmarks, indices):
        # 画像に描くためのポイントをリストに格納
        points = []
        x_coords = []
        y_coords = []

        # 各ランドマークの座標を取得
        for i in indices:
            x = face_landmarks.landmark[i].x * self.w
            y = (face_landmarks.landmark[i].y * self.h) + self.shift_value  # Y軸のシフトを適用
            x_coords.append(x)
            y_coords.append(y)
            points.append((int(x), int(y)))

        # x座標でソート
        sorted_indices = np.argsort(x_coords)
        points = [points[i] for i in sorted_indices]

        # 確保：少なくとも2点必要
        if len(points) < 2:
            return

        # 延長距離を設定
        extension_length = 10  # 基本の延長距離

        # 線を二重延長するための関数
        def extend_line(p_start, p_end, length):
            direction = p_end - p_start
            norm = np.linalg.norm(direction)
            if norm == 0:
                return p_end
            unit_vector = direction / norm
            p_new = p_end + unit_vector * length
            return tuple(p_new.astype(int))

        # 左目の線を延長
        p1 = np.array(points[-2])
        p2 = np.array(points[-1])
        p_new1 = extend_line(p1, p2, extension_length)
        p_new1_double = extend_line(p2, p_new1, extension_length)
        cv2.line(img, tuple(p2), p_new1, (0, 0, 0), 1)  # 一度目の延長線（黒色）
        cv2.line(img, tuple(p_new1), p_new1_double, (0, 0, 0), 1)  # 二度目の延長線（黒色）

        # 右目の線を延長
        p3 = np.array(points[0])
        p4 = np.array(points[1])
        p_new2 = extend_line(p4, p3, extension_length)
        p_new2_double = extend_line(p3, p_new2, extension_length)
        cv2.line(img, tuple(p3), tuple(p_new2), (0, 0, 0), 1)  # 一度目の延長線（黒色）
        cv2.line(img, tuple(p_new2), p_new2_double, (0, 0, 0), 1)  # 二度目の延長線（黒色）

    def show_image(self):
        if self.img is None:
            raise ValueError("画像がロードされていません。まず set_image() を呼び出してください。")
        # 結果を表示
        cv2.imshow('Facial Landmarks with Extension', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
