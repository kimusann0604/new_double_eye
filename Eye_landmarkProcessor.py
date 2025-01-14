# Eye_landmarkProcessor.py
import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import CubicSpline

class EyeLandmarkProcessor:
    def initialize_face_mesh(self):
        mp_face_mesh = mp.solutions.face_mesh

        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        return face_mesh

    def image_path(self, file_path):
        """
        画像を読み込み、RGB形式に変換します。
        """
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"画像が見つかりません: {file_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"画像を読み込みました: {file_path}, 画像サイズ: {img_rgb.shape}")
        return img_rgb

    def process_landmarks_and_create_mask(self, face_mesh, img_rgb, img, average_color_bgra=None):
        """
        顔のランドマークを検出し、目の部分にマスクを適用します。
        """
        # 顔のランドマーク検出
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            print("顔が検出されませんでした。")
            return img, None  # 元の画像を返す
        else:
            for face_landmarks in results.multi_face_landmarks:
                # 左目と右目のランドマークインデックス
                left_eye_indices = [159, 158, 157, 173, 133, 246, 161, 160]
                right_eye_indices = [386, 385, 384, 398, 362, 263, 466]

                # 画像のサイズを取得
                h, w, _ = img.shape

                # 左目の座標を取得
                upper_left_eyelid_coords = np.array([
                    [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                    for i in left_eye_indices
                ])

                # 右目の座標を取得
                upper_right_eyelid_coords = np.array([
                    [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                    for i in right_eye_indices
                ])

                # シフト値を設定
                shift_value = -8

                # 左目の座標とシフト適用
                x = upper_left_eyelid_coords[:, 0]
                y = upper_left_eyelid_coords[:, 1] + shift_value

                # 右目の座標とシフト適用
                x2 = upper_right_eyelid_coords[:, 0]
                y2 = upper_right_eyelid_coords[:, 1] + shift_value

                # x と y をソート
                sorted_indices = np.argsort(x)
                x = x[sorted_indices]
                y = y[sorted_indices]

                # x2 と y2 をソート
                sorted_indices2 = np.argsort(x2)
                x2 = x2[sorted_indices2]
                y2 = y2[sorted_indices2]

                # スプライン補間
                cs = CubicSpline(x, y)
                cs2 = CubicSpline(x2, y2)

                # 補間用の細かいx値を生成
                x_fine = np.linspace(x.min(), x.max(), 10)
                x_fine2 = np.linspace(x2.min(), x2.max(), 15)

                # 補間されたy値を計算
                y_fine = cs(x_fine)
                y_fine2 = cs2(x_fine2)

                # ポイントを整数に変換
                points_left = np.vstack((x_fine, y_fine)).astype(np.int32).T
                points_right = np.vstack((x_fine2, y_fine2)).astype(np.int32).T

                # アルファチャンネル付きのマスクを作成
                mask = np.zeros((h, w, 4), dtype=np.uint8)

                # マスクに線を描画（指定色、アルファチャンネルも設定）
                if average_color_bgra is None:
                    average_color_bgra = (158, 183, 207, 255)  # 黒色
                cv2.polylines(mask, [points_left], isClosed=False, color=average_color_bgra, thickness=1, lineType=cv2.LINE_AA)
                cv2.polylines(mask, [points_right], isClosed=False, color=average_color_bgra, thickness=1, lineType=cv2.LINE_AA)

                # マスクをカラー部分とアルファチャンネルに分離
                mask_rgb = mask[:, :, :3]
                mask_alpha = mask[:, :, 3]

                # ぼかし処理を行わない
                blurred_rgb = mask_rgb.copy()
                blurred_alpha = mask_alpha.copy()
                blurred_mask = np.dstack([blurred_rgb, blurred_alpha])

                # 元の画像にアルファチャンネルを追加
                img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

                # マスクを重ねる
                alpha_mask = blurred_mask[:, :, 3] / 255.0  # アルファチャンネルを正規化
                alpha_inv = 1.0 - alpha_mask

                # 各チャンネルに対して合成
                for c in range(0, 3):
                    img_rgba[:, :, c] = (alpha_inv * img_rgba[:, :, c] + alpha_mask * blurred_mask[:, :, c])

                # アルファチャンネルを削除してBGR画像に戻す
                result_img = cv2.cvtColor(img_rgba, cv2.COLOR_BGRA2BGR)
                return result_img, face_landmarks  # ランドマークを返す

    def result(self, result_img):
        """
        処理結果の画像を表示します。
        """
        cv2.imshow('Result', result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
