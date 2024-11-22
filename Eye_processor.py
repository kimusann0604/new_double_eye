import cv2
import mediapipe as mp
import numpy as np

class ExtendedEyeLandmarkDrawer:
    def __init__(self, static_image_mode=True, max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.9):
        # MediaPipe FaceMesh の初期化
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
        
    def set_image(self, img):
        self.img = img
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.h, self.w, _ = self.img.shape

    def load_image(self, file_name):
        # 画像の読み込み
        self.img = cv2.imread(file_name)
        if self.img is None:
            raise FileNotFoundError("画像が見つかりません。パスを確認してください。")
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.h, self.w, _ = self.img.shape

    def process_image(self):
        # 顔のランドマーク検出
        if self.img_rgb is None:
            raise ValueError("画像がロードされていません。まず load_image() を呼び出してください。")
        return self.face_mesh.process(self.img_rgb)

    def draw_landmarks_with_extension(self, results):
        if not results.multi_face_landmarks:
            print("顔が検出されませんでした。")
            return

        for face_landmarks in results.multi_face_landmarks:
            # 左目と右目のランドマークインデックス
            left_eye_indices = [159, 158, 157, 173, 133, 246, 161, 160]
            right_eye_indices = [386, 385, 384, 398, 362, 263, 466]

            # 左目と右目の線を描画
            self._draw_lines(self.img, face_landmarks, left_eye_indices)
            self._draw_lines(self.img, face_landmarks, right_eye_indices)

    def _draw_lines(self, img, face_landmarks, indices):
        # 画像に描くためのポイントをリストに格納
        points = []
        x_coords = []
        y_coords = []

        # 各ランドマークの座標を取得
        for i in indices:
            x = face_landmarks.landmark[i].x * self.w
            y = face_landmarks.landmark[i].y * self.h
            x_coords.append(x)
            y_coords.append(y)
            points.append((int(x), int(y)))

        # x座標でソート
        sorted_indices = np.argsort(x_coords)
        points = [points[i] for i in sorted_indices]

        # 最後の線を延長
        if len(points) >= 2:
            p1 = np.array(points[-2])
            p2 = np.array(points[-1])
            p3 = np.array(points[0])
            p4 = np.array(points[1])
            direction1 = p2 - p1
            direction2 = p3 - p4

            if np.linalg.norm(direction1) != 0:
                unit_vector1 = direction1 / np.linalg.norm(direction1)
                extension_length = 10  # 延長距離（ピクセル）
                p_new1 = p2 + unit_vector1 * extension_length
                p_new1 = tuple(p_new1.astype(int))
                cv2.line(img, tuple(p2), p_new1, (255, 0, 0), 1)  # 青い線で延長

            if np.linalg.norm(direction2) != 0:
                unit_vector2 = direction2 / np.linalg.norm(direction2)
                extension_length2 = 10  # 延長距離（ピクセル）
                p_new2 = p3 + unit_vector2 * extension_length2
                p_new2 = tuple(p_new2.astype(int))
                cv2.line(img, tuple(p3), p_new2, (255, 0, 0), 1)  # 青い線で延長

    def show_image(self):
        if self.img is None:
            raise ValueError("画像がロードされていません。まず load_image() を呼び出してください。")
        # 結果を表示
        cv2.imshow('Facial Landmarks with Extension', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 使用例
if __name__ == "__main__":
    drawer = ExtendedEyeLandmarkDrawer()
    try:
        drawer.load_image('/Users/kimurahotaka/Documents/flask_process/bbc6c9066fa41d8de797b46e34d91a39.jpg')
        results = drawer.process_image()
        drawer.draw_landmarks_with_extension(results)
        drawer.show_image()
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
