import cv2
import numpy as np
import dlib
from imutils import face_utils

# 画像のパスを指定
image_path = '/Users/kimurahotaka/Documents/新たなアルゴリズムdouble_eye/1.jpg'

# 画像を読み込む
image = cv2.imread(image_path)

# 画像が正しく読み込まれたか確認
if image is None:
    print("画像を読み込めません。パスを確認してください。")
    exit()

# 顔検出器とランドマーク検出器の初期化
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/kimurahotaka/Documents/新たなアルゴリズムdouble_eye/肌のrbg/shape_predictor_68_face_landmarks.dat 3')  # ダウンロードしたモデルのパス

# グレースケールに変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔検出
faces = detector(gray)

if len(faces) == 0:
    print("顔が検出されませんでした。")
    exit()

# 検出されたすべての顔に対して処理を実行
for face in faces:
    # ランドマーク検出
    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)

    # 鼻のランドマークポイントを取得
    nose_points = shape[[28, 29, 30, 31, 32, 33, 34, 35, 36]]

    # ランドマークポイントの最小矩形を計算
    x_coords = nose_points[:, 0]
    y_coords = nose_points[:, 1]
    x_min = max(0, np.min(x_coords) - 5)  # 余白を少し追加
    y_min = max(0, np.min(y_coords) - 5)
    x_max = min(image.shape[1], np.max(x_coords) + 5)
    y_max = min(image.shape[0], np.max(y_coords) + 5)

    # 鼻部分のROIを抽出
    nose_roi = image[y_min:y_max, x_min:x_max]

    # 矩形を画像に描画
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # 鼻部分の平均RGB値を計算
    if nose_roi.size > 0:
        average_b = np.mean(nose_roi[:, :, 0])
        average_g = np.mean(nose_roi[:, :, 1])
        average_r = np.mean(nose_roi[:, :, 2])
        print(f"鼻の中央領域RGB値 - R: {average_r:.2f}, G: {average_g:.2f}, B: {average_b:.2f}")
    else:
        print("鼻の領域が抽出されませんでした。")

# 画像の表示（オプション）
cv2.imshow('Image with Nose Region', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
