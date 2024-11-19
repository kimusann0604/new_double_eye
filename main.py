from Eye_landmarkProcessor import *
from img_inline import *

# インスタンス作成
processor = EyeLandmarkProcessor()
analyzer = SkinColorAnalyzer()

# FaceMeshの初期化
face_mesh = processor.initialize_face_mesh()

# 画像のパス
file_path = '/Users/kimurahotaka/Documents/flask_process/bbc6c9066fa41d8de797b46e34d91a39.jpg'
img_rgb = processor.image_path(file_path)

# 元画像を再度読み込み（RGB以外の処理用）
img = cv2.imread(file_path)

# 2. 肌領域を検出
image_ycrcb, image_rgb, lower, upper = analyzer.detect_skin_region(img)

# 3. マスク作成と肌色の抽出
skin_pixels, skin_pixel_values = analyzer.mask(image_ycrcb, lower, upper, image_rgb)

# 4. 平均色の計算
average_color_bgra = analyzer.calculate_average_color(skin_pixel_values)

# ランドマークを処理してマスクを生成（平均肌色を渡す）
result_img = processor.process_landmarks_and_create_mask(face_mesh, img_rgb, img, average_color_bgra)

# 結果を表示
if result_img is not None:
    processor.result(result_img)
