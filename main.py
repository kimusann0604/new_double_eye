from Eye_landmarkProcessor import *
from img_inline import *
from Eye_processor import *

# インスタンス作成
processor = EyeLandmarkProcessor()
analyzer = SkinColorAnalyzer()
drawer = ExtendedEyeLandmarkDrawer()

# FaceMeshの初期化
face_mesh = processor.initialize_face_mesh()

# 画像のパス
file_path = '/Users/kimurahotaka/Documents/flask_process/bbc6c9066fa41d8de797b46e34d91a39.jpg'

# 画像の読み込みと処理
img_rgb = processor.image_path(file_path)
img = cv2.imread(file_path)

# 肌領域の検出と平均色の計算
image_ycrcb, image_rgb, lower, upper = analyzer.detect_skin_region(img)
skin_pixels, skin_pixel_values = analyzer.mask(image_ycrcb, lower, upper, image_rgb)
average_color_bgra = analyzer.calculate_average_color(skin_pixel_values)

# ランドマークを処理してマスクを生成
result_img = processor.process_landmarks_and_create_mask(face_mesh, img_rgb, img, average_color_bgra)

# 結果を表示
if result_img is not None:
    processor.result(result_img)

# ここからEye_processor.pyの機能を利用
drawer.set_image(result_img)
results = drawer.process_image()
drawer.draw_landmarks_with_extension(results)
drawer.show_image()