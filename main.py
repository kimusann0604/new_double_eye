# main.py
import cv2
from Eye_landmarkProcessor import EyeLandmarkProcessor
from Eye_processor import ExtendedEyeLandmarkDrawer

def main():
    # インスタンス作成
    processor = EyeLandmarkProcessor()
    drawer = ExtendedEyeLandmarkDrawer()

    # FaceMeshの初期化
    face_mesh = processor.initialize_face_mesh()
    
    # 画像のパス
    file_path = '/Users/kimurahotaka/Documents/新たなアルゴリズムdouble_eye/1.jpg'

    # 画像の読み込みと処理
    try:
        img_rgb = processor.image_path(file_path)
    except FileNotFoundError as e:
        print(e)
        return

    img = cv2.imread(file_path)
    if img is None:
        print("画像が見つかりません。パスを確認してください。")
        return

    # 肌検出と平均色の計算を削除
    # 代わりに黒色を使用
    average_color_bgra = ( 158, 183, 207,255)  

    # ランドマークを処理してマスクを生成し、ランドマークを取得
    result_img, face_landmarks = processor.process_landmarks_and_create_mask(face_mesh, img_rgb, img, average_color_bgra)

    # 結果を表示
    if result_img is not None:
        processor.result(result_img)

    if face_landmarks is not None:
        # ランドマークの詳細を表示
        print("取得したランドマークの詳細:")
        for idx, landmark in enumerate(face_landmarks.landmark):
            print(f"ランドマーク {idx}: (x={landmark.x}, y={landmark.y}, z={landmark.z})")

        # Eye_processor.pyの機能を利用
        # 画像を設定してランドマークの描画のみを行う（再度のランドマーク検出を避ける）
        drawer.set_image(result_img)

        # `draw_landmarks_with_extension` メソッドを使用して延長線を描画
        drawer.draw_landmarks_with_extension([face_landmarks])  # 複数の顔がある場合はリストで渡す
        drawer.show_image()

if __name__ == "__main__":
    main()
