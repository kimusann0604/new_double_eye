import cv2
import numpy as np

class SkinColorAnalyzer:

    def detect_skin_region(self, image):
        # BGRからRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # YCrCb色空間に変換
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # 肌色の範囲を定義
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        return image_ycrcb, image_rgb, lower, upper

    def mask(self, image_ycrcb, lower, upper, image_rgb):
        # 肌領域のマスクを作成
        mask = cv2.inRange(image_ycrcb, lower, upper)
        # 肌ピクセルの抽出
        skin_pixels = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        # 有効なピクセルのインデックスを取得
        skin_indices = np.where(mask != 0)
        # 肌のピクセル値を抽出
        skin_pixel_values = image_rgb[skin_indices]
        return skin_pixels, skin_pixel_values

    def calculate_average_color(self, skin_pixel_values):
        # 平均色の計算
        average_color = skin_pixel_values.mean(axis=0).astype(int)
        # アルファ値（不透明度）を追加
        average_color = np.append(average_color, 255)
        print(average_color)
        # RGBからBGRAに変換（OpenCVはBGR順なので）
        average_color_bgra = tuple(int(c) for c in average_color[::-1])
        return average_color_bgra





