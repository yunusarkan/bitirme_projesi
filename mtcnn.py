import os
import numpy as np
from mtcnn import MTCNN
import cv2

detector = MTCNN()

video_dir = 'E:/Formatöncesi/Celeb-DF-v2/Celeb-real/'  # Video dosyalarının bulunduğu dizin
output_dir = 'E:/bitirme-projesi/Celeb-DF-v2/Celeb-real/frame'  # Frame'lerin kaydedileceği dizin

# Çıktı dizinini oluştur
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Dizin içindeki mp4 dosyalarını işle
video_num = 0  # Video numarası
for file_name in os.listdir(video_dir):
    if file_name.endswith('.mp4'):
        video_path = os.path.join(video_dir, file_name)
        output_subdir = os.path.join(output_dir, os.path.splitext(file_name)[0])

        # Çıktı alt dizinini oluştur
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        vidcap = cv2.VideoCapture(video_path)
        success = True
        count = 0
        max_confidence = 0
        max_confidence_frame = 0

        # Kaldığı yerden devam etmek için video numarasını kontrol et
        start_from_video = 201 # video id numarasını vererek istediğin videodan çıkartma yapmaya başlayabilirsin
        if video_num >= start_from_video:
            while success:
                success, image = vidcap.read()

                if success:
                    frame_result = detector.detect_faces(image)

                    if frame_result:  # Eğer frame_result boş değilse, yüz tespiti sonuçları mevcuttur
                        for i, result in enumerate(frame_result):
                            confidence = result['confidence']

                            if confidence > max_confidence:
                                max_confidence = confidence
                                max_confidence_frame = count

                            bounding_box = result['box']
                            x, y, w, h = bounding_box

                            # Yüz bölgesini JPEG dosyası olarak kaydet
                            cropped_image = image[y:y+h, x:x+w]
                            cropped_frame_path = os.path.join(output_subdir, f'cropped_frame{count}.jpg')
                            cv2.imwrite(cropped_frame_path, cropped_image)

                        count += 1

        video_num += 1
        # Her video için en yüksek güven düzeyine sahip kareyi bir metin dosyasına yazın
        result_text = f'{file_name}: En yüksek güven düzeyine sahip kare {max_confidence_frame}.'
        with open(os.path.join(output_dir, f'max_confidence_frames.txt'), 'a') as file:
            file.write(result_text + '\n')

       
print("İşlem tamamlandı.")