import numpy as np
import cv2
import time
from src.build_data import crop__defects
from src.main import preprocess_for_prediction
from src.preprocessing import defect_detection

def evaluate_model(template_folder, test_folder, model):
    start_detect = time.time()
    img_template = cv2.imread(template_folder)
    img_test = cv2.imread(test_folder)
    contours, mask = defect_detection(img_template, img_test)
    det_time = (time.time() - start_detect)*1000

    start_clsf = time.time()
    labels = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  
            x, y, w, h = cv2.boundingRect(cnt)
            x, y = max(x-10, 0), max(y-10, 0)
            defect_img = crop__defects(img_test, x, y, w+20, h+20)
            preprocessed_img = preprocess_for_prediction(defect_img)
            prediction = model.predict(preprocessed_img)
            label = np.argmax(prediction)+1
            labels.append(label)

    clf_time = (time.time() - start_clsf) * 1000 
    total_time = det_time + clf_time

    print(f"--- Thống kê thời gian ---")
    print(f"Phát hiện lỗi (OpenCV): {det_time:.2f} ms")
    print(f"Phân loại {len(labels)} lỗi (CNN): {clf_time:.2f} ms")
    print(f"Tổng cộng: {total_time:.2f} ms")
    
    return det_time, clf_time, total_time

if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    model = load_model("pcb_defect_classifier.h5")
    det_time, clf_time = 0, 0
    total_time = 0
    count = 0
    with open("PCBData/test.txt", "r") as f:
        for line in f:
            test_file, _ = line.strip().split()
            base = test_file.replace(".jpg", "")
            template_path = f"PCBData/{base}_temp.jpg"
            test_path = f"PCBData/{base}_test.jpg"
            det_t, clf_t, time_taken = evaluate_model(template_path, test_path, model)
            det_time += det_t
            clf_time += clf_t
            total_time += time_taken
            count += 1
    print(f"\n--- Trung bình thời gian trên {count} ảnh ---")
    print(f"Trung bình thời gian xử lý mỗi ảnh: {total_time/count:.2f} ms")
    print(f"Trung bình thời gian phát hiện lỗi: {det_time/count:.2f} ms")
    print(f"Trung bình thời gian phân loại lỗi: {clf_time/count:.2f} ms")

