from preprocessing import defect_detection
from build_data import crop__defects
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

CLASS_NAMES = ['Open Circuit', 'Short Circuit', 'Missing Component', 'Spur', 'Spurious Copper', 'Mouse Bite']

def preprocess_for_prediction(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype('float32') / 255.0
    final_img = np.expand_dims(normalized, axis=(0,-1))
    return final_img

def main(img_template_path, img_test_path):
    try:
        model = load_model("pcb_defect_classifier.h5")
        print("Model loaded successfully.")
    except Exception as e:
        print("Error")
        return
    img_template = cv2.imread(img_template_path)
    img_test = cv2.imread(img_test_path)
    contours, mask = defect_detection(img_template, img_test)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  
            x, y, w, h = cv2.boundingRect(cnt)
            x,y = max(x-10, 0), max(y-10, 0)
            defect_img = crop__defects(img_test, x, y, w+20, h+20)
            preprocessed_img = preprocess_for_prediction(defect_img)
            prediction = model.predict(preprocessed_img)
            label = np.argmax(prediction)
            cv2.rectangle(img_test, (x, y), (x + w+20, y + h+20), (0, 255, 0), 1)
            cv2.putText(img_test, CLASS_NAME[label], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Defects Detected", img_test)
    cv2.imshow("Template", img_template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    
    img_template_path = input("Enter the path of the template image: ")
    img_test_path = input("Enter the path of the test image: ")
    main(img_template_path, img_test_path)
