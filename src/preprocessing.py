import cv2
import numpy as np

def defect_detection(img_template, img_test):
    diff = cv2.bitwise_xor(img_template, img_test)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY) 
    kernel1 = np.ones((3,3), np.uint8)
    kernel2 = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
    mask = cv2.dilate(mask, kernel2, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

if __name__ == "__main__":
    img_template = cv2.imread("PCBData/group00041/00041/00041009_temp.jpg")
    img_test = cv2.imread("PCBData/group00041/00041/00041009_test.jpg")
    contours, mask = defect_detection(img_template, img_test)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  
            x, y, w, h = cv2.boundingRect(cnt)
            x, y = max(x-10, 0), max(y-10, 0)
            cv2.rectangle(img_test, (x, y), (x + w+20, y + h+20), (0, 255, 0), 1)
    cv2.imshow("mask", mask)
    cv2.imshow("Defects", img_test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
