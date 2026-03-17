import cv2
import os
import shutil
import random

def crop__defects(img, x, y, w, h):
    crop = img[y:(y+h), x:(x+w)]
    scale = min(64/w, 64/h)
    new_h, new_w = int(h*scale), int(w*scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.copyMakeBorder(resized,
                                            top=(64 - new_h) // 2,
                                            bottom=(64 - new_h) // 2,
                                            left=(64 - new_w) // 2,
                                            right=(64 - new_w) // 2,
                                            borderType=cv2.BORDER_CONSTANT,
                                            value=[255, 255, 255])
    return canvas


def save_defects(img, txt_path):
    with open(txt_path, 'r') as f:
        for idx, line in enumerate(f):
            x1, y1, x2, y2, label = line.strip().split()
            defect_img = crop__defects(img, int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1))
            save_folder = label
            save_path = os.path.join(save_folder, f"{os.path.basename(txt_path).split('.')[0]}_{label}_{idx}.jpg")
            cv2.imwrite(save_path, defect_img)



def split_data(img_folder, label, train=0.7, val=0.2, test=0.1):
    images = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
    images = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
    random.seed(42) 
    random.shuffle(images) 
    total = len(images)
    train_size = int(total*train)
    val_size = int(total*val)
    test_size = total - train_size - val_size
    train_file = images[:train_size]
    val_file = images[train_size:train_size+val_size]
    test_file = images[train_size+val_size:]
    for f in train_file:
        shutil.copy(os.path.join(img_folder, f), os.path.join("data/train", str(label), f))
    for f in val_file:
        shutil.copy(os.path.join(img_folder, f), os.path.join("data/val", str(label), f))
    for f in test_file:
        shutil.copy(os.path.join(img_folder, f), os.path.join("data/test", str(label), f))
    print(f"Data split completed: {len(train_file)} train, {len(val_file)} val, {len(test_file)} test")
        


if __name__ == "__main__":
    #make folders for each label
    for label in range(1, 7):
        os.makedirs(str(label), exist_ok=True)
    for split in ['train', 'val', 'test']:
        for label in range(1, 7):
            os.makedirs(os.path.join("data", split, str(label)), exist_ok=True)
    
    root = "PCBData"
    
    for sub_folder in os.listdir(root):
        sub_folder_path = os.path.join(root, sub_folder)
        if os.path.isdir(sub_folder_path):
            for child in os.listdir(sub_folder_path):
                child_path = os.path.join(sub_folder_path, child)
                if os.path.isdir(child_path) and not child.endswith('_not'):
                    label_folder = os.path.join(sub_folder_path, child + "_not")
                    print("images in folder:", child_path)
                    print("labels in folder:", label_folder)
                    for img_file in os.listdir(child_path):
                        if img_file.endswith("_test.jpg"):
                            img_path = os.path.join(child_path, img_file)
                            label_file = os.path.join(label_folder, img_file.replace("_test.jpg", ".txt"))
                            img = cv2.imread(img_path)
                            save_defects(img, label_file)
                            
    #split data
    for label in range(1,7):
        img_folder = str(label)
        split_data(img_folder, label)
        shutil.rmtree(img_folder)
