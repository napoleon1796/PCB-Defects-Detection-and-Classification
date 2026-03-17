from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("pcb_defect_classifier.h5")
CLASS_NAMES = ['Open Circuit', 'Short Circuit', 'Missing Component', 'Spur', 'Spurious Copper', 'Mouse Bite']

import tensorflow as tf
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/test",
    image_size=(64, 64),
    batch_size=32,
    shuffle=False,
    color_mode='grayscale' 
)

y_pred_probs = model.predict(test_ds)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

y_true = []
for images, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

accuracy = np.sum(y_true == y_pred_classes) / len(y_true)
print(f"Thực tế Accuracy tính từ Confusion Matrix: {accuracy * 100:.2f}%")

# 5. create confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.show()
