import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.rgb_to_grayscale(image)
    return image, label

def load_data(data_dir, img_size = (64,64), batch_size = 32):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=img_size,
        batch_size=batch_size
    )
    ds = ds.map(normalize)
    return ds   
def build_model(input_shape=(64, 64, 1), num_classes=6):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    train_ds = load_data("data/train").shuffle(buffer_size=7000).prefetch(tf.data.AUTOTUNE)
    val_ds = load_data("data/val").prefetch(tf.data.AUTOTUNE)
    test_ds = load_data("data/test").prefetch(tf.data.AUTOTUNE)

    model = build_model()
    model.summary()
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    loss, accuracy = model.evaluate(test_ds) 
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy*100:.2f}%")
    
    model.save("pcb_defect_classifier.h5")
    print("Model saved as pcb_defect_classifier.h5")
    
    plt.figure(figsize=(12, 8))
    plt.plot(train_loss,'b-o', label='Train Loss')
    plt.plot(val_loss, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()