"""
Plant Disease Model Training
AI Precision Agriculture System
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

DATASET_PATH = "C:/Users/Admin/OneDrive/Desktop/agriculture/Datasets/plant_disease_dataset"


IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 3


# -------------------------------------------------
# DATA GENERATOR
# -------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    color_mode="rgb",
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True,
    color_mode="rgb",
    
)

print("Class Indices:")
print(train_generator.class_indices)


# -------------------------------------------------
# MODEL
# -------------------------------------------------

base_model = MobileNetV2(

    input_shape=(224,224,3),

    include_top=False,

    weights="imagenet"

)

for layer in base_model.layers:

    layer.trainable = False


x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(256,activation="relu")(x)

predictions = Dense(

    train_generator.num_classes,

    activation="softmax"

)(x)


model = Model(

    inputs=base_model.input,

    outputs=predictions

)


model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    
)

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------

model.save("Models/plant_disease_model.h5")

print("Model Saved Successfully")


# -------------------------------------------------
# PLOT ACCURACY
# -------------------------------------------------

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.show()