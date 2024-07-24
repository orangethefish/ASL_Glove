import tensorflow as tf
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model


class TimeSeriesDataset:
    def __init__(self, root_dir, feature_names=[]):
        self.data = self.load_data(root_dir, feature_names)

    def load_data(self, root_dir, feature_names):
        data = []

        for individual_dir in sorted(os.listdir(root_dir)):
            individual_path = os.path.join(root_dir, individual_dir)
            for class_dir in sorted(os.listdir(individual_path)):
                class_path = os.path.join(individual_path, class_dir)
                if os.path.isdir(class_path):
                    for file in glob.glob(os.path.join(class_path, "*.csv")):
                        df = pd.read_csv(file, usecols=feature_names)
                        class_name = os.path.splitext(os.path.basename(file))[0]
                        df["class"] = class_name
                        data.append(df)


        # Concatenate all data frames into a single data frame
        data = pd.concat(data, ignore_index=True)
        
        return data
    
root_dir = "glove_data"
feature_names = [
    "flex_1", "flex_2", "flex_3", "flex_4", "flex_5",
    "GYRx", "GYRy", "GYRz",
    "ACCx", "ACCy", "ACCz"
]

dataset = TimeSeriesDataset(root_dir, feature_names).data
dataset = dataset.sort_values(by=["class"])
filter_classes = ["a" , "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
dataset = dataset[dataset["class"].isin(filter_classes)]


n_features = 11
num_classes = 40
samples_per_class = 250
timesteps = 150
total_samples = num_classes * samples_per_class

x = np.random.rand(total_samples, timesteps, n_features)
y = np.repeat(np.arange(num_classes), samples_per_class)

# One-hot encode the labels
y_one_hot = tf.keras.utils.to_categorical(y, num_classes)


# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.2, random_state=42, shuffle=True)


def build_mcdcnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    convs = []
    for i in range(input_shape[-1]):
        conv = Conv1D(filters=32, kernel_size=5, activation='relu')(inputs[:, :, i:i+1])
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Dropout(0.2)(conv)
        conv = Flatten()(conv)
        convs.append(conv)

    merged = Concatenate()(convs)
    dense = Dense(64, activation='relu')(merged)
    dense = Dropout(0.5)(dense)
    outputs = Dense(num_classes, activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Build and compile the model
input_shape = (timesteps, n_features)
model = build_mcdcnn(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()


# Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")