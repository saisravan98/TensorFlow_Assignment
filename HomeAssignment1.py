#1.Tensor Manipulations & Reshaping
#Task: Tensor Reshaping & Operations

import tensorflow as tf

# Step 1: Create a random tensor of shape (4, 6)
tensor = tf.random.uniform(shape=(4, 6), minval=0, maxval=10, dtype=tf.float32)
print("Original Tensor:")
print(tensor)

# Step 2: Find rank and shape
rank = tf.rank(tensor)
shape = tf.shape(tensor)

print(f"\nTensor Rank: {rank.numpy()}")
print(f"Tensor Shape: {shape.numpy()}")

# Step 3: Reshape and transpose
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])

print(f"\nReshaped Tensor Shape: {reshaped_tensor.shape}")
print(f"Transposed Tensor Shape: {transposed_tensor.shape}")

# Step 4: Broadcast a smaller tensor
smaller_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0]])  # Shape (1, 4)

# Broadcasting and adding
broadcasted_add = transposed_tensor + smaller_tensor

print(f"\nResult after Broadcasting and Adding:")
print(broadcasted_add)

# Step 5: Explain how broadcasting works in TensorFlow.
#Broadcasting in TensorFlow:
#•	Broadcasting allows TensorFlow to perform operations on tensors of different shapes by automatically expanding dimensions to match.
#•	In our example:
#o	The tensor has shape (3, 2, 4) and the smaller tensor has shape (1, 4).
#o	TensorFlow aligns dimensions from the right:
#	(3, 2, 4)
#	( 1, 4) → expanded as (1, 1, 4) to match.
#•	TensorFlow "stretches" the smaller tensor along the missing dimensions, effectively replicating its values for each row.

#2.Loss Functions & Hyperparameter Tuning

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# True values (y_true) for regression and classification
y_true_reg = tf.constant([2.0, 3.0, 4.0], dtype=tf.float32)  # Regression
y_true_class = tf.constant([1, 0, 0], dtype=tf.float32)      # Classification (one-hot)

# Initial model predictions (y_pred)
y_pred_reg = tf.constant([2.5, 2.8, 3.9], dtype=tf.float32)
y_pred_class = tf.constant([0.7, 0.2, 0.1], dtype=tf.float32)

# Mean Squared Error (MSE)
mse = tf.keras.losses.MeanSquaredError()
mse_loss = mse(y_true_reg, y_pred_reg).numpy()

# Categorical Cross-Entropy (CCE)
cce = tf.keras.losses.CategoricalCrossentropy()
cce_loss = cce(y_true_class, y_pred_class).numpy()

print(f"Initial MSE Loss: {mse_loss:.4f}")
print(f"Initial CCE Loss: {cce_loss:.4f}")

# Slightly modify predictions
y_pred_reg_modified = tf.constant([3.0, 3.5, 4.5], dtype=tf.float32)
y_pred_class_modified = tf.constant([0.6, 0.3, 0.1], dtype=tf.float32)

# Recompute losses
mse_loss_mod = mse(y_true_reg, y_pred_reg_modified).numpy()
cce_loss_mod = cce(y_true_class, y_pred_class_modified).numpy()

print(f"\nModified MSE Loss: {mse_loss_mod:.4f}")
print(f"Modified CCE Loss: {cce_loss_mod:.4f}")

# Bar Chart Comparison
loss_names = ['MSE Loss (Initial)', 'CCE Loss (Initial)', 'MSE Loss (Modified)', 'CCE Loss (Modified)']
loss_values = [mse_loss, cce_loss, mse_loss_mod, cce_loss_mod]

plt.figure(figsize=(8, 6))
plt.bar(loss_names, loss_values, color=['skyblue', 'salmon', 'blue', 'red'])
plt.ylabel("Loss Value")
plt.title("Loss Function Comparison")
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


#3.Train a Neural Network and Log to TensorBoard

import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add channel dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoard log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])

%load_ext tensorboard
%tensorboard --logdir logs/fit


