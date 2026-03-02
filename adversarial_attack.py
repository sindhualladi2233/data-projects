import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from Adversarialattack.simplecnn import SimpleCNN
from Adversarialattack.fgsm import generate_image_adversary

# Load MNIST dataset and scale the pixel values to the range [0, 1]
print("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX / 255.0
testX = testX / 255.0
# Add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
# One-hot encode our labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)
# Initialize our optimizer and model
print("[INFO] compiling model...")
opt = Adam(learning_rate=1e-3)  # Note: using learning_rate instead of lr
model = SimpleCNN.build(width=28, height=28, depth=1, classes=10)  # Provide input shape and num_classes
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# Train the simple CNN on MNIST
print("[INFO] training network...")
model.fit(trainX, trainY,
          validation_data=(testX, testY),
          batch_size=64,
          epochs=20,
          verbose=1)
# Make predictions on the testing set for the model trained on
# non-adversarial images
(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))

# Define a range of epsilon values
epsilons = [0, 0.15, 0.25, 0.35, 0.45]  # You can modify this range as needed

# Initialize an empty list to store images for all epsilon values
all_images = []

# Loop over epsilon values
for epsilon in epsilons:
    # Initialize an empty list to store images for the current epsilon value
    images = []
    # Loop over a sample of our testing images
    for i in np.random.choice(np.arange(0, len(testX)), size=(10,)):
        # Grab the current image and label
        image = testX[i]
        label = testY[i]
        # Generate an image adversary for the current image and make
        # a prediction on the adversary
        adversary = generate_image_adversary(model,
                                             image.reshape(1, 28, 28, 1), label, eps=epsilon)
        pred_orig = model.predict(image.reshape(1, 28, 28, 1)).argmax()
        pred_adv = model.predict(adversary).argmax()
        # Scale both the original image and adversary to the range
        # [0, 255] and convert them to unsigned 8-bit integers
        adversary = adversary.reshape((28, 28)) * 255
        adversary = np.clip(adversary, 0, 255).astype("uint8")
        image = image.reshape((28, 28)) * 255
        image = image.astype("uint8")
        # Add the images and their predictions to the list for the current epsilon value
        images.append((image, pred_orig, adversary, pred_adv))
    # Add the images for the current epsilon value to the list
    all_images.append(images)

# Plot the images for each epsilon value
num_epsilons = len(epsilons)
num_images_per_epsilon = 10

fig, axes = plt.subplots(num_epsilons, num_images_per_epsilon, figsize=(20, 20))

for i, epsilon_images in enumerate(all_images):
    for j, (original, pred_orig, adversarial, pred_adv) in enumerate(epsilon_images):
        stacked_image = np.hstack([original, adversarial])
        axes[i, j].imshow(stacked_image, cmap='gray')
        axes[i, j].axis('off')
        #axes[i, j].set_title(f"Org: {pred_orig} | Adv: {pred_adv}", fontsize=10)
        # Define color based on prediction match for the adversarial image
        color = 'blue' if pred_orig == pred_adv else 'red'
        # Place predictions on the images
        org_text_x = 0.1 * original.shape[1]  # Adjust the x-coordinate of the text
        org_text_y = 0.2 * original.shape[0]  # Adjust the y-coordinate of the text
        adv_text_x = original.shape[1] + 0.1 * adversarial.shape[1]  # Adjust the x-coordinate of the text for the adversarial image
        adv_text_y = 0.2 * adversarial.shape[0]  # Adjust the y-coordinate of the text for the adversarial image
        axes[i, j].text(org_text_x, org_text_y, str(pred_orig), color='blue', fontsize=10, weight='bold')
        axes[i, j].text(adv_text_x, adv_text_y, str(pred_adv), color=color, fontsize=10, weight='bold')

    axes[i, 0].set_title(f"Eps: {epsilons[i]}", fontsize=10)

plt.suptitle("Adversarial Examples for Different Epsilon Values")
plt.tight_layout()
plt.show()
