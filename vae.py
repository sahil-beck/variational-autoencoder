import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import os
#from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

import kagglehub
import cv2

# Download the dataset from Kaggle
dataset_path = kagglehub.dataset_download("udaysankarmukherjee/furniture-image-dataset")
print(f"Dataset downloaded to: {dataset_path}")

# Assuming the dataset contains a folder with images, you can load the images like this
image_folder = os.path.join(dataset_path, 'chair_dataset')  # Adjust according to the dataset's structure
print(f"Contents of the dataset folder: {os.listdir(dataset_path)}")

# Image parameters
img_width, img_height = 128, 128
num_channels = 3

# Initialize empty lists to store images and labels
images = []
labels = []

# Image Quality Check Functions
def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_too_dark_or_bright(image, low_threshold=50, high_threshold=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < low_threshold or avg_brightness > high_threshold

# Assuming you don't have predefined labels, assign a dummy label for all images
label = 0  # Since you have only chair images, all labels are '0'

# Loop over all JPEG images in the folder
for img_name in os.listdir(image_folder):
    if img_name.endswith(".jpg") or img_name.endswith(".jpeg"):
        img_path = os.path.join(image_folder, img_name)
        
        # Load and process image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping invalid image file: {img_name}")
            continue
            
        # Resize image to a fixed size
        img = cv2.resize(img, (img_width, img_height))
        
        # Filter out low-quality images
        if is_blurry(img):
            print(f"Image '{img_name}' is blurry. Skipping.")
            continue
        if is_too_dark_or_bright(img):
            print(f"Image '{img_name}' is too dark or bright. Skipping.")
            continue

        # Convert image from BGR (OpenCV) to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize image (scale pixel values to [0, 1])
        img = img / 255.0
        
        # Append the image data and label to the lists
        images.append(img)
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# One-hot encode the labels if you have more than one class
labels = to_categorical(labels)

# Split into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f"Number of images after filtering: {len(images)}")


# The data is now ready for use with your model
input_shape = (img_height, img_width, num_channels)

# Print the shapes to verify the data
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

#View a few images
plt.figure(1)
plt.subplot(221)
plt.imshow(x_train[10][:,:,0])

plt.subplot(222)
plt.imshow(x_train[40][:,:,0])

plt.subplot(223)
plt.imshow(x_train[100][:,:,0])

plt.subplot(224)
plt.imshow(x_train[200][:,:,0])
plt.show()


# BUILD THE MODEL

# # ================= #############
# # Encoder
#Let us define 4 conv2D, flatten and then dense
# # ================= ############

latent_dim = 2 # Number of latent dim parameters

input_img = Input(shape=input_shape, name='encoder_input')
x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = Conv2D(64, 3, padding='same', activation='relu',strides=(2, 2))(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)

conv_shape = K.int_shape(x) #Shape of conv to be provided to decoder
#Flatten
x = Flatten()(x)
x = Dense(32, activation='relu')(x)

# Two outputs, for latent mean and log variance (std. dev.)
#Use these to sample random variables in latent space to which inputs are mapped. 
z_mu = Dense(latent_dim, name='latent_mu')(x)   #Mean values of encoded input
z_sigma = Dense(latent_dim, name='latent_sigma')(x)  #Std dev. (variance) of encoded input

#REPARAMETERIZATION TRICK
# Define sampling function to sample from the distribution
# Reparameterize sample based on the process defined by Gunderson and Huang
# into the shape of: mu + sigma squared x eps
#This is to allow gradient descent to allow for gradient estimation accurately. 
def sample_z(args):
  z_mu, z_sigma = args
  eps = K.random_normal(shape=(K.shape(z_mu)[0], K.int_shape(z_mu)[1]))
  return z_mu + K.exp(z_sigma / 2) * eps

# sample vector from the latent distribution
# z is the labda custom layer we are adding for gradient descent calculations
  # using mu and variance (sigma)
z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([z_mu, z_sigma])

#Z (lambda layer) will be the last layer in the encoder.
# Define and summarize encoder model.
encoder = Model(input_img, [z_mu, z_sigma, z], name='encoder')
print(encoder.summary())

# ================= ###########
# Decoder
#
# ================= #################

# decoder takes the latent vector as input
decoder_input = Input(shape=(latent_dim, ), name='decoder_input')

# Need to start with a shape that can be remapped to original image shape as
#we want our final utput to be same shape original input.
#So, add dense layer with dimensions that can be reshaped to desired output shape
x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(decoder_input)
# reshape to the shape of last conv. layer in the encoder, so we can 
x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
# upscale (conv2D transpose) back to original shape
# use Conv2DTranspose to reverse the conv layers defined in the encoder
x = Conv2DTranspose(32, 3, padding='same', activation='relu',strides=(2, 2))(x)
#Can add more conv2DTranspose layers, if desired. 
#Using sigmoid activation
x = Conv2DTranspose(num_channels, 3, padding='same', activation='sigmoid', name='decoder_output')(x)

# Define and summarize decoder model
decoder = Model(decoder_input, x, name='decoder')
decoder.summary()

# apply the decoder to the latent sample 
z_decoded = decoder(z)


# =========================
#Define custom loss
#VAE is trained using two loss functions reconstruction loss and KL divergence
#Let us add a class to define a custom layer with loss
# =============================================================================
# class CustomLayer(tf.keras.layers.Layer):
# 
#     def vae_loss(self, x, z_decoded):
#         x = K.flatten(x)
#         z_decoded = K.flatten(z_decoded)
#         
#         # Reconstruction loss (as we used sigmoid activation we can use binarycrossentropy)
#         recon_loss = tf.keras.metrics.binary_crossentropy(x, z_decoded)
#         
#         # KL divergence
#         kl_loss = -5e-4 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
#         return K.mean(recon_loss + kl_loss)
# 
#     # add custom loss to the class
#     def call(self, inputs):
#         x = inputs[0]
#         z_decoded = inputs[1]
#         loss = self.vae_loss(x, z_decoded)
#         self.add_loss(loss, inputs=inputs)
#         return x
# =============================================================================
# Define custom loss
class CustomLayer(tf.keras.layers.Layer):
    def vae_loss(self, x, z_decoded, z_mu, z_sigma):
        # Reconstruction loss
        recon_loss = tf.keras.metrics.binary_crossentropy(tf.keras.backend.flatten(x), tf.keras.backend.flatten(z_decoded))
        
        # KL divergence
        kl_loss = -5e-4 * tf.keras.backend.mean(1 + z_sigma - tf.keras.backend.square(z_mu) - tf.keras.backend.exp(z_sigma), axis=-1)
        return tf.keras.backend.mean(recon_loss + kl_loss)
    
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        z_mu = inputs[2]
        z_sigma = inputs[3]
        loss = self.vae_loss(x, z_decoded, z_mu, z_sigma)
        self.add_loss(loss)  # Removed `inputs=inputs`
        return x

# Apply the custom loss to the input images and the decoded latent distribution sample
y = CustomLayer()([input_img, z_decoded, z_mu, z_sigma])


# y is basically the original image after encoding input img to mu, sigma, z
# and decoding sampled z values.
#This will be used as output for vae

# =================
# VAE 
# =================
vae = Model(input_img, y, name='vae')

# Compile VAE
vae.compile(optimizer='adam', loss=None)
vae.summary()

# Train autoencoder
#vae.fit(x_train, None, epochs = 10, batch_size = 32, validation_split = 0.2)
# Train autoencoder and store the history of loss
history = vae.fit(x_train, x_train, epochs=10, batch_size=32, validation_split=0.2)

#Plot the training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('VAE Loss Function During Training')
plt.legend()
plt.show()



# =================
# Visualize results
# =================
#Visualize inputs mapped to the Latent space
#Remember that we have encoded inputs to latent space dimension = 2. 
#Extract z_mu --> first parameter in the result of encoder prediction representing mean

# Step 1: Encode images into the latent space
# Use the encoder to predict latent space means (z_mu)
z_mu, _, _ = encoder.predict(x_test)

# Step 2: Apply K-means clustering to the latent representations
# Let's say we want to create 5 clusters
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(z_mu)

# Step 3: Visualize clusters in 2D space
# Using TSNE for better visualization in 2D if latent_dim > 2
if latent_dim > 2:
    z_mu_2d = TSNE(n_components=2).fit_transform(z_mu)
else:
    z_mu_2d = z_mu  # If latent_dim is already 2, no need for dimensionality reduction

# Plot the clusters
plt.figure(figsize=(10, 10))
plt.scatter(z_mu_2d[:, 0], z_mu_2d[:, 1], c=clusters, cmap='viridis', marker='o')
plt.colorbar(label='Cluster')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('K-means Clustering on Latent Space')
plt.show()

# Step 4: Generate and visualize images from each cluster center
# Get the centroid of each cluster in the latent space and decode it
plt.figure(figsize=(10, 10))
for cluster_idx in range(num_clusters):
    # Get the centroid of the cluster
    cluster_center = kmeans.cluster_centers_[cluster_idx].reshape(1, -1)

    # Decode the centroid to generate an image
    decoded_image = decoder.predict(cluster_center)
    decoded_image = decoded_image.reshape(img_width, img_height, num_channels)

    # Plot the generated image
    plt.subplot(1, num_clusters, cluster_idx + 1)
    plt.imshow(decoded_image)
    plt.axis('off')
    plt.title(f'Cluster {cluster_idx}')
plt.show()



mu, _, _ = encoder.predict(x_test)
#Plot dim1 and dim2 for mu
plt.figure(figsize=(5, 5))
plt.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap='brg')
plt.xlabel('dim 1')
plt.ylabel('dim 2')
plt.colorbar()
plt.show()


# Visualize images
#Single decoded image with random input latent vector (of size 1x2)
#Latent space range is about -5 to 5 so pick random values within this range
#Try starting with -1, 1 and slowly go up to -1.5,1.5 and see how it morphs from 
#one image to the other.
sample_vector = np.array([[-1, 2]])
decoded_example = decoder.predict(sample_vector)
print(f"Decoded example shape: {decoded_example.shape}")

# Remove the batch dimension and handle the color channels
decoded_example_reshaped = decoded_example[0]  # Shape should now be (128, 128, 3)

# Display the image with all color channels
plt.imshow(decoded_example_reshaped)
plt.axis('off')  # Optional: turn off axis for a cleaner display
plt.show()


#Let us automate this process by generating multiple images and plotting
#Use decoder to generate images by tweaking latent variables from the latent space
#Create a grid of defined size with zeros. 
#Take sample from some defined linear space. In this example range [-4, 4]
#Feed it to the decoder and update zeros in the figure with output.


n = 15  # generate 15x15 digits
figure = np.zeros((img_width * n, img_height * n, num_channels))

#Create a Grid of latent variables, to be provided as inputs to decoder.predict
#Creating vectors within range -5 to 5 as that seems to be the range in latent space
grid_x = np.linspace(-5, 5, n)
grid_y = np.linspace(-5, 5, n)[::-1]


# decoder for each square in the grid
for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        print(x_decoded.shape)
        digit = x_decoded[0].reshape(img_width, img_height, num_channels)
        figure[i * img_width: (i + 1) * img_width,
               j * img_height: (j + 1) * img_height] = digit

plt.figure(figsize=(10, 10))
#Reshape for visualization
fig_shape = np.shape(figure)
figure = figure.reshape((fig_shape[0], fig_shape[1],num_channels))

plt.imshow(figure, cmap='gnuplot2')
plt.show()
print (x_decoded.shape)

# Generate random latent vectors
random_latents = np.random.normal(size=(5, latent_dim))  # 5 random samples
generated_images = decoder.predict(random_latents)

# Plot original and generated images side by side for comparison
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i])
    plt.axis('off')
    plt.title("Original")

    plt.subplot(2, 5, i+6)
    plt.imshow(generated_images[i])
    plt.axis('off')
    plt.title("Generated")
plt.show()



import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Assuming `vae` is your trained VAE model, `encoder` is the encoder part, and `decoder` is the decoder part.
# `x_test` is your test dataset and `y_test` contains labels.

# Number of test images to use
num_images = x_test.shape[0]  # Use all available test images (495 in this case)

# Check the shape of x_test to understand the dimensions
print(f"Shape of x_test: {x_test.shape}")

# Flatten the input images for PCA (shape: num_images, height * width * channels)
flat_inputs = x_test.reshape(num_images, -1)  # Flatten each image into a vector
print(f"Shape of flattened input images: {flat_inputs.shape}")

# Get latent variables (mean of the distribution) for test images
z_mu, _, _ = encoder.predict(x_test)  # Predict latent space representation for all images

# Reconstruct the images using the decoder
reconstructed_images = decoder.predict(z_mu)

# Flatten the reconstructed images for PCA
flat_reconstructions = reconstructed_images.reshape(reconstructed_images.shape[0], -1)
print(f"Shape of flattened reconstructed images: {flat_reconstructions.shape}")

# Apply PCA to the flattened input images
pca_input = PCA(n_components=2)
input_pca = pca_input.fit_transform(flat_inputs)

# Apply PCA to the flattened reconstructed images
pca_decoder = PCA(n_components=2)
recon_pca = pca_decoder.fit_transform(flat_reconstructions)

# Plot PCA of input images and decoder output (reconstructed images)
plt.figure(figsize=(10, 8))

# Plot PCA of input images
plt.scatter(input_pca[:, 0], input_pca[:, 1], c='blue', label="Input Images (PCA)", alpha=0.6, s=50, marker='o')

# Plot PCA of decoder output (reconstructed images)
plt.scatter(recon_pca[:, 0], recon_pca[:, 1], c='red', label="Reconstructed Images (PCA)", alpha=0.6, s=50, marker='^')

# Add labels, title, and legend
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Input Images and Reconstructed Images")
plt.legend(loc="best")
plt.grid(True)
plt.show()