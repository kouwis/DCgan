from keras.layers import UpSampling2D, Dense, Conv2D, BatchNormalization, Dropout, Flatten, Reshape, Activation
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from PIL import Image
from tensorflow.keras.losses import BinaryCrossentropy as cross_entropy
from matplotlib import pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
import imageio

GENERATE_SQUARE = 96
IMG_CHANNEL = 3
NOISE = 100
BATCH_SIZE = 32
BUFFER_DATA = 20000
EPOCHS = 200
IMG_SHAPE = (GENERATE_SQUARE, GENERATE_SQUARE, IMG_CHANNEL)
PATH = "Anime"
train = []
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 16

for filename in tqdm(os.listdir(PATH)):
    img_ = Image.open(PATH+ "/"+ filename).resize((GENERATE_SQUARE, GENERATE_SQUARE),Image.ANTIALIAS)
    train.append(np.asarray(img_))
train = np.reshape(train,(-1, GENERATE_SQUARE, GENERATE_SQUARE, IMG_CHANNEL))
train = train.astype(np.float32)
train = train / 127.5 - 1.
train_ = tf.data.Dataset.from_tensor_slices(train).shuffle(BUFFER_DATA).batch(BATCH_SIZE)

def build_generator():

    model = Sequential()
    model.add(Dense(4*4*256, input_dim= NOISE, activation="relu"))
    model.add(Reshape((4,4,256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=(3,3), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D(size=(3,3)))
    model.add(Conv2D(128, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=(3, 3), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))


    model.add(Conv2D(IMG_CHANNEL, kernel_size=(3,3), padding="same"))
    model.add(Activation("tanh"))
    print(model.summary())

    return model

def build_discriminator():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), strides=2, padding="same", input_shape=IMG_SHAPE))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3,3), strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=(3, 3), strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    print(model.summary())

    return model

def save_images(cnt):

    fixed_seed = np.random.normal(0, 1, (4 * 7, NOISE))

    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (GENERATE_SQUARE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (GENERATE_SQUARE + PREVIEW_MARGIN)), 3),
        255, dtype=np.uint8)

    generated_images = generator.predict(fixed_seed)

    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (GENERATE_SQUARE + 16) + PREVIEW_MARGIN
            c = col * (GENERATE_SQUARE + 16) + PREVIEW_MARGIN
            image_array[r:r + GENERATE_SQUARE, c:c + GENERATE_SQUARE] \
                = generated_images[image_count] * 255
            image_count += 1

    filename = os.path.join("gif", f"train-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)

generator = build_generator()
noise_img = tf.random.normal([1, NOISE])
generate_imgs = generator(noise_img)
plt.imshow(generate_imgs[0, :, :, 0])

discriminator = build_discriminator()
decision = discriminator(generate_imgs)
print(decision)

cross_entropy = cross_entropy()

generator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)

def discriminator_loss(real_imgs, fake_imgs):

    real_loss = cross_entropy(tf.ones_like(real_imgs), real_imgs)
    fake_loss = cross_entropy(tf.zeros_like(fake_imgs), fake_imgs)
    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(fake_imgs):

    return cross_entropy(tf.ones_like(fake_imgs), fake_imgs)


def sub_train(images):

    noise = tf.random.normal([BATCH_SIZE, NOISE])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        generated_imgs = generator(noise, training= True)

        real_imgs = discriminator(images, training= True)
        fake_imgs = discriminator(generated_imgs, training= True)

        dis_loss = discriminator_loss(real_imgs, fake_imgs)
        gen_loss = generator_loss(fake_imgs)

        gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradient_dis = dis_tape.gradient(dis_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradient_gen, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradient_dis, discriminator.trainable_variables))

    return gen_loss, dis_loss

def train(dataset, epochs):

    for epoch in range(epochs):

        gen_loss_list = []
        dis_loss_list = []

        for image_batch in dataset:

            t = sub_train(image_batch)
            gen_loss_list.append(t[0])
            dis_loss_list.append(t[1])

        g_loss = sum(gen_loss_list) / len(gen_loss_list)
        d_loss = sum(dis_loss_list) / len(dis_loss_list)

        print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss}')
        save_images(epoch)

train(train_, EPOCHS)

folder = "gif"

images = []

for filename in tqdm(os.listdir(folder)):
    file = folder+ "/"+ filename
    img = imageio.imread(file)
    images.append(img)

imageio.mimwrite("DCgan.gif", images, fps= 3)

generator.save('generator_model.h5')