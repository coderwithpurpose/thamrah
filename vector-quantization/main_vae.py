"""CS446 2018 Spring MP10.
   Implementation of a variational autoencoder for image generation.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from vae import VariationalAutoencoder
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def train(model, mnist_dataset, learning_rate=0.0005, batch_size=16, num_steps=1000000):
    """Implements the training loop of mini-batch gradient descent.

    Performs mini-batch gradient descent with the indicated batch_size and
    learning_rate. (**Do not modify this function.**)

    Args:
        model(VariationalAutoencoder): Initialized VAE model.
        mnist_dataset: Mnist dataset.
        learning_rate(float): Learning rate.
        batch_size(int): Batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
    for step in range(0, num_steps):
        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        model.session.run(
                model.update_op_tensor,
                feed_dict={model.x_placeholder: batch_x,
                           model.learning_rate_placeholder: learning_rate}
                )

def enc_lin_interpolate_dec(model, train_mnist):
    """
    This function encodes sampled images to the latent space after training. After this is done, we find the linear inter-
    -polation between each pair (seven of them). We then decode each of these linear interpolates and plot them in a 10x9
    grid.
    :param: model: A trained Variational Autoencoder model that can decode and encode
    :param: train_mnist:
    :return:
    """

    one_hot_mnist = train_mnist.images
    normal_mnist = read_data_sets('MNIST_data')
    normal_mnist_train_labels = normal_mnist.train.labels

    # First, we pick 10 pairs of different digits
    digits = np.random.randint(low=0, high=10, size=20).reshape(10, 2)
    images = np.empty((10*28, 9*28))

    for pair_idx in range(10):
        pair = digits[pair_idx, :]

        # We need two images that correspond to the random digit pair that we extracted
        indx1 = np.where(normal_mnist_train_labels == pair[0])
        indx2 = np.where(normal_mnist_train_labels == pair[1])
        pair_1_image, pair_2_image = normal_mnist.train.images[indx1[0][0], :][:, np.newaxis].T, normal_mnist.train.images[indx2[0][0], :][:, np.newaxis].T

        l1 = model.session.run(model.z, feed_dict={model.x_placeholder:pair_1_image})
        l2 = model.session.run(model.z, feed_dict={model.x_placeholder:pair_2_image})

        l1_interp, l2_interp = np.linspace(start=l1[0, 0], stop=l2[0, 0], num=9), np.linspace(start=l1[0, 1], stop=l2[0, 1], num=9)
        for i in range(9):
            dec_img = model.generate_samples([[l1_interp[i], l2_interp[i]]])[0, :].reshape(28, 28)
            # if i != 0 and i != 8:
            images[pair_idx*28:(pair_idx + 1)*28, i*28:(i + 1)*28] = dec_img
            # elif i == 0:
            #     images[pair_idx*28:(pair_idx + 1)*28, i*28:(i + 1)*28] = pair_1_image[0, :].reshape(28, 28)
            # else:
            #     images[pair_idx*28:(pair_idx + 1)*28, i*28:(i + 1)*28] = pair_2_image[0, :].reshape(28, 28)
    plt.imsave('inter_images_diff_1M.png', images, cmap="gray")
    return images 

def enc_lin_interpolate_dec_same(model, train_mnist):
    # """
    # This function encodes sampled images to the latent space after training. After this is done, we find the linear inter-
    # -polation between each pair (seven of them). We then decode each of these linear interpolates and plot them in a 10x9
    # grid.
    # :param: model: A trained Variational Autoencoder model that can decode and encode
    # :param: train_mnist:
    # :return:
    # """

    one_hot_mnist = train_mnist.images
    normal_mnist = read_data_sets('MNIST_data')
    normal_mnist_train_labels = normal_mnist.train.labels

    # First, we pick 10 pairs of different digits
    digits = np.random.randint(low=0, high=10, size=10) 
    print("digit chosen is ",digits)
    images = np.empty((10*28, 9*28))

    for pair_idx in range(10):
        pair = digits[pair_idx]

        # We need two images that correspond to the random digit pair that we extracted
        indx1 = np.where(normal_mnist_train_labels == pair)
        indx2 = np.where(normal_mnist_train_labels == pair)
        pair_1_image, pair_2_image = normal_mnist.train.images[indx1[0][0], :][:, np.newaxis].T, normal_mnist.train.images[indx2[0][0], :][:, np.newaxis].T

        l1 = model.session.run(model.z, feed_dict={model.x_placeholder:pair_1_image})
        l2 = model.session.run(model.z, feed_dict={model.x_placeholder:pair_2_image})

        l1_interp, l2_interp = np.linspace(start=l1[0, 0], stop=l2[0, 0], num=9), np.linspace(start=l1[0, 1], stop=l2[0, 1], num=9)
        for i in range(9):
            dec_img = model.generate_samples([[l1_interp[i], l2_interp[i]]])[0, :].reshape(28, 28)
            # if i != 0 and i != 8:
            images[pair_idx*28:(pair_idx + 1)*28, i*28:(i + 1)*28] = dec_img
            # elif i == 0:
            #     images[pair_idx*28:(pair_idx + 1)*28, i*28:(i + 1)*28] = pair_1_image[0, :].reshape(28, 28)
            # else:
            #     images[pair_idx*28:(pair_idx + 1)*28, i*28:(i + 1)*28] = pair_2_image[0, :].reshape(28, 28)
    plt.imsave('inter_images_same_1M.png', images, cmap="gray")
    return images

def main(_):
    """High level pipeline.

    This script performs the training for VAEs.
    """
    # Get dataset.
    mnist_dataset = read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = VariationalAutoencoder()

    # Start training
    train(model, mnist_dataset)

    images = enc_lin_interpolate_dec(model, mnist_dataset.train)  
    # print(np.shape(images))
    iamges_same =enc_lin_interpolate_dec_same(model, mnist_dataset.train)

    pass

    #    Plot out latent space, for +/- 3 std.
    std = 1
    x_z = np.linspace(-3*std, 3*std, 20)
    y_z = np.linspace(-3*std, 3*std, 20)

    out = np.empty((28*20, 28*20))
    for x_idx, x in enumerate(x_z):
        for y_idx, y in enumerate(y_z):
            z_mu = np.array([[y, x]])
            img = model.generate_samples(z_mu)
            out[x_idx*28:(x_idx+1)*28,
                y_idx*28:(y_idx+1)*28] = img[0].reshape(28, 28)
    plt.imsave('latent_space_vae.png', out, cmap="gray")

if __name__ == "__main__":
    tf.app.run()
