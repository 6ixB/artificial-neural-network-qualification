import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data_loader import MnistDataloader


def next_batch(batch_size, data, labels):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    for i in range(0, len(data) - batch_size + 1, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield data[batch_indices], labels[batch_indices]


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    input_path = './data'
    training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0

    one_hot_encoder = OneHotEncoder(sparse=False)
    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_val = one_hot_encoder.transform(y_val.reshape(-1, 1))
    y_test = one_hot_encoder.transform(y_test.reshape(-1, 1))

    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28])
    y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    w_conv1 = tf.compat.v1.get_variable('w_conv1', shape=[5, 5, 1, 32],
                                        initializer=tf.compat.v1.glorot_uniform_initializer())
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = tf.compat.v1.get_variable('w_conv2', shape=[5, 5, 32, 64],
                                        initializer=tf.compat.v1.glorot_uniform_initializer())
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    w_fc1 = tf.compat.v1.get_variable('w_fc1', shape=[7 * 7 * 64, 1024],
                                      initializer=tf.compat.v1.glorot_uniform_initializer())
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    keep_prob = tf.compat.v1.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, rate=1 - keep_prob)

    w_fc2 = tf.compat.v1.get_variable('w_fc2', shape=[1024, 10], initializer=tf.compat.v1.glorot_uniform_initializer())
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    output = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    loss = tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(y, output))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    epochs = 10
    batch_size = 128

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(epochs):
            for batch_x, batch_y in next_batch(batch_size, x_train, y_train):
                sess.run(train_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: x_val, y: y_val, keep_prob: 1.0})
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
        print(f"Test Accuracy: {test_accuracy:.4f}")

        num_samples = 25
        grid_size = 5
        sample_indices = np.random.choice(len(x_test), num_samples)
        sample_images = x_test[sample_indices]
        sample_labels = y_test[sample_indices]

        predictions = sess.run(output, feed_dict={x: sample_images})
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(sample_labels, axis=1)

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 4))

        for i in range(grid_size):
            for j in range(grid_size):
                index = i * grid_size + j
                axes[i, j].imshow(sample_images[index].reshape(28, 28), cmap='gray')
                axes[i, j].set_title(f'Predicted: {predicted_labels[index]}, True: {true_labels[index]}')
                axes[i, j].axis('off')

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
