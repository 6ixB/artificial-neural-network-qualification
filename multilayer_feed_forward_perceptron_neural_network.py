import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data_loader import MnistDataloader


def next_batch(batch_size, data, labels):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    for i in range(0, len(data) - batch_size + 1, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield data[batch_indices], labels[batch_indices]


def main():
    input_path = 'data'
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

    n_input = x_train.shape[1] * x_train.shape[2]
    n_hidden1 = 32
    n_hidden2 = 32
    n_hidden3 = 32
    n_hidden4 = 32
    n_output = 10

    x_train = np.array(x_train).reshape(-1, n_input)
    y_train = np.array(y_train)

    x_val = np.array(x_val).reshape(-1, n_input)
    y_val = np.array(y_val)

    x_test = np.array(x_test).reshape(-1, n_input)
    y_test = np.array(y_test)

    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0

    one_hot_encoder = OneHotEncoder(sparse=False)
    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_val = one_hot_encoder.transform(y_val.reshape(-1, 1))
    y_test = one_hot_encoder.transform(y_test.reshape(-1, 1))

    x = tf.compat.v1.placeholder(tf.float32, shape=(None, n_input))
    y = tf.compat.v1.placeholder(tf.float32, shape=(None, n_output))

    initializer = tf.initializers.variance_scaling(mode='fan_avg', distribution='uniform')

    weights = {
        'hidden1': tf.compat.v1.Variable(initializer([n_input, n_hidden1])),
        'hidden2': tf.compat.v1.Variable(initializer([n_hidden1, n_hidden2])),
        'hidden3': tf.compat.v1.Variable(initializer([n_hidden2, n_hidden3])),
        'hidden4': tf.compat.v1.Variable(initializer([n_hidden3, n_hidden4])),
        'output': tf.compat.v1.Variable(initializer([n_hidden4, n_output]))
    }

    biases = {
        'hidden1': tf.compat.v1.Variable(tf.random.normal([n_hidden1])),
        'hidden2': tf.compat.v1.Variable(tf.random.normal([n_hidden2])),
        'hidden3': tf.compat.v1.Variable(tf.random.normal([n_hidden3])),
        'hidden4': tf.compat.v1.Variable(tf.random.normal([n_hidden4])),
        'output': tf.compat.v1.Variable(tf.random.normal([n_output]))
    }

    hidden1 = tf.nn.relu(tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1']))
    hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, weights['hidden2']), biases['hidden2']))
    hidden3 = tf.nn.relu(tf.add(tf.matmul(hidden2, weights['hidden3']), biases['hidden3']))
    hidden4 = tf.nn.relu(tf.add(tf.matmul(hidden3, weights['hidden4']), biases['hidden4']))
    output = tf.nn.softmax(tf.add(tf.matmul(hidden4, weights['output']), biases['output']))

    loss = tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(y, output))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    epochs = 50
    batch_size = 128

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(epochs):
            train_losses = []
            train_accuracies = []
            for batch_x, batch_y in next_batch(batch_size, x_train, y_train):
                _, train_loss, train_acc = sess.run([train_op, loss, accuracy], feed_dict={x: batch_x, y: batch_y})
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)

            train_loss_mean = np.mean(train_losses)
            train_acc_mean = np.mean(train_accuracies)
            val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: x_val, y: y_val})
            print(
                f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss_mean:.4f}, Training Accuracy: {train_acc_mean:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
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
