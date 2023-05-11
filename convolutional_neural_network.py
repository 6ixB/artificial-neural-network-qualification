import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data_loader import MnistDataloader


# Function to generate batches from data and labels
def next_batch(batch_size, data, labels):
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    for i in range(0, len(data) - batch_size + 1, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield data[batch_indices], labels[batch_indices]


# Function for 2D convolution
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# Function for 2x2 max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Main function
def main():
    input_path = './data'
    training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # Load the MNIST dataset using the data loader
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Convert data and labels to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    # Normalize pixel values to the range [0, 1]
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0

    # Perform one-hot encoding on the labels
    one_hot_encoder = OneHotEncoder(sparse=False)
    y_train = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_val = one_hot_encoder.transform(y_val.reshape(-1, 1))
    y_test = one_hot_encoder.transform(y_test.reshape(-1, 1))

    # Define the input and output placeholders
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28])
    y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

    # Reshape the input placeholder for convolutional layers
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Define the weights and biases for the first convolutional layer
    w_conv1 = tf.compat.v1.get_variable('w_conv1', shape=[5, 5, 1, 32],
                                        initializer=tf.compat.v1.glorot_uniform_initializer())
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

    # Apply the first convolutional layer, followed by ReLU activation and max pooling
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Define the weights and biases for the second convolutional layer
    w_conv2 = tf.compat.v1.get_variable('w_conv2', shape=[5, 5, 32, 64],
                                        initializer=tf.compat.v1.glorot_uniform_initializer())
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

    # Apply the second convolutional layer, followed by ReLU activation and max pooling
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Flatten the output from the second convolutional layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Define the weights and biases for the fully connected layer
    w_fc1 = tf.compat.v1.get_variable('w_fc1', shape=[7 * 7 * 64, 1024],
                                      initializer=tf.compat.v1.glorot_uniform_initializer())
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))

    # Apply the fully connected layer, followed by ReLU activation
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # Define the dropout placeholder
    keep_prob = tf.compat.v1.placeholder(tf.float32)

    # Apply dropout to the output of the fully connected layer
    h_fc1_drop = tf.nn.dropout(h_fc1, rate=1 - keep_prob)

    # Define the weights and biases for the output layer
    w_fc2 = tf.compat.v1.get_variable('w_fc2', shape=[1024, 10], initializer=tf.compat.v1.glorot_uniform_initializer())
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

    # Compute the logits (output) of the network
    output = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    # Define the loss function (softmax cross-entropy) and the optimizer
    loss = tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(y, output))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss)

    # Calculate the accuracy of the network
    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Set the number of epochs and batch size
    epochs = 1
    batch_size = 128

    # Start a TensorFlow session
    with tf.compat.v1.Session() as sess:
        # Initialize the variables
        sess.run(tf.compat.v1.global_variables_initializer())

        # Training loop
        for epoch in range(epochs):
            train_losses = []
            train_accuracies = []

            # Iterate over the training data in batches
            for batch_x, batch_y in next_batch(batch_size, x_train, y_train):
                # Run a training step
                _, train_loss, train_acc = sess.run([train_op, loss, accuracy],
                                                    feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)

                # Calculate the mean loss and accuracy for the training set
            train_loss_mean = np.mean(train_losses)
            train_acc_mean = np.mean(train_accuracies)

            # Evaluate the model on the validation set
            val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: x_val, y: y_val, keep_prob: 1.0})
            print(
                f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss_mean:.4f}, Training Accuracy: {train_acc_mean:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Evaluate the model on the test set
        test_accuracy, test_predictions = sess.run([accuracy, tf.argmax(output, 1)],
                                                   feed_dict={x: x_test, y: y_test, keep_prob: 1.0})
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Select a random sample of 25 test images
        sample_indices = np.random.choice(len(x_test), size=25, replace=False)
        sample_images = x_test[sample_indices]
        sample_labels = y_test[sample_indices]
        sample_predictions = test_predictions[sample_indices]

        # Reshape the images if necessary (assuming MNIST dataset)
        sample_images = sample_images.reshape(-1, 28, 28)

        # Create a grid of subplots to display the sample images
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        fig.suptitle('Sample of 25 Test Images with Predictions')

        for i, ax in enumerate(axes.flat):
            # Display the image
            ax.imshow(sample_images[i], cmap='gray')

            # Set the title as the true label and the predicted label
            true_label = np.argmax(sample_labels[i])
            predicted_label = sample_predictions[i]
            ax.set_title(f'True: {true_label}\nPredicted: {predicted_label}')

            # Remove the axis labels
            ax.axis('off')

        # Adjust the spacing between subplots
        plt.tight_layout()

        # Show the plot
        plt.show()


if __name__ == '__main__':
    main()
