import tensorflow as tf
import tensorflow_datasets as tfds
import math
import glob


dataset, dataset_info = tfds.load(name='fashion_mnist', as_supervised=True, with_info = True)
train_dataset, test_dataset = dataset['train'], dataset['test']
num_train, num_test = dataset_info.splits['train'].num_examples, dataset_info.splits['test'].num_examples

class_name = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

file_name = './model_weights'

if len(glob.glob(file_name + '*')) > 0:
    model.load_weights(file_name)
    print("Loaded weights from file")
else:
    cp_callback = tf.keras.callbacks.ModelCheckpoint(file_name, save_weights_only=True, verbose=1)

    train_dataset = train_dataset.repeat().shuffle(num_train).batch(32)

    model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train/32), callbacks=[cp_callback])
    print("Model fit finished")

test_dataset = test_dataset.batch(32)
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test/32))
print('Accuracy on test dataset :', test_accuracy)
