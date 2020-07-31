import tensorflow as tf
import os
import cnn

models = [cnn.Cnn()]
x_train, y_train, x_test, y_test = cnn.data_processing(dataset='cifar10')

for model in models:

    print('model:%s' % str(model))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    checkpoint_path = 'checkpoint/cifar10-%s.ckpt' % str(model)
    if os.path.exists(checkpoint_path + '.index'):
        print('------------------------load the model------------------------')
        model.load_weights(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    history = model.fit(x_train, y_train,
                        batch_size=64, epochs=50,
                        validation_data=(x_test, y_test),
                        validation_freq=1,
                        callbacks=[cp_callback])

    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    cnn.show_plt(acc, val_acc, loss, val_loss, model)