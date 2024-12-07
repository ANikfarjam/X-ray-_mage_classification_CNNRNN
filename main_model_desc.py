import tensorflow as tf


best_model = tf.keras.models.load_model('Main_model_saved.h5')

print("Mode Desc:", best_model.summary())