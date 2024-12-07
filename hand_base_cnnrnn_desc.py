# Author: Ashkan Nikfarjam
#in this script we are using the hand base model to:
# Label hands and forearms
# label wrist

import tensorflow as tf


best_model = tf.keras.models.load_model('hand_best_base_model.h5')

print("Mode Desc:", best_model.summery())