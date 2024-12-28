import keras

model = keras.models.load_model('./models/gru.keras')
print(model)