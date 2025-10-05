from model.test import load_model
import model.training as model
import fine_tuning

# print(model.model)

# model.model_training(epochs=50)
# load_model()
fine_tuning.fine_tune_model(epochs=200)