from codebase import constants as const
from codebase import ASMoEA
from codebase import utility as util

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import pandas as pd
import random
import numpy as np

# ------------------------------ Wide & Deep Model ------------------------------

input_layer = Input(shape=(3,), name="Input")

# Wide path: directly connect input to output
wide_output = Dense(3, activation=None, name="Wide_Part")(input_layer)

# Deep path: 3 hidden layers
x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.35)(x)

x = Dense(64, activation="relu", kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.35)(x)

x = Dense(32, activation="relu", kernel_regularizer=l2(0.001))(x)

# Combine wide and deep parts
combined_output = Concatenate()([x, wide_output])
final_output = Dense(3, name="Final_Output")(combined_output)

model = Model(inputs=input_layer, outputs=final_output)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mean_squared_error",
)

# ------------------------------ Callbacks ------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=0.0001, patience=7, verbose=1, restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1
)

# ------------------------------ Training & Prediction ------------------------------

def train(normalized_objective_vals, normalized_decision_vars, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_objective_vals,
        normalized_decision_vars,
        test_size=test_size,
        random_state=random_state,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=2,
    )

    return X_test, y_test


def predict(scaler_decision_vars, X_test, y_test):
    predicted_decision_vars = model.predict(X_test, verbose=2)

    predicted_decision_vars = util.NaN_handling(predicted_decision_vars)

    denormalized_predicted_decision_vars = scaler_decision_vars.inverse_transform(predicted_decision_vars)
    denormalized_predicted_decision_vars = util.NaN_handling(denormalized_predicted_decision_vars)
    denormalized_predicted_decision_vars = util.normalize_dv(denormalized_predicted_decision_vars)

    obj_predicted_DV = ASMoEA.combined_objective(denormalized_predicted_decision_vars)

    obj_test_DV = scaler_decision_vars.inverse_transform(y_test)
    obj_test_DV = ASMoEA.combined_objective(obj_test_DV)

    obj_predicted_DV = util.obj_NaN_handling(obj_predicted_DV)
    obj_test_DV = util.obj_NaN_handling(obj_test_DV)

    return denormalized_predicted_decision_vars, obj_predicted_DV, obj_test_DV
