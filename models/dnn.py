from codebase import ASMoEA
from codebase import utility as util


from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

model = Sequential(
    [
        Input(shape=(3,)),
        Dense(128, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.35),
        Dense(64, activation="relu", kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.35),
        Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
        Dense(3),
    ]
)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Adam optimizer
    loss="mean_squared_error",  # Assuming a regression task with MSE loss
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=7,
    verbose=1,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1
)


def train(
    normalized_objective_vals, normalized_decision_vars, test_size=0.2, random_state=42
):
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_objective_vals,
        normalized_decision_vars,
        test_size=test_size,
        random_state=random_state,
    )

    # Train the model using the training data with EarlyStopping
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=2,  # Clean, line-by-line progress per epoch
    )

    return X_test, y_test


def predict(scaler_decision_vars, X_test, y_test):
    predicted_decision_vars = model.predict(X_test, verbose=2)

    # Handle NaNs in prediction
    predicted_decision_vars = util.NaN_handling(predicted_decision_vars)

    # Inverse transform to original scale
    denormalized_predicted_decision_vars = scaler_decision_vars.inverse_transform(
        predicted_decision_vars
    )
    denormalized_predicted_decision_vars = util.NaN_handling(
        denormalized_predicted_decision_vars
    )
    denormalized_predicted_decision_vars = util.normalize_dv(
        denormalized_predicted_decision_vars
    )

    # Compute objective values
    obj_predicted_DV = ASMoEA.combined_objective(denormalized_predicted_decision_vars)

    # Inverse transform actual y_test to original scale
    obj_test_DV = scaler_decision_vars.inverse_transform(y_test)
    obj_test_DV = ASMoEA.combined_objective(obj_test_DV)

    # Final NaN handling
    obj_predicted_DV = util.obj_NaN_handling(obj_predicted_DV)
    obj_test_DV = util.obj_NaN_handling(obj_test_DV)

    return denormalized_predicted_decision_vars, obj_predicted_DV, obj_test_DV
