from codebase import ASMoEA
from codebase import utility as util

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ------------------------ Build ResNet for Tabular ------------------------

def build_resnet_tabular(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))

    # Input block
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)

    # Residual Block 1
    x1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x1)
    x1 = BatchNormalization()(x1)
    x = Add()([x, x1])  # Residual connection

    # Residual Block 2
    x2 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    x2 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x2)
    x2 = BatchNormalization()(x2)
    x = Add()([x, x2])  # Residual connection

    # Output head
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_dim)(x)

    model = Model(inputs, outputs)
    return model

# Initialize model
model = build_resnet_tabular(input_dim=3, output_dim=3)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="mean_squared_error"
)

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

# ------------------------ Training Function ------------------------

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

# ------------------------ Prediction Function ------------------------

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
