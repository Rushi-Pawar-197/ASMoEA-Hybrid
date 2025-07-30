from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statistics
import random
import numpy as np
import matplotlib.pyplot as plt

from codebase import constants as const


def eval_dnn_metrics(obj_test_DV, obj_predicted_DV):
    # Reshape the inputs to 2D arrays with shape (n_samples, n_outputs)
    obj_test_DV = np.array(obj_test_DV).reshape(-1, 1)  # Reshape to (n_samples, 1)
    obj_predicted_DV = np.array(obj_predicted_DV).reshape(
        -1, 1
    )  # Reshape to (n_samples, 1)

    # Mean Squared Error (MSE)
    mse = mean_squared_error(obj_test_DV, obj_predicted_DV)

    # R-squared score
    r2 = r2_score(obj_test_DV, obj_predicted_DV)

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(obj_test_DV, obj_predicted_DV)

    return mse, r2, mae

def ga_res_visualize(hv_per_gen, igd_per_gen):
    # Plot the performance metrics over iterations
    plt.figure(figsize=(14, 14))

    plt.subplot(1, 2, 1)
    plt.plot(hv_per_gen, marker="o", markersize=2)
    plt.title("Hypervolume over evolution")
    plt.xlabel("Generations")
    plt.ylabel("HV")

    plt.subplot(1, 2, 2)
    plt.plot(igd_per_gen, marker="o", markersize=2)
    plt.title("IGD over evolution")
    plt.xlabel("Generations")
    plt.ylabel("IGD")

    plt.tight_layout()
    plt.show()

def hypervolume(S, ref_points):
    if len(ref_points) == 0:
        print("No reference points provided.")
        return int(0)
    hv_values = []
    for ref_point in ref_points:
        hv = 0
        for s in S:
            volume = np.prod([ref_point[i] - s[i] for i in range(len(ref_point))])
            hv += volume
        hv_values.append(float(abs(hv)))

    avg_hv = statistics.mean(hv_values)

    min_val = 0
    max_val = 50000

    norm_HV = (avg_hv - min_val) / (max_val - min_val)

    return avg_hv, norm_HV


def compute_igd(obtained_front):
    total_distance = 0.0
    True_PF = const.TRUE_PF
    for v in True_PF:
        min_dist = float("inf")
        for u in obtained_front:
            dist = np.linalg.norm(np.array(u) - np.array(v))
            if dist < min_dist:
                min_dist = dist
        total_distance += min_dist

    igd_val = total_distance / len(True_PF)

    min_val = 0
    max_val = 50

    norm_IGD = (igd_val - min_val) / (max_val - min_val)

    return igd_val, norm_IGD


def DNN_res_visualize(mse_list, r2_list, mae_list):
    # Plot the performance metrics over iterations
    plt.figure(figsize=(14, 14))

    plt.subplot(1, 3, 1)
    plt.plot(mse_list, marker="o")
    plt.title("MSE over Training Bursts")
    plt.xlabel("Training Bursts")
    plt.ylabel("MSE")

    plt.subplot(1, 3, 2)
    plt.plot(r2_list, marker="o")
    plt.title("R² Score over Training Bursts")
    plt.xlabel("Training Bursts")
    plt.ylabel("R² Score")

    plt.subplot(1, 3, 3)
    plt.plot(mae_list, marker="o")
    plt.title("MAE over Training Bursts")
    plt.xlabel("Training Bursts")
    plt.ylabel("MAE")

    plt.tight_layout()
    plt.show()
