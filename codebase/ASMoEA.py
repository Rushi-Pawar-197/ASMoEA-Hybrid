from codebase import constants as const
from codebase import utility as util

import numpy as np
import itertools

from scipy.special import erfc


def initialize_population(pop_size, num_variables):

    # Initialize the population with random values within the bounds
    population = np.zeros((pop_size, num_variables))
    for i in range(pop_size):
        for j in range(num_variables):
            population[i, j] = np.random.uniform(const.bounds[j][0], const.bounds[j][1])

    # Round the values to 4 decimal places
    population = np.round(population, 4)

    # print("Initial Population : count = ", len(population), "\n", population)
    return population


def initialize_reference_points(num_objectives, num_divisions):

    # Generate equally spaced divisions within provided bounds for each objective
    divisions = [
        np.linspace(const.bounds[i][0], const.bounds[i][1], num_divisions)
        for i in range(num_objectives)
    ]

    # Create the Cartesian product of all divisions to form the grid
    # Example: if num_objectives = 2 and num_divisions = 3, grid will combine all (x,y) in the product of x in div1 and y in div2
    grid = list(itertools.product(*divisions))

    # Convert the grid to a NumPy array
    reference_points = np.array(grid)

    return reference_points


# Combined objective function returning individual objective values
def obj_fun(decision_variables, modulation_scheme="MPSK"):
    P_w, M, R_s = decision_variables
    M = max(2, M)  # Ensure M is at least 2

    S = P_w - const.P_L
    gamma = S / (R_s * const.N_0 * np.log2(M))
    gamma = max(gamma, 1e-10)  # Avoid non-positive or extremely small gamma values

    def Q_function(z):
        return 0.5 * erfc(z / np.sqrt(2))

    def safe_sqrt(x):
        """Returns the square root of x, ensuring non-negative input"""
        return np.sqrt(max(x, 0))

    def ber_mpsk(M, gamma):
        return (
            2
            / np.log2(M)
            * Q_function(safe_sqrt(2 * np.log2(M) * gamma * np.sin(np.pi / M)))
        )

    def ber_mqam(M, gamma):
        return (
            4
            / np.log2(M)
            * (1 - 1 / np.sqrt(M))
            * Q_function(safe_sqrt(3 * np.log2(M) * gamma / (M - 1)))
        )

    if modulation_scheme == "MPSK":
        Pei = ber_mpsk(M, gamma)
    elif modulation_scheme == "MQAM":
        Pei = ber_mqam(M, gamma)
    else:
        raise ValueError(f"Invalid modulation scheme: {modulation_scheme}")

    # Needed Bit Error Rate
    fbe = np.log10(0.5) / const.S_N * np.sum(np.log10(Pei))

    # Constants for normalization
    P_max = 25.2
    M_max = 64
    rho1, rho2, rho3 = 0.4, 0.3, 0.3

    # Power consumption
    fp = rho1 * P_w / P_max + rho2 * R_s / 10 + rho3 * np.log2(M) / np.log2(M_max)

    # Throughput efficiency
    Rc = 1
    fthr = (R_s * np.log2(M) * Rc) / (Rc * R_s * np.log2(M_max))

    # Normalize efficiency values as appropriate
    fthr = 1 - fthr

    return [fp, fbe, fthr]


def combined_objective(decision_variables_list, modulation_scheme="MPSK"):
    obj_values_list = []
    for ind in decision_variables_list:
        obj_values_list.append(obj_fun(ind, modulation_scheme))
    return obj_values_list


def RGDMatingSelection(P, ideal_points, nadir_points):

    # Initialize the mating pool
    mating_pool = []

    # Convert ideal and nadir points to numpy arrays if they aren't already
    ideal_points = np.array(ideal_points)
    nadir_points = np.array(nadir_points)

    # Extract the number of objectives from the dimensions of ideal_points
    n_obj = len(ideal_points)

    # Calculate the reference directions
    delta = nadir_points - ideal_points
    delta[delta == 0] = np.finfo(float).eps  # Avoid division by zero

    # Calculate the distances for each solution and select appropriately
    for solution in P:
        objectives = obj_fun(solution)
        direction = (objectives - ideal_points) / delta
        norm_direction = np.linalg.norm(direction, axis=0)

        if not np.any(np.isnan(norm_direction)) and not np.any(
            np.isinf(norm_direction)
        ):
            mating_pool.append(solution)

    mating_pool_nan = util.NaN_handling(mating_pool)
    return mating_pool_nan


def SBXCrossover(parent1, parent2, eta_c, crossover_prob):

    offspring1 = np.copy(parent1)
    offspring2 = np.copy(parent2)

    if np.random.rand() <= crossover_prob:
        for i in range(len(parent1)):
            if np.random.rand() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    y1 = min(parent1[i], parent2[i])
                    y2 = max(parent1[i], parent2[i])
                    lb, ub = const.bounds[i]

                    rand = np.random.rand()
                    beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
                    alpha = 2.0 - np.power(beta, -(eta_c + 1.0))

                    if rand <= 1.0 / alpha:
                        beta_q = np.power((rand * alpha), 1.0 / (eta_c + 1.0))
                    else:
                        beta_q = np.power(
                            (1.0 / (2.0 - rand * alpha)), 1.0 / (eta_c + 1.0)
                        )

                    c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                    # Ensure children are within the bounds
                    offspring1[i] = np.clip(c1, lb, ub)
                    offspring2[i] = np.clip(c2, lb, ub)

    return offspring1, offspring2


def NonUniformMutation(solution, iteration, max_iterations, b=5.0):

    mutated_solution = np.copy(solution)
    for i in range(len(solution)):
        if np.random.rand() < 1.0 / len(solution):
            x_i = solution[i]
            lb, ub = const.bounds[i]
            delta = np.random.rand()
            if np.random.rand() < 0.5:
                delta_val = delta * (ub - x_i) * (1 - iteration / max_iterations) ** b
                mutated_solution[i] = np.clip(x_i + delta_val, lb, ub)
            else:
                delta_val = delta * (x_i - lb) * (1 - iteration / max_iterations) ** b
                mutated_solution[i] = np.clip(x_i - delta_val, lb, ub)
    return mutated_solution


def SBXCrossoverAndNonUniformMutation(
    population, mating_pool, eta_c, crossover_prob, iteration, max_iterations
):

    Q = []
    while len(Q) < len(population):
        if len(mating_pool) < 2:
            # Skip the step if mating pool has less than 2 members
            break

        # Select two parents randomly from the mating pool
        i, j = np.random.choice(list(range(len(mating_pool))), 2, replace=False)
        parent1, parent2 = mating_pool[i], mating_pool[j]

        # Apply SBX Crossover
        offspring1, offspring2 = SBXCrossover(parent1, parent2, eta_c, crossover_prob)

        # Apply Non-Uniform Mutation
        offspring1 = NonUniformMutation(offspring1, iteration, max_iterations)
        offspring2 = NonUniformMutation(offspring2, iteration, max_iterations)

        # Add the offspring to the offspring population
        Q.append(offspring1)
        if len(Q) < len(population):
            Q.append(offspring2)

    Q_nan = util.NaN_handling(Q)

    return np.round(Q_nan, 4)


# Function to check if solution q dominates solution p
def dominated(p, q):
    return np.all(np.array(p) <= np.array(q)) and np.any(np.array(p) < np.array(q))


# Non-dominated sorting function
def nondominated_sort(population, objectives):
    """Perform non-dominated sorting on the population."""
    fronts = [[]]
    dom_count = np.zeros(len(population), dtype=int)
    dom_set = [set() for _ in range(len(population))]

    for i, p_obj1 in enumerate(objectives):
        for j, p_obj2 in enumerate(objectives):
            if i != j:  # Avoid self-comparison
                if dominated(p_obj1, p_obj2):
                    dom_set[i].add(j)
                elif dominated(p_obj2, p_obj1):
                    dom_count[i] += 1
        if dom_count[i] == 0:
            fronts[0].append(i)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dom_set[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # Removing the last empty front
    if not fronts[-1]:
        fronts.pop(-1)

    return fronts


def calculate_crowding_distance(front, objectives):
    num_solutions = len(front)
    num_objectives = len(objectives[0])
    distances = np.zeros(num_solutions)

    front_solutions = [objectives[idx] for idx in front]

    for i in range(num_objectives):
        sorted_indices = sorted(
            range(num_solutions), key=lambda idx: front_solutions[idx][i]
        )
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float("inf")
        obj_min = front_solutions[sorted_indices[0]][i]
        obj_max = front_solutions[sorted_indices[-1]][i]

        if obj_max != obj_min:
            for j in range(1, num_solutions - 1):
                distances[sorted_indices[j]] += (
                    front_solutions[sorted_indices[j + 1]][i]
                    - front_solutions[sorted_indices[j - 1]][i]
                ) / (obj_max - obj_min)

    return distances


def select_next_population(combined_population, objectives, N, fronts):
    selected_indices = []
    for front in fronts:
        if len(selected_indices) + len(front) <= N:
            selected_indices.extend(front)
        else:
            crowding_distances = calculate_crowding_distance(front, objectives)
            sorted_front = [
                idx for _, idx in sorted(zip(crowding_distances, front), reverse=True)
            ]
            selected_indices.extend(sorted_front[: N - len(selected_indices)])
            break

    selected_population = [combined_population[i] for i in selected_indices]

    # Debug: Print selected population indices
    # print("Selected indices for the next population:", selected_indices)
    return selected_population


def check_pareto_non_uniformity(F1, threshold=0.1):
    if len(F1) < 2:
        return False

    # Sort solutions based on one of the objectives (e.g., the first objective)
    F1_sorted = sorted(F1, key=lambda sol: sol[0])

    # Calculate the Euclidean distances between consecutive solutions
    distances = []
    for i in range(len(F1_sorted) - 1):
        distance = np.linalg.norm(np.array(F1_sorted[i + 1]) - np.array(F1_sorted[i]))
        distances.append(distance)

    # Calculate the variance of the distances
    variance = np.var(distances)

    # Check if the variance exceeds the threshold
    return variance > threshold


# Replaced with new suggestion ... for older version refer to the beginning of this notebook
def theta_dominated_selection(St, z_star, z_nad, reference_points, N0, t, pe=0.5):
    def calculate_theta(solution, z_star, z_nad):
        delta = z_nad - z_star
        delta[delta == 0] = np.finfo(float).eps  # Avoid division by zero

        direction = (solution - z_star) / delta
        norm_direction = np.linalg.norm(direction)
        theta = np.arctan(norm_direction)  # Calculate the angle in radians
        return theta

    def valid_selection(individuals, ref_point, N):
        individuals.sort(key=lambda x: np.linalg.norm(x[0] - ref_point))
        top_individuals = [ind[0] for ind in individuals[:N]]
        return top_individuals

    def calculate_effective_frequencies(St, reference_points, z_star, z_nad):
        effective_counts = np.zeros(len(reference_points))

        for ind in St:
            best_ref_idx = None
            best_angle = float("inf")

            for i, ref in enumerate(reference_points):
                if np.array_equal(ref, z_star):
                    continue  # Skip the point if it equals the ideal point

                norm_ind = np.linalg.norm(ind - z_star)
                norm_ref = np.linalg.norm(ref - z_star)
                if norm_ind == 0 or norm_ref == 0:
                    angle = np.pi / 2  # Max angle if norms are zero
                else:
                    direction = np.squeeze((ind - z_star) / norm_ind)
                    ref_direction = np.squeeze((ref - z_star) / norm_ref)

                    cos_theta = np.clip(np.dot(direction, ref_direction), -1.0, 1.0)
                    angle = np.arccos(cos_theta)

                if angle < best_angle:
                    best_angle = angle
                    best_ref_idx = i

            if best_ref_idx is not None:
                effective_counts[best_ref_idx] += 1

        return effective_counts / len(St)  # Normalize to get frequencies

    # Calculate frequencies of reference points being valid based on proximity
    freq_valid = calculate_effective_frequencies(St, reference_points, z_star, z_nad)
    valid_reference_points = [
        ref for i, ref in enumerate(reference_points) if freq_valid[i] >= pe
    ]

    combined_population = [(ind, calculate_theta(ind, z_star, z_nad)) for ind in St]
    combined_population = [
        item for item in combined_population if not np.isinf(item[1])
    ]  # Filter invalid θ values
    combined_population.sort(key=lambda x: x[1])

    selected_individuals = []

    if valid_reference_points:
        per_reference_point = N0 // len(valid_reference_points)
        for ref_point in valid_reference_points:
            selected_from_ref = valid_selection(
                individuals=combined_population,
                ref_point=ref_point,
                N=per_reference_point,
            )
            selected_individuals.extend(selected_from_ref)

    # Ensure population size does not exceed N0
    selected_individuals = selected_individuals[:N0]

    while len(selected_individuals) < N0:
        remaining_slots = N0 - len(selected_individuals)
        filtered_combined_pop = [
            ind
            for ind in combined_population
            if not any(np.array_equal(ind[0], x) for x in selected_individuals)
        ]
        selected_individuals.extend(
            [ind[0] for ind in filtered_combined_pop[:remaining_slots]]
        )

    return selected_individuals[:N0]


def calculate_angles(individuals, reference_points):
    num_individuals = len(individuals)
    num_ref_points = len(reference_points)
    angles = np.zeros((num_individuals, num_ref_points), dtype=float)

    for i in range(num_individuals):
        for j in range(num_ref_points):
            norm_indiv = np.linalg.norm(individuals[i])
            norm_ref = np.linalg.norm(reference_points[j])
            if norm_indiv == 0 or norm_ref == 0:
                angles[i, j] = np.pi / 2  # max angle if one of the norms is zero
            else:
                cos_theta = np.dot(individuals[i], reference_points[j]) / (
                    norm_indiv * norm_ref
                )
                cos_theta = np.clip(cos_theta, -1, 1)
                angles[i, j] = np.arccos(cos_theta)
    return angles


def angle_based_selection(St, Pt_next_N0, N1, reference_points):
    if not Pt_next_N0:
        raise ValueError("Pt_next_N0 should have been filled before calling angle_based_selection.")

    # Determine individuals not already selected
    remaining_individuals = np.array([
        ind for ind in St if not any(np.array_equal(ind, n0) for n0 in Pt_next_N0)
    ])

    if len(remaining_individuals) == 0:
        return np.empty((0, 3))  # Shape-safe empty

    # Compute angles and select those with minimal angle to reference points
    angles = calculate_angles(remaining_individuals, reference_points)
    if angles.size == 0:
        return np.empty((0, 3))  # Shape-safe empty

    min_angles = np.min(angles, axis=1)
    selected_indices = np.argsort(min_angles)[:N1]
    Pt_next_N1 = remaining_individuals[selected_indices]

    # Fill remaining slots if fewer than N1 selected
    if len(Pt_next_N1) < N1:
        remaining_slots = N1 - len(Pt_next_N1)

        extra_individuals = np.array([
            ind for ind in remaining_individuals
            if not any(np.array_equal(ind, selected) for selected in Pt_next_N1)
        ])

        to_add = extra_individuals[:remaining_slots]

        # Ensure dimensions match
        if to_add.size == 0:
            to_add = np.empty((0, Pt_next_N1.shape[1]))
        elif to_add.ndim == 1:
            to_add = np.expand_dims(to_add, axis=0)

        # If still not enough, fill with random individuals
        total_needed = N1 - (len(Pt_next_N1) + len(to_add))
        if total_needed > 0:
            random_fill = np.array([
                [np.random.uniform(5, 25.2),           # Power
                 np.random.choice([2, 4, 16, 64]),     # Modulation (restricted to real-world values)
                 np.random.uniform(1, 10)]             # Symbol rate
                for _ in range(total_needed)
            ])
            to_add = np.concatenate((to_add, random_fill), axis=0)

        Pt_next_N1 = np.concatenate((Pt_next_N1, to_add), axis=0)

    return Pt_next_N1

def usable_reference_count(Pt_next, z_star, z_nad, Lambda, t, pe=0.5):
    effective_references = []

    # Compute objectives for each individual in Pt_next
    Pt_objectives = combined_objective(Pt_next)

    for ref in Lambda:
        used_count = 0
        for obj_values in Pt_objectives:
            if np.all(obj_values >= z_star) and np.all(obj_values <= z_nad):
                denominator = z_nad - z_star
                denominator = np.where(denominator == 0, np.inf, denominator)
                direction = (obj_values - z_star) / denominator
                ref_direction = (ref - z_star) / denominator
                if np.linalg.norm(direction) == 0 or np.linalg.norm(ref_direction) == 0:
                    continue  # Skip this point due to zero magnitude direction vector

                cos_similarity = np.dot(direction, ref_direction) / (
                    np.linalg.norm(direction) * np.linalg.norm(ref_direction)
                )
                if cos_similarity > 0.9:
                    used_count += 1

        # Check if reference point meets the confidence threshold
        if used_count >= pe * t:
            effective_references.append(ref)

    return effective_references


def adjust_reference_points(Lambda, Re):
    Rt_e = len(Re)
    new_Re = list(Re)  # Ensure it's mutable

    # Determine number of new points needed
    num_new_points = len(Lambda) - Rt_e

    # Select randomly from the set of original reference points or generate new ones within bounds
    original_points_to_reintroduce = np.random.choice(
        len(Lambda), num_new_points, replace=False
    )

    for idx in original_points_to_reintroduce:
        new_point = Lambda[idx]
        if all(not np.array_equal(new_point, ref) for ref in Re):
            new_Re.append(new_point)

    return new_Re


def filter_sublists(input_list):
    return [
        sublist
        for sublist in input_list
        if isinstance(sublist, list) and len(sublist) == 3
    ]
