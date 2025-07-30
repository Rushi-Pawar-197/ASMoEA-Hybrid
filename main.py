# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import time
import importlib

sys.stdout.reconfigure(encoding="utf-8")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from codebase import utility as util
from codebase import constants as const

setup_start = time.time()
util.log("\n[bold cyan]🔧  Setup Initiated[bold cyan]\n")

from codebase import result_parameters as params
from codebase import ASMoEA

choice = util.prompt_model_choice()
model_type = const.model_list[choice]
model = importlib.import_module(f"models.{model_type}")

N, mod, T = util.set_params()

N0 = N // 2
N1 = N - N0

# Flags
dnn_use = False

util.clean_up()
data_path, headers = util.get_data_path()
setup_end = time.time()
setup_time = setup_end - setup_start
run_start_time = time.time()
util.rich_divider("-")
util.log("\n[bright_green]🛠️  Parameters for current Run [/bright_green]\n")
util.log(f"\n[sky_blue1]🧠  Model\t:  [/sky_blue1] [bold white]{const.model_names[choice-1]}[/bold white]")
util.log(f"[sky_blue1]📈  Interval\t:  [/sky_blue1] Every [bold white]{mod} [/bold white]Generations")
util.log(f"[sky_blue1]🧬  Generations\t:  [/sky_blue1] [bold white]{T}[/bold white] Total")
util.log(f"[sky_blue1]👥  Population \t:  [/sky_blue1] [bold white]{N}[/bold white] Individuals")

util.init_csv(data_path, headers)
util.rich_divider("-")

util.log(f"✅  Setup Completed in [bold cyan]{util.format_time(setup_time)} ⏱️[/bold cyan]\n")
util.log("[bold bright_white]🚀  Starting Execution [/bold bright_white]\n")

util.rich_divider("-")

# Initialize population and reference points
P0 = ASMoEA.initialize_population(N, const.num_variables)
init_reference_points = ASMoEA.initialize_reference_points(
    const.num_objectives, const.num_divisions
)

# Determine z_star and z_nad
objectives_initial = ASMoEA.combined_objective(P0)
z_star = np.min(objectives_initial, axis=0)
z_nad = np.max(objectives_initial, axis=0)

Lambda = init_reference_points
Pt = P0  # Current population

PF_last_gen_inx = []
PF_last_gen = []
True_PF = []

mae_list = []
mse_list = []
r2_list = []

hv_per_gen = []
igd_per_gen = []

for t in range(1, T + 1):

    Qt = []

    # Generate the mating pool H
    H = ASMoEA.RGDMatingSelection(Pt, z_star, z_nad)

    # Generate offspring via crossover and mutation
    Qt = ASMoEA.SBXCrossoverAndNonUniformMutation(Pt, H, const.eta_c, const.Pc, t, T)

    combined_population = list(Qt) + list(Pt)

    # Evaluate the combined objective function for each individual in the combined population
    objectives = ASMoEA.combined_objective(combined_population)

    # Adding DNN related code here

    if t % mod == 0:

        if t == mod:
            interval = [t // 2, t]
        else:
            interval = [t - (mod - 1), t]

        # 1 Training will go here

        # util.print_line(105,"-",head_tail=["+","+"])
        util.rich_divider("-", "🔁 DNN Training Block", head_tail=["+", "+"])
        util.log(
            f"\n[yellow]🧬  Gen {str(t).zfill(2)}[/yellow]  →  [green]Training interval[/green] : [yellow]{interval}[/yellow]\n"
        )
        util.log("[cyan]⚙️  Preparing to train DNN ...[/cyan]\n")

        # Read the CSV file with-in interval
        df = util.fetch_interval_data(data_path, interval)

        (
            normalized_decision_vars,
            normalized_objective_vals,
            scaler_decision_vars,
            scaler_objective_vals,
        ) = util.normalize_general(df)

        # print(df)

        X_test, y_test = model.train(
            normalized_objective_vals,
            normalized_decision_vars,
            test_size=0.2,
            random_state=42,
        )

        # 2 Predicting will happen here
        predicted_DV, obj_predicted_DV, obj_test_DV = model.predict(
            scaler_decision_vars, X_test, y_test
        )

        mse, r2, mae = params.eval_dnn_metrics(obj_test_DV, obj_predicted_DV)

        mse_list.append(mse)
        r2_list.append(r2)
        mae_list.append(mae)

        dnn_use = True

        util.log("\n[magenta]🧠  DNN training complete[/magenta] → [bright_green]evolution resumes  🧬[/bright_green]\n")

    elif t > mod:

        util.rich_divider("-", head_tail=["+", "+"])
        util.log(f"\n🧬  Gen {str(t).zfill(2)}  ←  [blue]Predicting[/blue]\n")

        norm_DV, norm_OV, scaler_decision_vars, scaler_objective_vals = (
            util.normalize_general([combined_population, objectives])
        )

        # 3 Predicting will happen here as well
        predicted_DV, obj_predicted_DV, obj_test_DV = model.predict(
            scaler_decision_vars, norm_OV, norm_DV
        )
        dnn_use = True

    # Convert lists to NumPy arrays
    combined_population = np.array(combined_population)
    objectives = np.array(objectives)

    # Ensure all elements are of type np.float64
    combined_population = combined_population.astype(np.float64)
    objectives = objectives.astype(np.float64)

    if dnn_use == True:

        obj_predicted_DV = ASMoEA.filter_sublists(obj_predicted_DV)

        # Convert lists to NumPy arrays
        predicted_DV = np.array(predicted_DV)
        obj_predicted_DV = np.array(obj_predicted_DV)
        predicted_DV = predicted_DV.astype(np.float64)
        obj_predicted_DV = obj_predicted_DV.astype(np.float64)

        GA_DNN_DV = np.concatenate((combined_population, predicted_DV), axis=0)
        obj_GA_DNN_DV = np.concatenate((objectives, obj_predicted_DV), axis=0)
    else:
        GA_DNN_DV = combined_population
        obj_GA_DNN_DV = objectives

    fronts = ASMoEA.nondominated_sort(GA_DNN_DV, obj_GA_DNN_DV)

    if t == T:
        PF_last_gen_inx = fronts[0]
        for idx in PF_last_gen_inx:
            PF_last_gen.append(GA_DNN_DV[idx])

    St = ASMoEA.select_next_population(GA_DNN_DV, obj_GA_DNN_DV, N, fronts)
    F1 = [obj_GA_DNN_DV[i] for i in fronts[0]]

    z_star = np.min(F1, axis=0)
    z_nad = np.max(F1, axis=0)

    is_pareto_non_uniform = ASMoEA.check_pareto_non_uniformity(F1)

    if (t > T * const.alpha) and is_pareto_non_uniform:
        Pt_next_N0 = ASMoEA.theta_dominated_selection(St, z_star, z_nad, Lambda, N0, t)
        Pt_next_N1 = ASMoEA.angle_based_selection(St, Pt_next_N0, N1, Lambda)
        Pt_next = np.concatenate((Pt_next_N0, Pt_next_N1), axis=0)
    else:
        Pt_next = ASMoEA.theta_dominated_selection(St, z_star, z_nad, Lambda, N, t)

    # Reference points assignment
    Re = ASMoEA.usable_reference_count(Pt_next, z_star, z_nad, Lambda, t)
    Rt_e = len(Re)
    # Adjust if fewer effective points:
    if Rt_e < len(Lambda):
        Re = ASMoEA.adjust_reference_points(Lambda, Re)
    Lambda = Re

    Pt = Pt_next

    progress = round((t / T) * 100, 1)
    util.log(
        f"\n[green]🧬  Gen {str(t).zfill(2)}[/green]  -  done ✅   (Progress : [green]{progress} %[/green])\n"
    )

    gen_DV = [list(ind) for ind in Pt]
    gen_DV = util.NaN_handling(gen_DV)
    util.normalize_dv(gen_DV)
    gen_OV = ASMoEA.combined_objective(gen_DV)

    avg_HV, norm_HV = params.hypervolume(Pt, Lambda)
    igd_val, norm_IGD = params.compute_igd(F1)

    hv_per_gen.append(avg_HV)
    igd_per_gen.append(igd_val)

    util.store_gen(t, data_path, gen_DV, gen_OV)

run_end_time = time.time()
run_time = run_end_time - run_start_time
# min, sec = divmod(run_end_time - run_start_time, 60)
util.rich_divider("-")
util.log(f"\n[bright_green]✨  Execution Successful ✨[/bright_green]")
util.log(f"\n[bright_white]🧠  Model\t[/bright_white]:  [bold white]{const.emojis[choice-1]} {const.model_names[choice-1]}[/bold white]")
util.log(f"[bright_white]⏱️   Duration\t[bright_white]:  [bold cyan]{util.format_time(run_time)}[bold cyan]\n")

util.check_duplicates(Pt)

avg_HV, norm_HV = params.hypervolume(Pt, Lambda)
igd_val, norm_IGD = params.compute_igd(PF_last_gen)

util.rich_divider("-")

util.log(f"\n📈  Hypervolume\t: {round(avg_HV,4)}")
util.log(f"⚖️   Normalized HV\t: {round(norm_HV,4)}\n")
util.log(f"📉  IGD\t: {round(igd_val,4)}")
util.log(f"⚖️   Normalized IGD\t: {round(norm_IGD,4)}\n")
util.rich_divider("-")


# print("mse list = ",mse_list)
# print("r-squared list", r2_list)
# print("MAE list", mae_list)

# params.ga_res_visualize(hv_per_gen, igd_per_gen)
# params.DNN_res_visualize(mse_list, r2_list, mae_list)
