import csv
import pandas as pd
from datetime import datetime
import os
import numpy as np
from pathlib import Path
from typing import List, Optional
import sys
import re
import random

from codebase import constants as const
from codebase import ASMoEA
from sklearn.preprocessing import MinMaxScaler

iso_8601_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

model_folder = f"run_{iso_8601_str}"

# === Optional Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "csv"
log_dir = BASE_DIR / "logs" / model_folder
log_path = log_dir / "log.info"
err_path = log_dir / "log.err"

# === Regex to strip ANSI ===
ansi_escape = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    return ansi_escape.sub("", text)


# === Dual Output Logger ===
class TeeLogger:
    def __init__(self, path, mode="w", encoding="utf-8"):
        self.terminal = sys.__stdout__
        self.log_file = open(path, mode, encoding=encoding)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(strip_ansi(message))

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


# === Set up tee logging (before initializing rich) ===

os.makedirs(log_dir, exist_ok=True)

sys.stdout = TeeLogger(log_path)
sys.stderr = TeeLogger(err_path)

# === Rich Console Setup ===
from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.panel import Panel
from rich.prompt import Prompt
from rich.align import Align

console = Console(
    file=sys.stdout,
    force_terminal=True,
    color_system="truecolor",
    highlighter=NullHighlighter(),
)


# === Logging Wrapper ===
def log(msg):
    console.print(msg)


# === CSV Functions ===
def init_csv(data_path, headers):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    log(f"\n📂  Dataset path set at : [yellow]{data_path}[/yellow]\n")


def store_gen(t, data_path, gen_DV, gen_OV):
    df = pd.DataFrame(
        {
            "gen": [t] * len(gen_DV),
            "TP": [ind[0] for ind in gen_DV],
            "MO": [ind[1] for ind in gen_DV],
            "SR": [ind[2] for ind in gen_DV],
            "TP_o": [ind[0] for ind in gen_OV],
            "MO_o": [ind[1] for ind in gen_OV],
            "SR_o": [ind[2] for ind in gen_OV],
        }
    )
    df.to_csv(data_path, mode="a", header=False, index=False)


def get_data_path():
    headers = ["gen", "TP", "MO", "SR", "TP_o", "MO_o", "SR_o"]
    safe_timestamp = iso_8601_str.replace(":", "-")
    data_path = os.path.join(data_dir, f"{safe_timestamp}.csv")
    return data_path, headers


def fetch_interval_data(data_path, interval):
    df = pd.read_csv(data_path)
    filtered_df = df[(df["gen"] >= interval[0]) & (df["gen"] <= interval[1])]
    filtered_df.columns = range(filtered_df.shape[1])
    return filtered_df


# === Utility Functions ===
def check_duplicates(P):
    population_set = set(tuple(x) for x in P)
    x = len(population_set)
    y = len(P)
    z = round((x / y) * 100, 4)
    rich_divider("-")
    log("\n🔍  Checking duplicates ...")
    if x == y:
        log("\n🧹  No duplicates found")
    else:
        log(f"\n🧩  Unique: {x}/{y}\n📊  Diversity: {z}%\n")
        rich_divider("-")


def clip_modulation_order(value):
    Mod_val = [2, 4, 16, 64]
    return min(Mod_val, key=lambda x: x if x >= value else float("inf"))


def print_line(length: int, char: str, head_tail: Optional[List[str]] = None) -> None:
    if head_tail is not None and len(head_tail) == 2:
        print(f"\n{head_tail[0]}", char * length, f"{head_tail[1]}\n")
    else:
        print("\n" + char * length + "\n")


def normalize_dv(gen_DV):
    for ind in gen_DV:
        ind[1] = clip_modulation_order(ind[1])
    normalized_data = []
    for arr in gen_DV:
        clipped = np.array(
            [
                np.clip(val, min_val, max_val)
                for val, (min_val, max_val) in zip(arr, const.bounds)
            ]
        )
        normalized_data.append(clipped)
    return normalized_data

def NaN_handling(population):
    for member in population:
        for i in range(len(member)):
            if np.isnan(member[i]) or np.isinf(member[i]):  # Check for inf as well
                member[i] = random.uniform(const.bounds[i][0], const.bounds[i][1])
                # print(f"NaN found at {i}: replaced with {member[i]}")
    return population


def obj_NaN_handling(obj_list):
    # print("During NaN handling of OBJ VAL :\n", obj_list)
    for i, obj in enumerate(obj_list):
        if np.any(np.isnan(obj)) or np.any(np.isinf(obj)):  # Check for inf as well
            obj_list[i] = random.random()
    return obj_list


def normalize_general(input_data):
    if isinstance(input_data, pd.DataFrame):
        combined_pop = input_data.iloc[:, [1, 2, 3]]
    else:
        combined_pop = input_data[0]

    # Apply NaN handling to decision variables
    cleaned_pop = NaN_handling(np.array(combined_pop).tolist())
    np_combined_pop = np.array(cleaned_pop, dtype=np.float64)
    np_combined_pop = normalize_dv(np_combined_pop)

    # Compute objective values from cleaned decision variables
    np_combined_obj = ASMoEA.combined_objective(np_combined_pop)
    np_combined_obj = np.array(np_combined_obj, dtype=np.float64)

    # Fit scalers
    scaler_decision_vars = MinMaxScaler()
    scaler_decision_vars.fit(np_combined_pop)

    scaler_objective_vals = MinMaxScaler()
    scaler_objective_vals.fit(np_combined_obj)

    # Apply transformation
    normalized_decision_vars = scaler_decision_vars.transform(np_combined_pop)
    normalized_objective_vals = scaler_objective_vals.transform(np_combined_obj)

    return (
        normalized_decision_vars,
        normalized_objective_vals,
        scaler_decision_vars,
        scaler_objective_vals,
    )

def clean_up():
    data_dir_list = os.listdir(data_dir)
    if data_dir_list is not None:
        for filename in data_dir_list:
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


def rich_divider(char="-", label=None, head_tail=["", ""]):
    label_text = f" {label} " if label else ""
    total_fill = (
        const.line_width - len(label_text) - len(head_tail[0]) - len(head_tail[1])
    )
    half = total_fill // 2
    extra = total_fill % 2
    line = (
        f"{head_tail[0]}{char * half}{label_text}{char * (half + extra)}{head_tail[1]}"
    )
    console.print(line)

def format_time(elapsed_time):
    seconds = int(elapsed_time)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or (hours == 0 and minutes == 0):
        parts.append(f"{seconds}s")

    return " ".join(parts)

def format_model_name(name: str) -> str:
    parts = name.split("_")
    return " ".join(part.upper() if part.isupper() or len(part) <= 3 else part.capitalize() for part in parts)

def prompt_model_choice():

    # Model prompt header
    console.print("[magenta]🧠  Choose the model:[/magenta]\n")

    # Menu options
    labels = const.model_names
    labels.append("Exit")
    bullets = list(range(1,len(labels)+1))

    # Get longest label width for alignment
    max_len = max(len(label) for label in labels)
    padded = [label.ljust(max_len) for label in labels]

    # Display aligned menu with colored numbers
    for i, (emoji, label) in enumerate(zip(const.emojis, padded), start=1):
        console.print(f"    [cyan][{i}][/cyan]   {emoji}  {label}")

    # Input loop
    while True:
        try:
            choice = int(Prompt.ask("\n[bold white]>[/bold white]"))
            if choice in list(range(1,len(labels))):
                return choice
            elif choice == bullets[-1]:
                console.print("\n[red]❌  Execution Terminated[/red]\n")
                sys.exit(0)
        except ValueError:
            pass
        console.print("[red]❌ Invalid choice. Try again.[/red]\n")

def set_params():

    rich_divider()
    log("\n⚙️   [magenta]Set Parameters for Execution[/magenta]\n\n")

    # Inputs with icons
    N = int(console.input("[cyan]👥  Enter Population        : [/cyan]"))
    mod = int(console.input("[cyan]🔁  Enter DL Interval       : [/cyan]"))
    T = int(console.input("[cyan]🧬  Enter Total Generations : [/cyan]"))

    return N, mod, T

