from pathlib import Path

fit_session_name = "sonica-sept-25-2026"
base_dir = Path("/mnt/home/ebalzani/ceph/synaptic_connectivity/")
script_dir = Path("/mnt/home/ebalzani/Code/infer-connectivity/scripts/hpc_fits")
model_dir = base_dir / "models" / fit_session_name
simulations_dir = base_dir / "simulations" / fit_session_name
path_to_config = base_dir / "configs" / fit_session_name
path_to_output = base_dir / "outputs" / fit_session_name
log_dir = base_dir / "logs" / fit_session_name
fit_glm_script = "fit_glm.py"

model_dir.mkdir(exist_ok=True, parents=True)
log_dir.mkdir(exist_ok=True, parents=True)
simulations_dir.mkdir(exist_ok=True, parents=True)
path_to_output.mkdir(exist_ok=True, parents=True)


disbatch_script_path = base_dir / "run_experiment.dsb"


def create_dsbatch_script() -> int:
    tot_configs = 0
    tot_datasets = 0
    with open(disbatch_script_path, "w") as f:
        for config_file in path_to_config.iterdir():
            for dataset in simulations_dir.iterdir():

                # Lines for loading the virtual environment
                lines = [
                    "source ~/cudaenv.sh",
                    "source ~/Code/infer-connectivity/.venv/bin/activate",
                ]
                lines.append(
                    f"python -u {(script_dir / fit_glm_script).as_posix()} {config_file} {dataset} {path_to_output}"
                )

                log_name = f"conf_{config_file.stem}_{dataset.stem}_fit_glm.log"
                command = f'( {" && ".join(lines)} ) &> {log_dir / log_name}'
                f.write(command + "\n")
                tot_datasets += 1
            tot_configs += 1

    print(f"Disbatch script written to {disbatch_script_path}")

    return tot_configs * tot_datasets


def run_experiment():

    num_fits = create_dsbatch_script()
    print("To run:")
    num_jobs = min(20, num_fits)  # Use at most 20 tasks
    print(
        "module load disBatch; "
        "mkdir disbatch_logs; "
        f"sbatch -n {num_jobs} -p gpu --gpus-per-task=1 -t 0-12 --mem-per-cpu=16GB -c 6 disBatch -p disbatch_logs/ {disbatch_script_path}"
    )


if __name__ == "__main__":
    run_experiment()
