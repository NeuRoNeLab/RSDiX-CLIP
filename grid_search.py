from argparse import ArgumentParser

import yaml


def generate_combinations(params, keys_array, keys_count, current_key_index=0, run_params=None):
    """
    Generate parameter combinations and print commands.

    Args:
        params (dict): Dictionary of parameter names and values.
        keys_array (list): List of parameter keys.
        keys_count (int): Number of parameter combinations to generate.
        current_key_index (int): Index of the current parameter key.
        run_params (dict): Dictionary to store the generated parameter combination.
    """
    if run_params is None:
        run_params = {}

    if keys_count == 0:
        params_string = " ".join([f"--{key} {run_params[key]}" for key in run_params])
        print(f"Running {script} with config file: {config_file} and parameters: {params_string}")
        # Uncomment the following line to execute the Python script with the generated parameters
        # You'll need to replace `python3` with the correct Python executable for your system.
        # subprocess.run(["python3", script, "fit", "--config", config_file] + params_string.split())
    else:
        current_key = keys_array[current_key_index]
        current_key_index += 1
        keys_count -= 1

        if params[current_key] == "":
            run_params[current_key] = ""
            generate_combinations(params, keys_array, keys_count, current_key_index, run_params)
        else:
            if params[current_key].find(","):
                values = params[current_key].split(",")
                for value in values:
                    run_params[current_key] = value
                    generate_combinations(params, keys_array, keys_count, current_key_index, run_params)
            else:
                run_params[current_key] = params[current_key]
                generate_combinations(params, keys_array, keys_count, current_key_index, run_params)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--grid_search_file", type=str, default="clip_grid.yaml")
    args = parser.parse_args()

    with open(args.grid_search_file, "r") as f:
        data = yaml.safe_load(f)

    script = data["script"]
    config_file = data["config_file"]
    attr_keys = data["attr_keys"]

    model_params = {}
    params_keys_array = []
    keys_len = 0

    for attr_key, parameters in attr_keys.items():
        for param_key, param_value in parameters.items():
            if isinstance(param_value, str):
                model_params[f"{attr_key}.{param_key}"] = param_value
            elif isinstance(param_value, list):
                model_params[f"{attr_key}.{param_key}"] = ",".join(str(val) for val in param_value)

            params_keys_array.append(f"{attr_key}.{param_key}")
            keys_len += 1

    generate_combinations(model_params, params_keys_array, keys_len)
