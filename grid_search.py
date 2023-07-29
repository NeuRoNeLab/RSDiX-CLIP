
import yaml


def generate_combinations(params, keys_array, keys_count, current_key_index=0, run_params=None):
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
    with open("grid.yaml", "r") as f:
        data = yaml.safe_load(f)

    script = data["script"]
    config_file = data["config_file"]
    attr_keys = data["attr_keys"]

    params = {}
    keys_array = []
    keys_len = 0

    for attr_key, parameters in attr_keys.items():
        for param_key, param_value in parameters.items():
            if isinstance(param_value, str):
                params[f"{attr_key}.{param_key}"] = param_value
            elif isinstance(param_value, list):
                params[f"{attr_key}.{param_key}"] = ",".join(str(val) for val in param_value)

            keys_array.append(f"{attr_key}.{param_key}")
            keys_len += 1

    generate_combinations(params, keys_array, keys_len)