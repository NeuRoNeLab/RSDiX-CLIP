import os
from argparse import ArgumentParser
import yaml
from bayes_opt import BayesianOptimization


args = None
script_to_run = None
config_file = None
parameters = {}


def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def is_int(num):
    try:
        int(num)
        return True
    except ValueError:
        return False


def get_last_version():
    items = os.scandir(os.path.join(args.default_root_dir, args.logs_dir))

    # Filter out only the directories from the list
    directories = [item for item in items if os.path.isdir(os.path.join(args.default_root_dir, item))]

    if directories:
        # If there are directories in the list, return the last one
        return os.path.join(args.default_root_dir, directories[-1])
    else:
        # If no directories found, return None or handle the case as needed
        return None


def get_best_val_loss_from_ckpt(path):
    items = os.listdir(path)

    # Get only the checkpoints
    checkpoints = [ckpt for ckpt in items if ckpt.endswith(".ckpt")]

    best_val_loss = float('inf')

    for ckpt in checkpoints:
        loss = float(ckpt.split("=")[-1].split(".ckpt")[0])
        if loss < best_val_loss:
            best_val_loss = loss

    return best_val_loss


# Evaluation function for Bayesian Optimization
def evaluate_clip_model(lr, weight_decay, use_warmup):
    if use_warmup == 0:
        use_warmup = "cosine"
    else:
        use_warmup = "linear"

    print(f"running with config {config_file} and parameters: --model.lr {parameters['lr'][round(lr)]} "
          f"--model.weight_decay {parameters['weight_decay'][round(weight_decay)]}"
          f" --model.use_warmup {use_warmup}")
    os.system(f"python {script_to_run} fit --config {config_file} --model.lr {parameters['lr'][round(lr)]} "
              f"--model.weight_decay {parameters['weight_decay'][round(weight_decay)]}"
              f" --model.use_warmup {use_warmup} --trainer.default_root_dir {args.default_root_dir}")
    # navigate to the trainer's default root dir, get the latest version, find the checkpoint and pick the best val_loss
    last_version = get_last_version()

    if last_version is None:
        raise Exception("Last directory version not found!")

    last_version = os.path.join(last_version, "checkpoints")

    # Return the negative accuracy to maximize (Bayesian Optimization expects a maximization problem)
    return -get_best_val_loss_from_ckpt(os.path.join(args.default_root_dir, last_version))


def evaluate_clipcap_model(clipcap_lr, dropout_transformer, dropout_gpt2, clipcap_weight_decay):
    print(f"Running with config {config_file} and parameters: "
          f"--model.clipcap_lr {clipcap_lr} "
          f"--model.dropout_transformer {dropout_transformer} "
          f"--model.dropout_gpt2 {dropout_gpt2} "
          f"--model.clipcap_weight_decay {clipcap_weight_decay}")

    os.system(f"python {script_to_run} fit "
              f"--config {config_file} "
              f"--model.clipcap_lr {clipcap_lr} "
              f"--model.dropout_transformer {dropout_transformer} "
              f"--model.dropout_gpt2 {dropout_gpt2} "
              f"--model.clipcap_weight_decay {clipcap_weight_decay} "
              f"--trainer.default_root_dir {args.default_root_dir}")

    # Navigate to the trainer's default root dir, get the latest version, find the checkpoint and pick the best val_loss
    last_version = get_last_version()

    if last_version is None:
        raise Exception("Last directory version not found!")
    last_version = os.path.join(last_version, "checkpoints")

    # Return the negative accuracy to maximize (Bayesian Optimization expects a maximization problem)
    return -get_best_val_loss_from_ckpt(os.path.join(args.default_root_dir, last_version))


# Define the hyperparameter search space for Bayesian Optimization
def hyper_search_space(grid_file="grid.yaml"):
    pbounds = {}

    with open(grid_file, "r") as f:
        data = yaml.safe_load(f)

        global script_to_run
        global parameters
        global config_file

        script_to_run = data["script"]
        config_file = data["config_file"]

        attr_keys = data["attr_keys"]

        for attr_key, params in attr_keys.items():
            for param_key, param_value in params.items():
                if param_value.find(","):
                    parameters[param_key] = [int(value) if is_int(value) else float(value) if is_float(value) else value
                                             for value in param_value.split(",")]
                    if len(parameters[param_key]) > 2:
                        # get min and max value
                        pbounds[param_key] = [0, len(parameters[param_key]) - 1]
                    else:
                        pbounds[param_key] = parameters[param_key]
                else:
                    pbounds[param_key] = param_value

        return pbounds


hyper_search_space()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--default_root_dir", type=str, default=os.getcwd())
    parser.add_argument("--logs_dir", type=str, default="lightning_logs")
    parser.add_argument("--grid_file", type=str, default="grid.yaml")
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--n_init_points", type=int, default=5)
    parser.add_argument("--train_clipcap", type=bool, default=False)

    args = parser.parse_args()

    optimizer = BayesianOptimization(f=evaluate_clip_model if args.train_clipcap is False else evaluate_clipcap_model,
                                     pbounds=hyper_search_space(args.grid_file), verbose=2, random_state=42)
    optimizer.maximize(init_points=5, n_iter=args.n_iter)

    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))
