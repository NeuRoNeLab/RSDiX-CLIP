import os
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict


# Base directory
base_dir = "rsdix-clipcap"

# List of versions (replace with actual versions you need to process)
versions = ["68o2nd3f", "s2vesbxu", "o8mkwvk5", "9d0370it", "oy3csoyy", "khcz45rh", "g1j2y0yh", "dopmncdx"]  # Add the versions you need here

for version in versions:
    # Path to the checkpoints directory
    checkpoints_dir = os.path.join(base_dir, version, "checkpoints")

    # Check if checkpoints directory exists
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory not found: {checkpoints_dir}")
        continue

    # Iterate over all subdirectories (model names) in the checkpoints directory
    for model_name in os.listdir(checkpoints_dir):
        model_path = os.path.join(checkpoints_dir, model_name)

        # Check if it's a directory (since we are looking for model directories)
        if os.path.isdir(model_path):
            # Construct input and output paths
            input_checkpoint_dir = os.path.join(checkpoints_dir, model_name) + "/"
            output_checkpoint_path = os.path.join(base_dir, version, model_name)

            # Call the function with dynamic parameters
            convert_zero_checkpoint_to_fp32_state_dict(input_checkpoint_dir, output_checkpoint_path)
