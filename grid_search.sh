#!/bin/bash

# get script to run
script=$(yq e ".script" grid.yaml)
# get config.yaml to use as base configuration
config_file=$(yq e ".config_file" grid.yaml)
# corresponds to [model, data] etc.
readarray -t attr_keys < <(yq ".attr_keys | keys | .[]" grid.yaml)

# create an associative array to store the parameter key-value pairs
declare -A params
declare keys_array

keys_len=0
for attr_key in "${attr_keys[@]}"
  do
    readarray -t parameters_keys < <(yq ".attr_keys.$attr_key | keys | .[]" grid.yaml)
    for parameter_key in "${parameters_keys[@]}"
      do
          params["$attr_key.$parameter_key"]=$(yq ".attr_keys.$attr_key.$parameter_key" grid.yaml)
          keys_array+=("$attr_key.$parameter_key")
          ((keys_len++))
      done
  done

declare -A run_params

generate_combinations() {
  # $1 current key index $2 total keys count
  declare -i keys_count=$2

  if [[ $keys_count -eq 0 ]];
    then
      # run current combination
      local params_string=""
      for key in "${!run_params[@]}";
        do
          params_string="${params_string} --${key} ${run_params[$key]}"
        done
      echo "Running ${script} with config file: ${config_file} and parameters: ${params_string}"
      echo python3 "${script}" fit --config "${config_file}" ${params_string}
      python3 "${script}" fit --config "${config_file}" ${params_string}
    else
        declare -i current_key_index=$1
        local current_key="${keys_array[$current_key_index]}"
        ((current_key_index++))
        ((keys_count--))

        if [[ ${params[$current_key]} == *","* ]];
          then
            # value is list
            IFS=',' read -ra values <<< "${params[$current_key]}"
            for value in "${values[@]}"
              do
                run_params[$current_key]=$value
                generate_combinations "$current_key_index" "$keys_count"
              done
          else
            run_params[$current_key]=${params[$current_key]}
            generate_combinations "$current_key_index" "$keys_count"
        fi
  fi
}

generate_combinations 0 keys_len

