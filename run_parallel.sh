#!/bin/bash

# Title: Parallel Command Executor
# Description: This script reads commands from a text file and runs them in parallel.
# For each command, it activates a specific Conda environment and exports an API key.
# Usage: ./run_parallel.sh /path/to/your/commands.txt

# --- Configuration ---
# You can modify these paths directly in the script if you prefer
CONDA_ACTIVATION_COMMAND="source /home/youssef/anaconda3/bin/activate /home/youssef/Projects/ArabicMMLU/.conda"
API_KEY_EXPORT_COMMAND="export GROQ_API_KEY='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'"


# --- Script Logic ---

# 1. Check if a file path is provided as an argument.
if [ -z "$1" ]; then
    echo "Error: No command file specified."
    echo "Usage: $0 <path_to_commands_file.txt>"
    exit 1
fi

COMMANDS_FILE="$1"

# 2. Check if the specified file exists.
if [ ! -f "$COMMANDS_FILE" ]; then
    echo "Error: File '$COMMANDS_FILE' not found."
    exit 1
fi

# 3. Read the file line by line and execute each command in parallel.
echo "Starting parallel execution of commands from '$COMMANDS_FILE'..."
echo "---"

while IFS= read -r command_to_run || [[ -n "$command_to_run" ]]; do
    # Skip any empty lines in the file
    if [ -z "$command_to_run" ]; then
        continue
    fi

    # Use a subshell (...) to run the commands. This isolates the environment
    # for each parallel process. The '&' at the end sends it to the background.
    (
        echo "[PID: $$] Spawning process for command: '$command_to_run'"
        
        # Set up the environment
        eval "$CONDA_ACTIVATION_COMMAND"
        eval "$API_KEY_EXPORT_COMMAND"
        
        # Execute the actual command from the file
        # 'eval' is used to properly execute the command string with its arguments.
        eval "$command_to_run"

        echo "[PID: $$] Finished command."

    ) &

done < "$COMMANDS_FILE"

# 4. Wait for all background jobs to complete.
echo "---"
echo "All commands have been spawned. Waiting for all processes to finish..."
wait
echo "All processes have completed successfully."
