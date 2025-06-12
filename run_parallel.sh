#!/bin/bash

# Title: Parallel Command Executor with Filtering
# Description: Reads commands from a file and runs them in parallel.
#              It securely prompts for an API key at runtime.
#              Optional flags can be used to filter which commands from the file to run.
#              The script can be stopped with Ctrl+C, which terminates all child processes.
#
# Usage:
#   Run all commands: ./run_parallel.sh /path/to/commands.txt
#   Run filtered commands: ./run_parallel.sh /path/to/commands.txt --tasks "abductive" --models "gemma2-9b-it"

# --- Configuration ---
CONDA_ACTIVATION_COMMAND="source /home/youssef/anaconda3/bin/activate /home/youssef/Projects/ArabicMMLU/.conda"

# --- Script State ---
declare -A pid_to_command
declare -a pids

# --- Functions ---

# Displays usage information
usage() {
    echo "Usage: $0 <command_file> [filters]"
    echo ""
    echo "  The command_file is required. It should contain one command per line."
    echo ""
    echo "  Optional Filters:"
    echo "    --tasks <types>      Filter by comma-separated task types (e.g., 'abductive,deductive')."
    echo "    --prompts <methods>    Filter by comma-separated prompting methods (e.g., 'direct,chain_of_thought')."
    echo "    --models <names>       Filter by comma-separated model names (e.g., 'gemma2-9b-it')."
    echo "    --help                 Display this help message."
    echo ""
    exit 1
}

# Cleanup function to be called on script exit/interrupt
cleanup() {
    echo -e "\n---"
    echo "Interrupt signal received. Terminating all child processes..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping PID: $pid | Command: '${pid_to_command[$pid]}'"
            kill "$pid"
        fi
    done
    echo "All child processes terminated."
    exit 1
}

# --- Main Logic ---

# Set the trap for Ctrl+C
trap 'cleanup' INT

# Securely prompt for the GROQ API key
read -s -p "Enter your GROQ API Key: " GROQ_API_KEY
echo # Move to a new line after the prompt
if [ -z "$GROQ_API_KEY" ]; then
    echo "Error: GROQ API Key cannot be empty."
    exit 1
fi
# Export the key so it's available to all child processes spawned by this script
export GROQ_API_KEY

# 1. Check for command file and parse arguments
if [ -z "$1" ] || [[ "$1" == --* ]]; then
    echo "Error: A command file must be the first argument."
    usage
fi
COMMANDS_FILE="$1"
shift # Remove the file path from arguments to process filters

if [ ! -f "$COMMANDS_FILE" ]; then
    echo "Error: File '$COMMANDS_FILE' not found."
    exit 1
fi

tasks_str=""
prompts_str=""
models_str=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tasks) tasks_str="$2"; shift ;;
        --prompts) prompts_str="$2"; shift ;;
        --models) models_str="$2"; shift ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# 2. Read and Filter Commands
declare -a all_commands_from_file
declare -a commands_to_run_list

# Read all commands from the file first
while IFS= read -r command_to_run || [[ -n "$command_to_run" ]]; do
    if [ -n "$command_to_run" ]; then
        all_commands_from_file+=( "$command_to_run" )
    fi
done < "$COMMANDS_FILE"

# If no filters are set, run all commands
if [ -z "$tasks_str" ] && [ -z "$prompts_str" ] && [ -z "$models_str" ]; then
    commands_to_run_list=("${all_commands_from_file[@]}")
else
    # Apply filters
    IFS=',' read -ra tasks_filter <<< "$tasks_str"
    IFS=',' read -ra prompts_filter <<< "$prompts_str"
    IFS=',' read -ra models_filter <<< "$models_str"

    for command in "${all_commands_from_file[@]}"; do
        keep_command=true

        # Filter by task
        if [ ${#tasks_filter[@]} -gt 0 ]; then
            task_match=false
            for task in "${tasks_filter[@]}"; do
                if [[ "$command" == *"--task_type='$task'"* ]]; then task_match=true; break; fi
            done
            if ! $task_match; then keep_command=false; fi
        fi

        # Filter by prompt
        if $keep_command && [ ${#prompts_filter[@]} -gt 0 ]; then
            prompt_match=false
            for prompt in "${prompts_filter[@]}"; do
                if [ "$prompt" == "direct" ]; then
                    if [[ "$command" != *"--chain_of_thought"* && "$command" != *"--tree_of_thought"* ]]; then prompt_match=true; break; fi
                else
                    if [[ "$command" == *"--$prompt"* ]]; then prompt_match=true; break; fi
                fi
            done
            if ! $prompt_match; then keep_command=false; fi
        fi
        
        # Filter by model
        if $keep_command && [ ${#models_filter[@]} -gt 0 ]; then
            model_match=false
            for model in "${models_filter[@]}"; do
                if [[ "$command" == *"--openai_model='$model'"* || "$command" == *"--groq_model='$model'"* ]]; then model_match=true; break; fi
            done
            if ! $model_match; then keep_command=false; fi
        fi

        if $keep_command; then
            commands_to_run_list+=( "$command" )
        fi
    done
fi

if [ ${#commands_to_run_list[@]} -eq 0 ]; then
    echo "No commands matched the specified filters. Exiting."
    exit 0
fi

# 3. Execute Commands
echo "Found ${#commands_to_run_list[@]} matching commands to execute."
echo "Press Ctrl+C to terminate all processes at any time."
echo "---"

for command_to_run in "${commands_to_run_list[@]}"; do
    (
        eval "$CONDA_ACTIVATION_COMMAND"
        # The GROQ_API_KEY is inherited from the parent shell via export
        eval "$command_to_run"
    ) &

    pid=$!
    pids+=("$pid")
    pid_to_command[$pid]="$command_to_run"
    
    echo "[RUNNING] PID: $pid | Command: '${command_to_run}'"
done

# 4. Wait for completion
echo "---"
echo "All commands have been spawned. Waiting for completion..."
has_failed=0

for pid in "${pids[@]}"; do
    if wait "$pid"; then
        echo "[COMPLETED] PID: $pid | Command: '${pid_to_command[$pid]}'"
    else
        exit_code=$?
        echo "[FAILED] PID: $pid | Exit Code: $exit_code | Command: '${pid_to_command[$pid]}'"
        has_failed=1
    fi
done

echo "---"
if [ "$has_failed" -eq 1 ]; then
    echo "All processes have finished, but one or more commands failed."
else
    echo "All processes have completed successfully."
fi
