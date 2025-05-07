import argparse
import sys
import pandas as pd
import os
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
from tqdm import tqdm
import torch
import time # Import time for potential delays/timing

# Import utility functions
from util_prompt import load_and_format_data
from util_compute import (predict_classification_causal_by_letter,
                          predict_classification_mt0_by_letter,
                          predict_classification_gemini,
                          configure_gemini,
                          predict_classification_groq) # <-- Import the Groq function

# --- Add Groq import ---
try:
    from groq import Groq, RateLimitError, APIError
    import httpx # Import httpx for timeout configuration
except ImportError:
    # Allow script to run without groq if not used
    Groq = None
    RateLimitError = Exception # Define dummy exceptions if Groq not installed
    APIError = Exception
    httpx = None # Define httpx as None if not installed
# --- End Groq import ---
# --- End Groq import ---

# Optional: Define Hugging Face token if needed for private models
# TOKEN = 'YOUR_HF_TOKEN' # Replace with your token if necessary
TOKEN = None # Or set to None if not needed

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="mmlu", choices=["mmlu", "abductive"], help="Type of task to evaluate ('mmlu' or 'abductive')")
    parser.add_argument("--data_file", type=str, default=None, help="Path to the data file (used if task_type is 'abductive' or to override default MMLU path)")
    parser.add_argument("--load_8bit", action='store_true', help="Load Hugging Face models in 8-bit")
    parser.add_argument("--share_gradio", action='store_true', help="Enable Gradio sharing (if applicable)") # Keep if Gradio is used elsewhere
    parser.add_argument("--base_model", type=str, help="Path/ID for Hugging Face model or Gemini ID (e.g., 'google/gemini-pro')", default=None) # Make optional
    parser.add_argument("--lora_weights", type=str, default="x", help="Path to LoRA weights (if using HF model)")
    parser.add_argument("--lang_alpa", type=str, default="ar", help="Language of answer choices ('ar' or 'en')")
    parser.add_argument("--lang_prompt", type=str, default="ar", help="Language of the prompt ('ar' or 'en')")
    parser.add_argument("--output_folder", type=str, default="super_new_output", help="Folder to save the results CSV")
    parser.add_argument("--chain_of_thought", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--tree_of_thought", action="store_true", help="Use tree-of-thought prompting (experimental, overrides CoT)") # <-- Add ToT argument
    parser.add_argument("--offload_folder", type=str, default=None, help="Folder for offloaded weights when loading HF in 8-bit")
    # --- Add Groq arguments ---
    parser.add_argument("--use_groq", action='store_true', help="Use Groq API instead of local/Gemini models")
    parser.add_argument("--groq_model", type=str, default="llama3-70b-8192", help="Groq model ID to use (e.g., 'llama3-70b-8192', 'mixtral-8x7b-32768')") # Updated default
    # --- End of Groq arguments ---
    args = parser.parse_args()

    # --- Add validation ---
    if args.tree_of_thought and args.chain_of_thought:
        print("Warning: --tree_of_thought and --chain_of_thought are mutually exclusive. Using Tree of Thought.")
        args.chain_of_thought = False # Prioritize ToT if both are specified
    if args.use_groq and args.base_model:
        print("Warning: --base_model is ignored when --use_groq is specified.")
        args.base_model = None # Clear base_model if using Groq
    elif not args.use_groq and not args.base_model:
        parser.error("--base_model (for HF/Gemini) is required unless --use_groq is specified.")
    if args.use_groq and args.lora_weights != "x":
        print("Warning: --lora_weights are ignored when --use_groq is specified.")
        args.lora_weights = "x"
    if args.use_groq and args.load_8bit:
        print("Warning: --load_8bit is ignored when --use_groq is specified.")
    # --- End of validation ---

    return args

def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # --- Determine model type and setup ---
    is_groq_model = args.use_groq
    is_gemini_model = not is_groq_model and args.base_model and args.base_model.startswith("gemini-")
    is_hf_model = not is_groq_model and not is_gemini_model and args.base_model is not None

    cot_suffix = "cot_" if args.chain_of_thought else ""
    tot_suffix = "_tot" if args.tree_of_thought else "" # <-- Add ToT suffix
    prompt_method_suffix = tot_suffix if args.tree_of_thought else cot_suffix

     # --- Replace the existing filename generation block with this ---
    if is_groq_model:
        model_identifier = args.groq_model.replace("/", "-") # Sanitize model name for filename
        SAVE_FILE = f'result_prompt_{args.lang_prompt}_alpa_{args.lang_alpa}{prompt_method_suffix}_groq_{model_identifier}.csv' # Use prompt_method_suffix
    elif is_gemini_model:
        model_identifier = args.base_model # Gemini IDs are usually filename-safe
        SAVE_FILE = f'result_prompt_{args.lang_prompt}_alpa_{args.lang_alpa}{prompt_method_suffix}_{model_identifier}.csv' # Use prompt_method_suffix
    elif is_hf_model:
        model_identifier = args.base_model.split("/")[-1] # Get last part of HF path
        # Update save file name if LoRA is used with HF model
        if args.lora_weights != "x":
             lora_identifier = args.lora_weights.split("/")[-1]
             SAVE_FILE = f'result_prompt_{args.lang_prompt}_alpa_{args.lang_alpa}{prompt_method_suffix}_{model_identifier}_{lora_identifier}.csv' # Use prompt_method_suffix
        else:
             SAVE_FILE = f'result_prompt_{args.lang_prompt}_alpa_{args.lang_alpa}{prompt_method_suffix}_{model_identifier}.csv' # Use prompt_method_suffix
    else:
        # Fallback filename if neither Groq nor base_model is specified (shouldn't happen with validation)
        print("Error: Could not determine model type or identifier.")
        # Use a default/fallback name or exit
        SAVE_FILE = f'result_prompt_{args.lang_prompt}_alpa_{args.lang_alpa}{prompt_method_suffix}_unknown_model.csv'
        # sys.exit(1) # Optionally exit if model type is mandatory

    # Join with the output folder
    SAVE_FILE = os.path.join(args.output_folder, SAVE_FILE)
    # --- End of replacement ---

    model = None
    tokenizer = None
    predict_classification = None
    groq_client = None # Initialize Groq client variable

    if is_groq_model:
        print(f"Using Groq model: {args.groq_model}")
        if Groq is None or httpx is None: # Check both Groq and httpx imports
             print("Error: The 'groq' and 'httpx' libraries are required to use --use_groq. Please install them (`pip install groq httpx`).")
             sys.exit(1)
        try:
            # Initialize Groq client with custom timeouts
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.") # Use ValueError for clarity

            # Configure longer timeouts (e.g., 30 seconds connect, 180 seconds read)
            timeout_config = httpx.Timeout(1000.0, read=1800.0) # Adjust values as needed

            groq_client = Groq(
                api_key=groq_api_key,
                timeout=timeout_config # Pass the timeout config
            )
            print("Groq client initialized with custom timeouts.")

            # Set the prediction function for Groq
            predict_classification = lambda input_text, labels, lang_alpa: predict_classification_groq(
                groq_client, args.groq_model, input_text, labels, lang_alpa
            )
        except ValueError as ve: # Catch specific ValueError for API key
             print(f"Configuration Error: {ve}")
             sys.exit(1)
        except Exception as e: # Catch other potential initialization errors
            print(f"Error initializing Groq client: {e}")
            sys.exit(1)

    elif is_gemini_model:
        print(f"Using Gemini model: {args.base_model}")
        # Configure Gemini API once
        try:
            configure_gemini()
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        # Set the prediction function for Gemini
        # Pass model name via lambda
        predict_classification = lambda input_text, labels, lang_alpa: predict_classification_gemini(
            args.base_model, input_text, labels, lang_alpa
        )
        # No need to load HF model/tokenizer for Gemini

    elif is_hf_model:
        # --- Existing Hugging Face Model Loading Logic ---
        print(f"Loading Hugging Face model: {args.base_model}")
        # Determine tokenizer and model classes based on model name
        if 'llama' in args.base_model.lower():
            tokenizer_class = LlamaTokenizer
            model_class = LlamaForCausalLM
        elif 'mt0' in args.base_model.lower() or 'mt5' in args.base_model.lower() or 'arat5' in args.base_model.lower():
             tokenizer_class = AutoTokenizer
             model_class = AutoModelForSeq2SeqLM # T5-based models are Seq2Seq
        else: # Default to Auto classes for other causal LMs
            tokenizer_class = AutoTokenizer
            model_class = AutoModelForCausalLM

        print(f"Using Tokenizer: {tokenizer_class.__name__}, Model: {model_class.__name__}")

        tokenizer = tokenizer_class.from_pretrained(args.base_model, trust_remote_code=True, use_auth_token=TOKEN)

        load_in_8bit = args.load_8bit
        # Seq2Seq models might not support 8-bit loading in the same way, adjust if needed
        if model_class == AutoModelForSeq2SeqLM and load_in_8bit:
             print("Warning: 8-bit loading might behave differently for Seq2Seq models. Attempting anyway.")
             # For T5/MT0, device_map='auto' often handles distribution better than explicit 8-bit
             # model = model_class.from_pretrained(args.base_model, device_map="auto", load_in_8bit=load_in_8bit, trust_remote_code=True, use_auth_token=TOKEN)
             model = model_class.from_pretrained(args.base_model, device_map="auto", trust_remote_code=True, use_auth_token=TOKEN) # Try without 8bit first for T5
             load_in_8bit = False # Disable explicit 8bit flag if using device_map for T5
        else:
             model = model_class.from_pretrained(
                 args.base_model,
                 load_in_8bit=load_in_8bit,
                 trust_remote_code=True,
                 device_map="auto", # Let HF handle device placement
                 use_auth_token=TOKEN,
                 offload_folder=args.offload_folder if load_in_8bit else None # Only use offload with 8bit
             )

        # Load adapter if specified
        if args.lora_weights != "x":
            print(f"Loading LoRA weights from: {args.lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                torch_dtype=torch.float16, # Adapters often expect float16
                # device_map="auto" # PeftModel usually inherits device map
            )
            print("LoRA weights loaded.")

        # Configure model/tokenizer specific settings
        if 'llama' in args.base_model.lower():
            # unwind broken decapoda-research config
            model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2
            print("Applied LLaMA specific token configurations.")

        if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
             print("Warning: Setting pad_token to eos_token.")
             tokenizer.pad_token = tokenizer.eos_token
             model.config.pad_token_id = tokenizer.eos_token_id


        model.eval() # Set model to evaluation mode

        # Optional: Compile model for potential speedup (requires PyTorch 2.0+)
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     print("Compiling model...")
        #     try:
        #         model = torch.compile(model)
        #         print("Model compiled successfully.")
        #     except Exception as e:
        #         print(f"Model compilation failed: {e}")


        # Set the prediction function based on model type
        if isinstance(model, AutoModelForSeq2SeqLM):
            print("Using Seq2Seq prediction function.")
            predict_classification = lambda input_text, labels, lang_alpa: predict_classification_mt0_by_letter(model, tokenizer, input_text, labels, device, lang_alpa)
        else:
            print("Using Causal LM prediction function.")
            predict_classification = lambda input_text, labels, lang_alpa: predict_classification_causal_by_letter(model, tokenizer, input_text, labels, device, lang_alpa)
        # --- End of Hugging Face Model Loading Logic ---

    else:
        print("Error: No valid model type specified (HF, Gemini, or Groq).")
        sys.exit(1)


    # --- Prepare Data ---
    print(f"Preparing data for task: {args.task_type}...")
    # Assume prepare_data returns: prompts, gold_indices, options_lists, subjects
    inputs, golds, outputs_options, subjects, indices, abilities = load_and_format_data(args)
    print(f"Data prepared. Number of examples: {len(inputs)}")

    preds = []
    raw_preds = [] # Store raw model outputs for debugging

    print("Starting predictions...")
    start_time = time.time()
    for idx in tqdm(range(len(inputs)), desc="Evaluating"):
        input_text = inputs[idx]
        labels = outputs_options[idx] # The actual option strings like ['Option A', 'Option B', ...]
        lang_alpa = args.lang_alpa

        # --- Call the selected prediction function ---
        # The lambda functions defined earlier handle passing the correct arguments
        pred, raw_pred = predict_classification(input_text, labels, lang_alpa)

        preds.append(pred) # Store the parsed prediction (e.g., 'A', 'B', 'أ', 'ب')
        raw_preds.append(raw_pred) # Store the raw output

    end_time = time.time()
    print(f"Predictions finished in {end_time - start_time:.2f} seconds.")

    # --- Save Results ---
    if len(inputs) == len(golds) == len(preds) == len(raw_preds) == len(subjects) == len(outputs_options) == len(abilities):
        out = pd.DataFrame({
            'index': indices, # Use the indices list from load_and_format_data
            'prompt': inputs,
            'golds': golds,
            'preds': preds,
            'raw_preds': raw_preds,
            'subject': subjects,
            'options': outputs_options,
            'ABILITY': abilities # Added ABILITY column
        })
        out.to_csv(SAVE_FILE, index=False, encoding='utf-8')
        print(f"Results saved to {SAVE_FILE}")
    else:
        print("Error: Mismatch in lengths of data lists. Cannot save results.")
        print(f"Lengths: Inputs={len(inputs)}, Golds={len(golds)}, Preds={len(preds)}, RawPreds={len(raw_preds)}, Subjects={len(subjects)}, Options={len(outputs_options)}, Abilities={len(abilities)}")


if __name__ == "__main__":
    main()