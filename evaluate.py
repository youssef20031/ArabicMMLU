import argparse
import sys
import pandas as pd
import os
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM 
from tqdm import tqdm
from numpy import argmax
import torch
from sklearn.metrics import accuracy_score
from util_prompt import prepare_data
from util_compute import predict_classification_causal_by_letter, predict_classification_mt0_by_letter, predict_classification_gemini, configure_gemini


TOKEN = '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'

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_8bit", action='store_true')
    parser.add_argument("--share_gradio", action='store_true')
    parser.add_argument("--base_model", type=str, help="Path to pretrained model", required=True)
    parser.add_argument("--lora_weights", type=str, default="x")
    parser.add_argument("--lang_alpa", type=str, default="ar")
    parser.add_argument("--lang_prompt", type=str, default="ar")
    parser.add_argument("--output_folder", type=str, default="english_output")
    parser.add_argument("--chain_of_thought", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--offload_folder", type=str, default=None, help="Folder for offloaded weights when loading in 8-bit")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # --- Check if using Gemini ---
    is_gemini_model = args.base_model.startswith("gemini-")

    cot_suffix = "cot_" if args.chain_of_thought else ""
    SAVE_FILE = f'{args.output_folder}/result_prompt_{args.lang_prompt}_alpa_{args.lang_alpa}_{cot_suffix}{args.base_model.split("/")[-1]}.csv'

    model = None
    tokenizer = None
    predict_classification = None

    if is_gemini_model:
        print(f"Using Gemini model: {args.base_model}")
        # Configure Gemini API once
        try:
            configure_gemini()
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        # Set the prediction function for Gemini
        predict_classification = predict_classification_gemini
        # No need to load HF model/tokenizer for Gemini
    else:
        # --- Existing Hugging Face Model Loading Logic ---
        print(f"Loading Hugging Face model: {args.base_model}")
        tokenizer_class = LlamaTokenizer if 'llama' in args.base_model else AutoTokenizer
        model_class = LlamaForCausalLM if 'llama' in args.base_model else AutoModelForCausalLM
        base_model_str = str(args.base_model)
        tokenizer = tokenizer_class.from_pretrained(base_model_str, trust_remote_code=True, use_auth_token=TOKEN)

        if 'mt0' in args.base_model or 'arat5' in args.base_model.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, device_map="auto", load_in_8bit="xxl" in args.base_model)
            predict_classification = predict_classification_mt0_by_letter
        else:
            model = model_class.from_pretrained(args.base_model, load_in_8bit=args.load_8bit, trust_remote_code=True, device_map="auto", use_auth_token=TOKEN, offload_folder=args.offload_folder)
            predict_classification = predict_classification_causal_by_letter

        # Load adapter if we use adapter
        if args.lora_weights != "x":
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                torch_dtype=torch.float16,
            )
            # Update save file name if LoRA is used
            SAVE_FILE = f'{args.output_folder}/result_prompt_{args.lang_prompt}_alpa_{args.lang_alpa}_{args.lora_weights.split("/")[-1]}.csv'


        # unwind broken decapoda-research config
        if 'llama' in args.base_model:
            model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        # --- End of Hugging Face Model Loading Logic ---

    # Assume prepare_data now returns subjects as well
    inputs, golds, outputs_options, subjects = prepare_data(args)
    preds = []
    probs = []

    print("Starting predictions...")
    for idx in tqdm(range(len(inputs))):
        if is_gemini_model:
            # Call Gemini prediction function
            conf, pred = predict_classification(args.base_model, inputs[idx], outputs_options[idx], args.lang_alpa)
        else:
            # Call Hugging Face prediction function
            conf, pred = predict_classification(model, tokenizer, inputs[idx], outputs_options[idx], device, args.lang_alpa)

        probs.append(conf) # Will be None for Gemini
        preds.append(pred)

    output_df = pd.DataFrame()
    output_df['input'] = inputs
    output_df['golds'] = golds
    output_df['options'] = outputs_options
    output_df['preds'] = preds
    output_df['probs'] = probs # Note: Will contain None for Gemini models
    output_df['subject'] = subjects  # added subject column
    output_df.to_csv(SAVE_FILE, index=False)
    print(f"Results saved to {SAVE_FILE}")

if __name__ == "__main__":
    main()

