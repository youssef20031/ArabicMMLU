import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import time
import re
import torch

# Attempt to import Groq and related errors
try:
    from groq import Groq, RateLimitError, APIError
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    # Define dummy classes if Groq is not installed
    class Groq: pass
    class RateLimitError(Exception): pass
    class APIError(Exception): pass

# Attempt to import Hugging Face classes
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    # Define dummy classes if Transformers is not installed
    class AutoModelForCausalLM: pass
    class AutoTokenizer: pass
    class BitsAndBytesConfig: pass
    class PeftModel: pass

# Determine device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")


# Alphabet maps for hypotheses
hyp_alpa_en = {0: 'A', 1: 'B'}
hyp_alpa_ar = {0: 'أ', 1: 'ب'}

# --- Prompts for Deduction ---
DEDUCTION_PROMPT_ABDUCTIVE_EN = """
You are an expert in abductive reasoning.
Below is a library of reasoning rules that have been found useful for determining the plausibility of hypotheses based on observations.

Rule Library:
---
{rules_text}
---

Now, consider the following new case:

Observation 1: {observation_1}
Observation 2: {observation_2}

Hypothesis A: {hypothesis_A_text}
Hypothesis B: {hypothesis_B_text}

Based *strictly* on the provided Rule Library, which hypothesis (A or B) is more plausible?
Your answer should be only the single letter of the most plausible hypothesis (A or B).
Do not provide explanations or any other text.
"""

DEDUCTION_PROMPT_ABDUCTIVE_AR = """
أنت خبير في الاستدلال الافتراضي.
فيما يلي مكتبة من قواعد الاستدلال التي وُجدت مفيدة في تحديد مدى قبول الفرضيات بناءً على الملاحظات.

مكتبة القواعد:
---
{rules_text}
---

الآن، ضع في اعتبارك الحالة الجديدة التالية:

الملاحظة الأولى: {observation_1}
الملاحظة الثانية: {observation_2}

الفرضية أ: {hypothesis_A_text}
الفرضية ب: {hypothesis_B_text}

بناءً *فقط* على مكتبة القواعد المقدمة، أي فرضية (أ أو ب) هي الأكثر قبولاً؟
يجب أن تكون إجابتك هي الحرف الوحيد للفرضية الأكثر قبولاً (أ أو ب).
لا تقدم أي تفسيرات أو نصوص أخرى.
"""

# --- CoT Prompts for Deduction ---
COT_DEDUCTION_PROMPT_ABDUCTIVE_EN = """
You are an expert in abductive reasoning using a rule library.
Below is a library of reasoning rules:
Rule Library:
---
{rules_text}
---

Consider the new case:
Observation 1: {observation_1}
Observation 2: {observation_2}
Hypothesis A: {hypothesis_A_text}
Hypothesis B: {hypothesis_B_text}

Based *strictly* on the provided Rule Library, reason step-by-step to determine which hypothesis (A or B) is more plausible.
1. Analyze Observation 1 in light of the Rule Library. Which rules are relevant?
2. Analyze Observation 2 in light of the Rule Library. Which rules are relevant?
3. Evaluate Hypothesis A: How well do the identified rules explain or support the observations if Hypothesis A is true?
4. Evaluate Hypothesis B: How well do the identified rules explain or support the observations if Hypothesis B is true?
5. Conclusion: Based on the rule-based evaluation, which hypothesis is more strongly supported by the rules?

Conclude with your final answer in the format: 'Final Answer: X' where X is the single letter of the most plausible hypothesis (A or B). Do not add any other text after this line.
"""

COT_DEDUCTION_PROMPT_ABDUCTIVE_AR = """
أنت خبير في الاستدلال الافتراضي باستخدام مكتبة قواعد.
فيما يلي مكتبة من قواعد الاستدلال:
مكتبة القواعد:
---
{rules_text}
---

ضع في اعتبارك الحالة الجديدة:
الملاحظة الأولى: {observation_1}
الملاحظة الثانية: {observation_2}
الفرضية أ: {hypothesis_A_text}
الفرضية ب: {hypothesis_B_text}

بناءً *فقط* على مكتبة القواعد المقدمة، استدل خطوة بخطوة لتحديد أي فرضية (أ أو ب) هي الأكثر قبولاً.
١. حلل الملاحظة الأولى في ضوء مكتبة القواعد. ما هي القواعد ذات الصلة؟
٢. حلل الملاحظة الثانية في ضوء مكتبة القواعد. ما هي القواعد ذات الصلة؟
٣. قيّم الفرضية أ: إلى أي مدى تشرح القواعد المحددة أو تدعم الملاحظات إذا كانت الفرضية أ صحيحة؟
٤. قيّم الفرضية ب: إلى أي مدى تشرح القواعد المحددة أو تدعم الملاحظات إذا كانت الفرضية ب صحيحة؟
٥. الاستنتاج: بناءً على التقييم المستند إلى القواعد، أي فرضية مدعومة بقوة أكبر بواسطة القواعد؟

اختتم بإجابتك النهائية بالتنسيق: 'الإجابة النهائية: X' حيث X هو الحرف الوحيد للفرضية الأكثر قبولاً (أ أو ب). لا تضف أي نص آخر بعد هذا السطر.
"""

# --- ToT Prompts for Deduction (Simulated) ---
TOT_DEDUCTION_PROMPT_ABDUCTIVE_EN = """
You are an expert in abductive reasoning using a rule library.
Below is a library of reasoning rules:
Rule Library:
---
{rules_text}
---

Consider the new case:
Observation 1: {observation_1}
Observation 2: {observation_2}
Hypothesis A: {hypothesis_A_text}
Hypothesis B: {hypothesis_B_text}

Based *strictly* on the provided Rule Library, use a Tree of Thought approach to determine the most plausible hypothesis (A or B):
1.  Initial Rule Brainstorm & Hypothesis Connection:
    *   For Hypothesis A: Identify rules from the library that could potentially link the observations to this hypothesis. Consider rules that support or might contradict it.
    *   For Hypothesis B: Identify rules from the library that could potentially link the observations to this hypothesis. Consider rules that support or might contradict it.
2.  Evaluate Plausibility Paths - Path A (Assuming Hypothesis A is true):
    *   Assess how strongly the identified rules for A connect Observation 1 to Hypothesis A.
    *   Assess how strongly the identified rules for A connect Observation 2 to Hypothesis A.
    *   Overall coherence of Path A based on the rules.
3.  Evaluate Plausibility Paths - Path B (Assuming Hypothesis B is true):
    *   Assess how strongly the identified rules for B connect Observation 1 to Hypothesis B.
    *   Assess how strongly the identified rules for B connect Observation 2 to Hypothesis B.
    *   Overall coherence of Path B based on the rules.
4.  Compare Paths and Decide: Based *only* on the rule-based evaluation, which hypothesis (A or B) has the more coherent and strongly supported explanatory path from observations to hypothesis using the Rule Library?

Conclude with your final answer in the format: 'Final Answer: X' where X is the single letter of the most plausible hypothesis (A or B). Do not add any other text after this line.
"""

TOT_DEDUCTION_PROMPT_ABDUCTIVE_AR = """
أنت خبير في الاستدلال الافتراضي باستخدام مكتبة قواعد.
فيما يلي مكتبة من قواعد الاستدلال:
مكتبة القواعد:
---
{rules_text}
---

ضع في اعتبارك الحالة الجديدة:
الملاحظة الأولى: {observation_1}
الملاحظة الثانية: {observation_2}
الفرضية أ: {hypothesis_A_text}
الفرضية ب: {hypothesis_B_text}

بناءً *فقط* على مكتبة القواعد المقدمة، استخدم نهج شجرة الأفكار لتحديد الفرضية الأكثر قبولاً (أ أو ب):
١. عصف ذهني أولي للقواعد وربط الفرضيات:
    *   بالنسبة للفرضية أ: حدد القواعد من المكتبة التي يمكن أن تربط الملاحظات بهذه الفرضية. ضع في اعتبارك القواعد التي تدعمها أو قد تتعارض معها.
    *   بالنسبة للفرضية ب: حدد القواعد من المكتبة التي يمكن أن تربط الملاحظات بهذه الفرضية. ضع في اعتبارك القواعد التي تدعمها أو قد تتعارض معها.
٢. تقييم مسارات القبول - المسار أ (بافتراض أن الفرضية أ صحيحة):
    *   قيّم مدى قوة ربط القواعد المحددة للفرضية أ للملاحظة الأولى بالفرضية أ.
    *   قيّم مدى قوة ربط القواعد المحددة للفرضية أ للملاحظة الثانية بالفرضية أ.
    *   التماسك العام للمسار أ بناءً على القواعد.
٣. تقييم مسارات القبول - المسار ب (بافتراض أن الفرضية ب صحيحة):
    *   قيّم مدى قوة ربط القواعد المحددة للفرضية ب للملاحظة الأولى بالفرضية ب.
    *   قيّم مدى قوة ربط القواعد المحددة للفرضية ب للملاحظة الثانية بالفرضية ب.
    *   التماسك العام للمسار ب بناءً على القواعد.
٤. قارن المسارات وقرر: بناءً *فقط* على التقييم المستند إلى القواعد، أي فرضية (أ أو ب) لديها المسار التفسيري الأكثر تماسكًا ودعمًا قويًا من الملاحظات إلى الفرضية باستخدام مكتبة القواعد؟

اختتم بإجابتك النهائية بالتنسيق: 'الإجابة النهائية: X' حيث X هو الحرف الوحيد للفرضية الأكثر قبولاً (أ أو ب). لا تضف أي نص آخر بعد هذا السطر.
"""


# --- Helper Functions ---

def load_rule_library(file_path: str) -> list[str]:
    """Loads rules from a JSON file, expecting a list of dicts with a 'rule' key."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_rules = json.load(f)
        rule_library_texts = [item['rule'] for item in raw_rules if 'rule' in item and isinstance(item['rule'], str) and item['rule'].strip()]
        if not rule_library_texts:
            print(f"Warning: No valid rules found in {file_path} or rules are empty.")
            return []
        return rule_library_texts
    except FileNotFoundError:
        print(f"Error: Rule library file not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from rule library file {file_path}")
        return []
    except Exception as e:
        print(f"Error loading rule library from {file_path}: {e}")
        return []

def get_abductive_data_from_row(row: pd.Series, lang_alpa: str) -> dict | None:
    """
    Extracts observations, hypotheses, and gold answer letter from a DataFrame row.
    """
    obs1 = str(row.get('observation_1', '')).strip()
    obs2 = str(row.get('observation_2', '')).strip()
    hyp1_text = str(row.get('hypothesis_1', '')).strip()
    hyp2_text = str(row.get('hypothesis_2', '')).strip()
    label_str = str(row.get('label', '')).strip()

    if not all([obs1, obs2, hyp1_text, hyp2_text, label_str]):
        return None

    current_prompt_alpa_map = hyp_alpa_ar if lang_alpa == 'ar' else hyp_alpa_en
    gold_answer_letter = None

    # Assuming CSV label '1' means hypothesis_1 (A/أ) is correct, and '2' means hypothesis_2 (B/ب) is correct.
    if label_str == '1':
        gold_answer_letter = current_prompt_alpa_map[0]
    elif label_str == '2':
        gold_answer_letter = current_prompt_alpa_map[1]
    # Fallback for other label formats (e.g., '0', 'A', 'B')
    elif label_str == '0': # If '0' is used for the first hypothesis
        gold_answer_letter = current_prompt_alpa_map[0]
    else: # Check for direct letter labels 'A'/'B' or 'أ'/'ب'
        temp_label_upper = label_str.upper() # Ensure consistent casing for A/B
        # For Arabic, 'أ' and 'ب' don't typically change with .upper(), which is fine.
        if temp_label_upper == current_prompt_alpa_map[0].upper(): # Compare with uppercased A/أ
            gold_answer_letter = current_prompt_alpa_map[0]
        elif temp_label_upper == current_prompt_alpa_map[1].upper(): # Compare with uppercased B/ب
            gold_answer_letter = current_prompt_alpa_map[1]
        else:
            # print(f"Warning: Unrecognized label format '{label_str}' in row. Expected '1', '2', '0', 'A', 'B', 'أ', or 'ب'.")
            return None

    if gold_answer_letter is None: # Should only be None if the above logic fails to assign
        return None

    return {
        "obs1": obs1, "obs2": obs2,
        "hyp1_text": hyp1_text, "hyp2_text": hyp2_text,
        "gold_answer_letter": gold_answer_letter
    }

def format_deduction_abductive_prompt(rules_text_list: list[str], obs1: str, obs2: str,
                                      hyp_A_text: str, hyp_B_text: str, lang_prompt: str,
                                      use_cot: bool = False, use_tot: bool = False) -> str:
    rules_str = "\n".join(f"- {rule.strip()}" for rule in rules_text_list if rule.strip())

    if use_tot: # ToT overrides CoT
        template = TOT_DEDUCTION_PROMPT_ABDUCTIVE_AR if lang_prompt == 'ar' else TOT_DEDUCTION_PROMPT_ABDUCTIVE_EN
    elif use_cot:
        template = COT_DEDUCTION_PROMPT_ABDUCTIVE_AR if lang_prompt == 'ar' else COT_DEDUCTION_PROMPT_ABDUCTIVE_EN
    else: # Zero-shot
        template = DEDUCTION_PROMPT_ABDUCTIVE_AR if lang_prompt == 'ar' else DEDUCTION_PROMPT_ABDUCTIVE_EN
    
    return template.format(
        rules_text=rules_str,
        observation_1=obs1, observation_2=obs2,
        hypothesis_A_text=hyp_A_text, hypothesis_B_text=hyp_B_text
    )


def call_llm_for_deduction(prompt_text: str, llm_config: dict, max_retries: int = 5, initial_backoff: int = 2) -> str | None:
    """Calls the configured LLM (Groq or local HF) for deduction."""
    llm_type = llm_config.get("type")
    # Get max_tokens from llm_config, with different defaults based on strategy (indirectly via main)
    # These are just fallback defaults if not set in llm_config, main should set them.
    default_max_tokens_hf = llm_config.get("max_tokens_to_generate", 250 if llm_config.get("is_cot_or_tot") else 2)
    default_max_tokens_groq = llm_config.get("max_tokens_to_generate", 250 if llm_config.get("is_cot_or_tot") else 10)


    if llm_type == "groq":
        if not GROQ_AVAILABLE:
            print("Error: Groq library not available for Groq LLM call.")
            return None
        client = llm_config.get('client')
        model_name = llm_config.get('model_name')
        if not client or not model_name:
            print("Error: Groq client or model_name not provided in llm_config.")
            return None
        
        max_tokens_for_call = llm_config.get("max_tokens_to_generate", default_max_tokens_groq)
        current_attempt = 0
        backoff_time = initial_backoff
        while current_attempt < max_retries:
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_text}],
                    model=model_name,
                    temperature=0.0,
                    max_tokens=max_tokens_for_call
                )
                return chat_completion.choices[0].message.content.strip()
            except RateLimitError:
                print(f"Rate limit. Retrying in {backoff_time}s... (Attempt {current_attempt + 1}/{max_retries})")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 60)
            except APIError as e:
                print(f"Groq API Error: {e}. Retrying in {backoff_time}s... (Attempt {current_attempt + 1}/{max_retries})")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 60)
            except Exception as e:
                print(f"Unexpected Groq LLM error: {e}. Attempt {current_attempt + 1}/{max_retries}")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 60)
            current_attempt += 1
        print(f"Groq LLM call failed after {max_retries} retries.")
        return None

    elif llm_type == "hf":
        if not HF_AVAILABLE:
            print("Error: Hugging Face Transformers library not available for HF LLM call.")
            return None
        model = llm_config.get("model")
        tokenizer = llm_config.get("tokenizer")
        max_len = llm_config.get("max_length", 2048) # Default max_length for tokenizer
        max_new_tokens_for_call = llm_config.get("max_tokens_to_generate", default_max_tokens_hf)

        if not model or not tokenizer:
            print("Error: HF model or tokenizer not provided in llm_config.")
            return None
        try:
            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_len, padding=False).to(device)
            with torch.no_grad():
                # Ensure pad_token_id is set for generation
                pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
                generation_output = model.generate(
                    **inputs,
                    temperature=0.0,   # Deterministic output
                    do_sample=False,   # No sampling
                    top_p=None,        # Explicitly set for greedy
                    top_k=None,        # Explicitly set for greedy
                    pad_token_id=pad_token_id,
                    max_new_tokens=max_new_tokens_for_call
                )
            # Decode only the newly generated tokens
            response_text = tokenizer.decode(generation_output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            # Ensure that an empty or whitespace-only response becomes None
            return response_text if response_text else None
        except Exception as e:
            print(f"Error during Hugging Face model generation: {e}")
            return None
    else:
        print(f"Error: Unknown LLM type '{llm_type}' in llm_config.")
        return None

def parse_deduction_response(llm_output: str | None, lang_alpa: str, is_cot_or_tot: bool = False) -> str | None:
    """Parses the LLM's response to get 'A'/'B' or 'أ'/'ب'."""
    if not llm_output: # Handles None or empty string
        return None
    
    cleaned_output_for_final_answer_check = llm_output.strip() # Keep original case for regex
    
    current_alpa_map = hyp_alpa_ar if lang_alpa == 'ar' else hyp_alpa_en
    valid_choices_set = set(current_alpa_map.values()) # e.g. {'A', 'B'} or {'أ', 'ب'}
    
    # For CoT/ToT, primarily look for "Final Answer: X"
    if is_cot_or_tot:
        # Regex to find "Final Answer: X" or "الإجابة النهائية: X" (case-insensitive for "Final Answer")
        # It captures the single letter choice.
        # Handles optional brackets [[X]] or [X] as sometimes seen.
        patterns = [
            r"Final Answer:\s*\[?\[?([A-B])\]?\]?$", # English
            r"الإجابة النهائية:\s*\[?\[?([أ-ب])\]?\]?$" # Arabic
        ]
        
        final_answer_letter = None
        for pattern_str in patterns:
            # Search from the end of the string for the last occurrence if multiple.
            # However, prompts ask for no text after, so a standard search should be fine.
            # For more robustness, one might iterate all matches and take the last one.
            match = re.search(pattern_str, cleaned_output_for_final_answer_check, re.IGNORECASE if "Final Answer" in pattern_str else 0)
            if match:
                letter = match.group(1).upper() if lang_alpa == 'en' else match.group(1) # Ensure 'A'/'B' for English
                if letter in valid_choices_set:
                    final_answer_letter = letter
                    break # Found a valid letter
        if final_answer_letter:
            return final_answer_letter
        # If CoT/ToT was used but "Final Answer: X" not found, it's likely a parse error for this mode.
        # We could fall through to simpler parsing, but it might be less reliable for verbose CoT/ToT.
        # For now, if is_cot_or_tot and pattern fails, return None.
        # print(f"Warning: CoT/ToT mode - Could not parse 'Final Answer: X' from: '{llm_output}'")
        # return None # Stricter for CoT/ToT if pattern fails.

    # Standard parsing (Zero-shot or fallback if CoT/ToT pattern fails and we decide to allow fallback)
    # Standardize: strip whitespace. For English, also uppercase.
    cleaned_output_simple_parse = llm_output.strip().upper() if lang_alpa == 'en' else llm_output.strip()
    if not cleaned_output_simple_parse: # If stripping results in an empty string
        return None

    # 1. Exact match (single character that is a valid choice)
    if cleaned_output_simple_parse in valid_choices_set:
        return cleaned_output_simple_parse

    # 2. Single unique valid character in the output (handles "A.", "(A)", "Answer: A")
    present_valid_chars = set()
    # Use the appropriately cased output for character iteration
    # For English, valid_choices_set is {'A', 'B'}, cleaned_output_simple_parse is uppercased.
    # For Arabic, valid_choices_set is {'أ', 'ب'}, cleaned_output_simple_parse is not uppercased.
    # So, iterate over cleaned_output_simple_parse for this logic.
    for char_token in cleaned_output_simple_parse: 
        if char_token in valid_choices_set: 
            present_valid_chars.add(char_token)
    
    if len(present_valid_chars) == 1:
        return list(present_valid_chars)[0]
    
    # Fallback for CoT/ToT if the "Final Answer: X" was not found, try the simple parse on the whole output.
    # This part is reached if is_cot_or_tot is true AND the regex failed, OR if is_cot_or_tot is false.
    # The above simple parsing (exact match, unique char) would have already run.
    # If we are here and is_cot_or_tot was true, it means the regex failed, and the simple parse also failed.
    # If is_cot_or_tot was false, it means simple parse failed.
    # So, if we reach here, it's likely unparseable by current methods.

    return None



# --- Main Function ---
# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Deduce hypothesis plausibility using a rule library and LLMs.")
    parser.add_argument("--eval_data_file", type=str, required=True, help="Path to evaluation CSV (obs1, obs2, hyp1, hyp2, label).")
    parser.add_argument("--rule_library_file", type=str, required=True, help="Path to JSON rule library.")
    parser.add_argument("--output_results_file", type=str, default="deduction_results.csv", help="Filename for results CSV.")
    parser.add_argument("--output_folder", type=str, default="results_deduction", help="Folder for results CSV.")
    parser.add_argument("--lang_prompt", type=str, default="en", choices=["en", "ar"], help="Prompt language.")
    parser.add_argument("--lang_alpa", type=str, default="en", choices=["en", "ar"], help="Hypothesis label language (A/B vs أ/ب).")
    parser.add_argument("--max_examples", type=int, default=None, help="Max evaluation examples (for testing).")

    # Prompting strategy
    parser.add_argument("--use_cot", action='store_true', help="Use Chain of Thought prompting for deduction.")
    parser.add_argument("--use_tot", action='store_true', help="Use Tree of Thought prompting for deduction (overrides CoT).")

    # LLM choice
    parser.add_argument("--use_hf_model", action='store_true', help="Use a local Hugging Face model.")
    
    # Groq arguments (used if --use_hf_model is false)
    parser.add_argument("--groq_model", type=str, default="llama3-70b-8192", help="Groq model ID.")
    
    # Hugging Face arguments (used if --use_hf_model is true)
    parser.add_argument("--hf_model_path", type=str, default=None, help="Path or ID for Hugging Face model.")
    parser.add_argument("--lora_weights", type=str, default=None, help="Path to LoRA weights (optional).")
    parser.add_argument("--load_8bit", action='store_true', help="Load HF model in 8-bit.")
    parser.add_argument("--hf_max_length", type=int, default=2048, help="Max input length for HF tokenizer.")
    
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    
    llm_config = {"type": None}
    model_identifier_suffix = "" # For filename

    is_cot_or_tot_enabled = args.use_tot or args.use_cot # ToT takes precedence if both are somehow true
    llm_config["is_cot_or_tot"] = is_cot_or_tot_enabled

    # Determine max tokens based on prompting strategy
    if args.use_tot:
        hf_max_new_tokens = 350  # Increased for ToT
        groq_max_tokens = 2048    # Increased for ToT
        model_identifier_suffix += "_tot"
    elif args.use_cot:
        hf_max_new_tokens = 300  # Increased for CoT
        groq_max_tokens = 2048    # Increased for CoT
        model_identifier_suffix += "_cot"
    else: # Zero-shot
        hf_max_new_tokens = 3    # Short for zero-shot (A, B, or أ, ب)
        groq_max_tokens = 10     # Short for zero-shot, Groq might need a bit more for safety

    if args.use_hf_model:
        if not HF_AVAILABLE:
            print("Error: Hugging Face Transformers library not found. Please install it (`pip install transformers peft accelerate bitsandbytes`).")
            return
        if not args.hf_model_path:
            print("Error: --hf_model_path must be specified when using --use_hf_model.")
            return
        
        print(f"Using Hugging Face model: {args.hf_model_path}")
        llm_config["type"] = "hf"
        model_name_for_file = args.hf_model_path.split("/")[-1]
        if args.lora_weights:
            model_name_for_file += f"_{args.lora_weights.split('/')[-1]}"
        model_identifier_for_filename = f"hf_{model_name_for_file}{model_identifier_suffix}"
        llm_config["max_tokens_to_generate"] = hf_max_new_tokens


        try:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
            
            quantization_config = None
            if args.load_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=torch.float16 
                )
                print("Loading HF model in 8-bit.")

            model = AutoModelForCausalLM.from_pretrained(
                args.hf_model_path,
                quantization_config=quantization_config,
                device_map="auto", 
                trust_remote_code=True,
                torch_dtype=torch.float16 if args.load_8bit or "cuda" in device else None
            )

            if args.lora_weights:
                print(f"Loading LoRA weights from: {args.lora_weights}")
                model = PeftModel.from_pretrained(model, args.lora_weights)
                print("LoRA weights loaded.")
            
            if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
                print("Setting pad_token to eos_token for HF model.")
                tokenizer.pad_token = tokenizer.eos_token
                if hasattr(model.config, 'pad_token_id'):
                    model.config.pad_token_id = tokenizer.eos_token_id

            tokenizer.padding_side = "left" 
            model.eval()
            
            llm_config["model"] = model
            llm_config["tokenizer"] = tokenizer
            llm_config["max_length"] = args.hf_max_length
            print(f"Hugging Face model '{args.hf_model_path}' loaded. Max new tokens: {hf_max_new_tokens}")

        except Exception as e:
            print(f"Error loading Hugging Face model '{args.hf_model_path}': {e}")
            return
    else: # Default to Groq
        if not GROQ_AVAILABLE:
            print("Error: Groq SDK not found. Please install it (`pip install groq`) or use --use_hf_model.")
            return
        print(f"Using Groq model: {args.groq_model}")
        llm_config["type"] = "groq"
        model_identifier_for_filename = f"groq_{args.groq_model.replace('/', '-')}{model_identifier_suffix}"
        llm_config["max_tokens_to_generate"] = groq_max_tokens
        try:
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                print("Error: GROQ_API_KEY environment variable not set.")
                return
            llm_config["client"] = Groq(api_key=groq_api_key)
            llm_config["model_name"] = args.groq_model
            print(f"Groq client initialized for model: {args.groq_model}. Max tokens: {groq_max_tokens}")
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            return

    # Construct output filename
    base_output_filename = os.path.splitext(args.output_results_file)[0]
    ext_output_filename = os.path.splitext(args.output_results_file)[1] if os.path.splitext(args.output_results_file)[1] else ".csv"
    full_output_path = os.path.join(args.output_folder, f"{base_output_filename}_{model_identifier_for_filename}{ext_output_filename}")

    rule_library_texts = load_rule_library(args.rule_library_file)
    if not rule_library_texts:
        print(f"Exiting due to issues loading rule library from {args.rule_library_file}.")
        return
    print(f"Loaded {len(rule_library_texts)} rules.")

    try:
        eval_df = pd.read_csv(args.eval_data_file)
        if args.max_examples:
            eval_df = eval_df.head(args.max_examples)
        print(f"Loaded {len(eval_df)} evaluation examples from {args.eval_data_file}.")
    except Exception as e:
        print(f"Error loading evaluation data from {args.eval_data_file}: {e}")
        return

    results_data = []
    correct_predictions = 0
    total_processed = 0
    parse_errors = 0
    data_errors = 0
    llm_call_errors = 0

    print("Starting deduction process...")
    for index, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Deducing"):
        data_item = get_abductive_data_from_row(row, args.lang_alpa)
        if not data_item:
            data_errors += 1
            results_data.append({
                "obs1": row.get('observation_1', ''), "obs2": row.get('observation_2', ''),
                "hyp1": row.get('hypothesis_1', ''), "hyp2": row.get('hypothesis_2', ''),
                "gold_label_original": row.get('label', ''), "gold_label_letter": "DATA_ERROR",
                "predicted_label_letter": "DATA_ERROR", "is_correct": False,
                "llm_raw_response": "Skipped due to data error."
            })
            continue

        prompt = format_deduction_abductive_prompt(
            rule_library_texts, data_item["obs1"], data_item["obs2"],
            data_item["hyp1_text"], data_item["hyp2_text"], args.lang_prompt,
            use_cot=args.use_cot, use_tot=args.use_tot # Pass CoT/ToT flags
        )

        llm_response = call_llm_for_deduction(prompt, llm_config)
        
        predicted_letter_for_csv = "ERROR" # Default
        is_correct = False

        if llm_response is None:
            llm_call_errors +=1
            predicted_letter_for_csv = "LLM_CALL_ERROR"
        else:
            predicted_letter = parse_deduction_response(llm_response, args.lang_alpa, is_cot_or_tot=is_cot_or_tot_enabled)
            if predicted_letter is None:
                parse_errors += 1
                predicted_letter_for_csv = "PARSE_ERROR"
            elif data_item["gold_answer_letter"]: # Parsed successfully
                is_correct = (predicted_letter == data_item["gold_answer_letter"])
                if is_correct:
                    correct_predictions += 1
                predicted_letter_for_csv = predicted_letter
            else: # Parsed successfully but no gold answer (should not happen if data_item is valid)
                predicted_letter_for_csv = predicted_letter 
        
        total_processed +=1

        results_data.append({
            "obs1": data_item["obs1"], "obs2": data_item["obs2"],
            "hyp1": data_item["hyp1_text"], "hyp2": data_item["hyp2_text"],
            "gold_label_original": row.get('label', ''),
            "gold_label_letter": data_item["gold_answer_letter"],
            "predicted_label_letter": predicted_letter_for_csv,
            "is_correct": is_correct,
            "llm_raw_response": llm_response if llm_response else "LLM_ERROR_NO_RESPONSE"
        })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(full_output_path, index=False, encoding='utf-8')
    print(f"\nDeduction results saved to: {full_output_path}")

    if data_errors > 0:
        print(f"Note: {data_errors} rows skipped due to input data errors.")
    if llm_call_errors > 0:
        print(f"Note: {llm_call_errors} LLM calls failed or returned no response.")
    if parse_errors > 0: 
        print(f"Note: {parse_errors} LLM responses could not be parsed into a valid choice (excluding LLM call failures).")
        
    if total_processed > 0:
        accuracy = (correct_predictions / total_processed) * 100
        print(f"Deduction Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_processed} correctly predicted examples)")
    else:
        print("No examples were successfully processed to calculate accuracy.")


if __name__ == "__main__":
    main()