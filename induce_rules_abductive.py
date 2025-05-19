# induce_rules_abductive.py
import argparse
import pandas as pd
import os
import json
from collections import defaultdict
import re
import time
from tqdm import tqdm

# Attempt to import Groq and related errors
try:
    from groq import Groq, RateLimitError, APIError
    import httpx # Required by Groq client for timeouts
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    # Define dummy classes if Groq is not installed, so the script can still be parsed
    # The script will exit later if Groq is actually needed but not available.
    class Groq: pass
    class RateLimitError(Exception): pass
    class APIError(Exception): pass
    # Define dummy Timeout class separately
    class _DummyTimeout: pass 
    # Define dummy httpx class and assign the dummy Timeout
    class httpx: 
        Timeout = _DummyTimeout

# Attempt to import Sentence Transformers and PyTorch
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    # Define dummy classes if not available, script will warn and use exact matching.
    class SentenceTransformer: pass
    class util: pass
    class torch: pass # Basic dummy
    print("Warning: 'sentence-transformers' or 'torch' not found. Rule matching will be exact string matching.")

# --- Prompts and Formatters for Abductive Reasoning ---
# You should move these to your util_prompt.py and import them.

RULE_GENERATION_PROMPT_ABDUCTIVE_EN = (
    "Given the following observations and two hypotheses, where one hypothesis is known to be more plausible:\n\n"
    "Observation 1: {observation_1}\n"
    "Observation 2: {observation_2}\n\n"
    "Hypothesis A: {hypothesis_A_text}\n"
    "Hypothesis B: {hypothesis_B_text}\n\n"
    "The more plausible hypothesis is: Hypothesis {correct_hypothesis_letter} ({correct_hypothesis_text})\n\n"
    "What is a general rule or reasoning principle that explains why Hypothesis {correct_hypothesis_letter} is more plausible than the other, based on the observations? "
    "The rule should be concise and broadly applicable if possible. "
    "Output the rule directly, starting with 'Rule: ' and then the rule text on the same line. "
    "If you can identify multiple distinct rules, output each on a new line, each starting with 'Rule: '.\n"
    "Example:\nRule: If observation 1 suggests a cause and observation 2 is an effect consistent with that cause, a hypothesis linking them is plausible."
)
RULE_GENERATION_PROMPT_ABDUCTIVE_AR = (
    "بالنظر إلى الملاحظات والفرضيتين التاليتين، حيث من المعروف أن إحدى الفرضيات أكثر قبولاً:\n\n"
    "الملاحظة الأولى: {observation_1}\n"
    "الملاحظة الثانية: {observation_2}\n\n"
    "الفرضية أ: {hypothesis_A_text}\n"
    "الفرضية ب: {hypothesis_B_text}\n\n"
    "الفرضية الأكثر قبولاً هي: الفرضية {correct_hypothesis_letter} ({correct_hypothesis_text})\n\n"
    "ما هي القاعدة العامة أو مبدأ الاستدلال الذي يفسر لماذا الفرضية {correct_hypothesis_letter} هي أكثر قبولاً من الأخرى، بناءً على الملاحظات؟ "
    "يجب أن تكون القاعدة موجزة وقابلة للتطبيق على نطاق واسع إن أمكن. "
    "أخرج القاعدة مباشرة، بادئًا بـ 'Rule: ' ثم نص القاعدة على نفس السطر. "
    "إذا كان بإمكانك تحديد قواعد متعددة ومتميزة، فأخرج كل قاعدة على سطر جديد، تبدأ كل منها بـ 'Rule: '.\n"
    "مثال:\nRule: إذا كانت الملاحظة 1 تشير إلى سبب وكانت الملاحظة 2 نتيجة متوافقة مع هذا السبب، فإن الفرضية التي تربطهما تكون مقبولة."
)

RULE_VERIFICATION_PROMPT_ABDUCTIVE_EN = (
    "Consider the following observations and hypotheses:\n\n"
    "Observation 1: {observation_1}\n"
    "Observation 2: {observation_2}\n\n"
    "Hypothesis A: {hypothesis_A_text}\n"
    "Hypothesis B: {hypothesis_B_text}\n\n"
    "The known more plausible hypothesis is: Hypothesis {correct_hypothesis_letter} ({correct_hypothesis_text})\n\n"
    "Now, consider the following rule: \"{rule_to_verify}\"\n\n"
    "If you strictly apply ONLY this rule to the observations and hypotheses, does it help you correctly identify Hypothesis {correct_hypothesis_letter} as the more plausible one? "
    "Answer with only 'Yes' or 'No'. Do not provide explanations or any other text."
)
RULE_VERIFICATION_PROMPT_ABDUCTIVE_AR = (
    "بالنظر إلى الملاحظات والفرضيات التالية:\n\n"
    "الملاحظة الأولى: {observation_1}\n"
    "الملاحظة الثانية: {observation_2}\n\n"
    "الفرضية أ: {hypothesis_A_text}\n"
    "الفرضية ب: {hypothesis_B_text}\n\n"
    "الفرضية المعروفة الأكثر قبولاً هي: الفرضية {correct_hypothesis_letter} ({correct_hypothesis_text})\n\n"
    "الآن، ضع في اعتبارك القاعدة التالية: \"{rule_to_verify}\"\n\n"
    "إذا طبقت هذه القاعدة فقط بصرامة على الملاحظات والفرضيات، فهل تساعدك في تحديد الفرضية {correct_hypothesis_letter} بشكل صحيح على أنها الأكثر قبولاً؟ "
    "أجب فقط بـ 'نعم' أو 'لا'. لا تقدم أي تفسيرات أو نصوص أخرى."
)

# Alphabet maps for hypotheses (A/B or أ/ب)
hyp_alpa_en = {1: 'A', 2: 'B'}
hyp_alpa_ar = {1: 'أ', 2: 'ب'} # Assuming 'أ' for hypothesis_1, 'ب' for hypothesis_2

def get_abductive_data_from_row(row, lang_alpa):
    """
    Extracts observations, hypotheses, and correct hypothesis details from a row
    for abductive reasoning tasks.
    """
    obs1 = str(row.get('observation_1', '')).strip()
    obs2 = str(row.get('observation_2', '')).strip()
    hyp1_text = str(row.get('hypothesis_1', '')).strip()
    hyp2_text = str(row.get('hypothesis_2', '')).strip()
    label = str(row.get('label', '')).strip() # Expected '0' or '1'

    if not all([obs1, obs2, hyp1_text, hyp2_text, label]):
        return None, None, None, None, None, None

    current_hyp_alpa = hyp_alpa_ar if lang_alpa == 'ar' else hyp_alpa_en

    try:
        correct_hyp_idx = int(label) # 0 or 1
        if correct_hyp_idx not in [1, 2]:
             print(f"Warning: Invalid label '{label}' found. Expected '1' or '2'. Skipping row.")
             return None, None, None, None, None, None
    except ValueError:
        print(f"Warning: Non-integer label '{label}' found. Skipping row.")
        return None, None, None, None, None, None

    correct_hypothesis_letter = current_hyp_alpa.get(correct_hyp_idx)

    if correct_hypothesis_letter is None:
        print(f"Warning: Could not map label index {correct_hyp_idx} to letter for lang_alpa '{lang_alpa}'.")
        return None, None, None, None, None, None # Invalid label or lang_alpa mapping

    correct_hypothesis_text = hyp1_text if correct_hyp_idx == 0 else hyp2_text

    return obs1, obs2, hyp1_text, hyp2_text, correct_hypothesis_letter, correct_hypothesis_text


def format_rule_generation_abductive_prompt(obs1, obs2, hyp_A_text, hyp_B_text, correct_hyp_letter, correct_hyp_text, lang_prompt):
    """Formats the prompt for rule generation in abductive reasoning."""
    template = RULE_GENERATION_PROMPT_ABDUCTIVE_AR if lang_prompt == 'ar' else RULE_GENERATION_PROMPT_ABDUCTIVE_EN
    return template.format(
        observation_1=obs1,
        observation_2=obs2,
        hypothesis_A_text=hyp_A_text,
        hypothesis_B_text=hyp_B_text,
        correct_hypothesis_letter=correct_hyp_letter,
        correct_hypothesis_text=correct_hyp_text
    )



def format_rule_verification_abductive_prompt(obs1, obs2, hyp_A_text, hyp_B_text, rule_to_verify, correct_hyp_letter, correct_hyp_text, lang_prompt):
    """Formats the prompt for rule verification in abductive reasoning."""
    template = RULE_VERIFICATION_PROMPT_ABDUCTIVE_AR if lang_prompt == 'ar' else RULE_VERIFICATION_PROMPT_ABDUCTIVE_EN
    return template.format(
        observation_1=obs1,
        observation_2=obs2,
        hypothesis_A_text=hyp_A_text,
        hypothesis_B_text=hyp_B_text,
        rule_to_verify=rule_to_verify,
        correct_hypothesis_letter=correct_hyp_letter,
        correct_hypothesis_text=correct_hyp_text
    )
# --- End Prompts and Formatters ---


def prepend_rule_library_to_prompt(prompt, rule_library):
    """
    Prepends the rule library to the deduction prompt.
    """
    rules_text = "\n".join(rule_library)
    return f"{rules_text}\n\n{prompt}"

# --- Groq LLM Interaction Function ---
def call_llm_for_induction(prompt_text, llm_config, max_retries=5, initial_backoff=2):
    """
    Calls the configured Groq LLM and returns its text response, with retry logic.
    Args:
        prompt_text (str): The prompt to send to the LLM.
        llm_config (dict): Configuration containing Groq 'client' and 'model_name'.
        max_retries (int): Maximum number of retries for API calls.
        initial_backoff (int): Initial backoff time in seconds.
    Returns:
        str: The LLM's text response, or None if an error occurs after retries.
    """
    if not GROQ_AVAILABLE:
        print("Error: Groq library is not available. Cannot make LLM calls.")
        return None

    llm_type = llm_config.get('type')
    if llm_type != 'groq':
        print(f"Error: This function is configured only for Groq, but LLM type is '{llm_type}'.")
        return None

    client = llm_config.get('client')
    model_name = llm_config.get('model_name')
    if not client or not model_name:
        print("Error: Groq client or model_name missing in llm_config.")
        return None

    current_attempt = 0
    backoff_time = initial_backoff

    # Debugging print statements (optional, can be commented out)
    # print(f"\n--- Calling Groq ({model_name}) ---")
    # print(f"Prompt (first 200 chars): {prompt_text[:200]}...")

    while current_attempt < max_retries:
        try:
            # Determine temperature based on prompt type (heuristic)
            # Use slightly higher temp for generation, lower for verification
            is_generation = "general rule or reasoning principle" in prompt_text.lower() or \
                            "ما هي القاعدة العامة أو مبدأ الاستدلال" in prompt_text
            temp = 0.2 if is_generation else 0.0
            # Determine max tokens based on prompt type
            max_tok = 500 if is_generation else 50 # Shorter for verification (Yes/No)

            # Make the API call to Groq
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_text}],
                model=model_name,
                temperature=temp,
                max_tokens=max_tok
            )
            # Extract the response text
            response = chat_completion.choices[0].message.content.strip()
            # Debugging print statement (optional)
            # print(f"Groq Response (Attempt {current_attempt+1}): {response[:200]}...")
            return response # Return successful response

        except RateLimitError as e:
            # Handle rate limit errors
            current_attempt += 1
            print(f"Groq Rate Limit Error (Attempt {current_attempt}/{max_retries}): {e}. Retrying in {backoff_time} seconds...")
        except APIError as e:
            # Handle other API errors (e.g., server errors)
            current_attempt += 1
            print(f"Groq API Error (Attempt {current_attempt}/{max_retries}): Status {e.status_code}, Message: {e.message}. Retrying in {backoff_time} seconds...")
        except Exception as e:
            # Handle unexpected errors during the API call
            current_attempt += 1
            print(f"Unexpected Groq Error (Attempt {current_attempt}/{max_retries}): {e}. Retrying in {backoff_time} seconds...")

        # If max retries reached, log failure and return None
        if current_attempt >= max_retries:
            print("LLM call failed after all retries.")
            return None

        # Wait before retrying with exponential backoff
        time.sleep(backoff_time)
        backoff_time = min(backoff_time * 2, 60) # Double backoff time, max 60 seconds

    return None # Should not be reached if loop logic is correct
# --- End Groq LLM Interaction ---

def parse_generated_rules(llm_output):
    """Parses rules from LLM output. Assumes each rule starts with 'Rule: '."""
    if not llm_output: return []
    rules = []
    # Handle potential variations in capitalization and spacing, multiline output
    rule_pattern = re.compile(r"^\s*Rule:\s*(.*)", re.IGNORECASE | re.MULTILINE)
    matches = rule_pattern.findall(llm_output)
    for match in matches:
        rule_text = match.strip()
        if rule_text: # Ensure rule is not empty after stripping
            rules.append(rule_text)
    return rules

def organize_rules_with_tags(rules):
    """
    Organizes rules hierarchically with XML tags for efficient retrieval.
    """
    tagged_rules = []
    for idx, rule in enumerate(rules):
        tag = f"<RuleCluster id='{idx}'>"
        tagged_rules.append(f"{tag}{rule['rule']}</RuleCluster>")
    return tagged_rules

def save_rule_library_with_tags(tagged_rules_list, output_file):
    """
    Saves the already tagged rule library (list of strings) to a file.
    """
    # tagged_rules = organize_rules_with_tags(rule_library) # Removed this line
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(tagged_rules_list)) # Use the passed list of strings directly


def parse_verification_response(llm_output):
    """Parses 'Yes' or 'No' from verification LLM output."""
    if not llm_output: return False
    # More robust check for variations like "Yes.", "yes!", "نعم" etc.
    cleaned_output = llm_output.strip().lower()
    # Check for English "yes" or Arabic "نعم" at the beginning
    return cleaned_output.startswith("yes") or cleaned_output.startswith("نعم")


def main():
    """Main function to run the rule induction process."""
    parser = argparse.ArgumentParser(description="Induce a rule library for Abductive Reasoning using H->T (Groq Focused).")
    parser.add_argument("--training_data_file", type=str, required=True, help="Path to the training data CSV file for abductive reasoning (e.g., obs1, obs2, hyp1, hyp2, label).")
    parser.add_argument("--output_rule_library_file", type=str, default="results/abductive_rule_library.json", help="Path to save the induced rule library.")
    parser.add_argument("--output_folder", type=str, default="results", help="Folder to save the rule library.")

    # Groq Configuration Arguments
    parser.add_argument("--groq_model", type=str, default="llama3-70b-8192", help="Groq model ID.")

    # Language Arguments
    parser.add_argument("--lang_prompt", type=str, default="en", choices=["en", "ar"], help="Language of the prompts for rule induction.")
    parser.add_argument("--lang_alpa", type=str, default="en", choices=["en", "ar"], help="Language of hypothesis labels (A/B vs أ/ب).")

    # Filtering Arguments
    parser.add_argument("--min_coverage", type=int, default=2, help="Minimum number of times a rule must occur.")
    parser.add_argument("--min_confidence", type=float, default=0.75, help="Minimum confidence (correct_associations / occurrences) for a rule.")
    parser.add_argument("--similarity_threshold", type=float, default=0.9, help="Cosine similarity threshold for grouping rules (0.0 to 1.0). Effective only if sentence-transformers is available. Set to 1.0 for exact matching if SBERT is available.")
    
    # Processing Control
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum training examples to process (for testing).")
    # SBERT model name argument
    parser.add_argument("--sbert_model_name", type=str, default="all-MiniLM-L6-v2", help="Name of the Sentence Transformer model to use for semantic similarity.")


    args = parser.parse_args()

    # Check if Groq library is actually available if needed
    if not GROQ_AVAILABLE:
        print("Error: Groq library is required but not installed. Please run `pip install groq httpx`.")
        return

    # --- Initialize Sentence Transformer Model ---
    sbert_model = None
    if SENTENCE_TRANSFORMER_AVAILABLE:
        try:
            sbert_model = SentenceTransformer(args.sbert_model_name)
            print(f"Sentence Transformer model '{args.sbert_model_name}' loaded.")
        except Exception as e:
            print(f"Warning: Could not load Sentence Transformer model '{args.sbert_model_name}': {e}. Falling back to exact string matching.")
            # SENTENCE_TRANSFORMER_AVAILABLE = False # Effectively disable if model load fails
            sbert_model = None # Ensure model is None
    else:
        print("Info: 'sentence-transformers' library not installed. Using exact string matching for rules.")


    # Ensure output directory exists
    os.makedirs(args.output_folder, exist_ok=True)
    # Construct full output path
    args.output_rule_library_file = os.path.join(args.output_folder, os.path.basename(args.output_rule_library_file))

    # --- Rule Induction Process ---
    # rule_stats = defaultdict(...) # Old
    rule_clusters = {} # New: canonical_rule_text -> {occurrence, correct_association, task_types, embedding}
    task_name = "abductive_reasoning" 

    print("Starting rule induction for abductive reasoning...")
    # ... (Load Training Data as before) ...
    try:
        df_train = pd.read_csv(args.training_data_file)
        required_cols = ['observation_1', 'observation_2', 'hypothesis_1', 'hypothesis_2', 'label']
        if not all(col in df_train.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_train.columns]
            print(f"Error: Training data file '{args.training_data_file}' missing required columns: {missing}")
            return
        if args.max_examples:
            df_train = df_train.head(args.max_examples)
        print(f"Loaded {len(df_train)} abductive reasoning examples from {args.training_data_file}")
    except FileNotFoundError:
        print(f"Error: Training data file not found at {args.training_data_file}")
        return
    except Exception as e:
        print(f"Error loading training data: {e}"); return
    
    # --- Initialize Groq LLM ---
    llm_config = {'type': 'groq', 'model_name': args.groq_model}
    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key: raise ValueError("GROQ_API_KEY environment variable not set.")
        timeout_config = httpx.Timeout(60.0, read=300.0) 
        llm_config['client'] = Groq(api_key=groq_api_key, timeout=timeout_config)
        print(f"Groq client initialized for model: {args.groq_model}")
    except Exception as e:
        print(f"Error initializing Groq: {e}"); return


    processed_count = 0
    skipped_count = 0
    # Iterate through training data with a progress bar
    for index, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Processing Examples"):
        obs1, obs2, hyp1_text, hyp2_text, correct_hyp_letter, correct_hyp_text = \
            get_abductive_data_from_row(row, args.lang_alpa)

        if not correct_hyp_letter:
            skipped_count += 1
            continue

        gen_prompt = format_rule_generation_abductive_prompt(
            obs1, obs2, hyp1_text, hyp2_text, correct_hyp_letter, correct_hyp_text, args.lang_prompt
        )
        llm_gen_output = call_llm_for_induction(gen_prompt, llm_config)
        if not llm_gen_output:
            skipped_count += 1
            continue

        generated_rules_text_list = parse_generated_rules(llm_gen_output)
        if not generated_rules_text_list:
            processed_count += 1
            continue

        current_rules_embeddings = None
        if sbert_model and generated_rules_text_list: # Check sbert_model is loaded
            # Prepare texts for embedding: ensure they are stripped
            texts_to_embed = [r.strip() for r in generated_rules_text_list if r.strip()]
            if texts_to_embed:
                current_rules_embeddings = sbert_model.encode(texts_to_embed, convert_to_tensor=True)

        for i, new_rule_original_text in enumerate(generated_rules_text_list):
            new_rule_text = new_rule_original_text.strip()
            if not new_rule_text:
                continue

            current_rule_embedding = None
            if sbert_model and current_rules_embeddings is not None and i < len(current_rules_embeddings):
                 # Check if texts_to_embed was non-empty and embeddings were generated
                if texts_to_embed and new_rule_text == texts_to_embed[i]: # Ensure correct mapping if some rules were empty
                    current_rule_embedding = current_rules_embeddings[i]
                elif not texts_to_embed and len(generated_rules_text_list) == len(current_rules_embeddings): # Fallback if all rules were empty then stripped
                     current_rule_embedding = current_rules_embeddings[i]


            matched_canonical_key = None
            highest_similarity_score = -1.0

            if sbert_model and current_rule_embedding is not None and rule_clusters:
                candidate_keys = []
                candidate_embeddings = []
                for key, data in rule_clusters.items():
                    if data.get("embedding") is not None:
                        candidate_keys.append(key)
                        candidate_embeddings.append(data["embedding"])
                
                if candidate_embeddings:
                    stacked_candidate_embeddings = torch.stack(candidate_embeddings)
                    cosine_scores = util.pytorch_cos_sim(current_rule_embedding.unsqueeze(0), stacked_candidate_embeddings)[0]

                    for j, score_tensor in enumerate(cosine_scores):
                        score = score_tensor.item()
                        if score > highest_similarity_score:
                            if score >= args.similarity_threshold:
                                highest_similarity_score = score
                                matched_canonical_key = candidate_keys[j]
                            # If score is highest but below threshold, it's not a match for this rule.
                            # We are looking for the best match *above* the threshold.
            
            target_key_for_stats = None
            text_for_verification_prompt = None

            if matched_canonical_key: # A semantic match was found above threshold
                target_key_for_stats = matched_canonical_key
                text_for_verification_prompt = matched_canonical_key # Verify using the canonical text
            else: # No semantic match, or SBERT not used/available. Use exact text.
                target_key_for_stats = new_rule_text
                text_for_verification_prompt = new_rule_text
                if new_rule_text not in rule_clusters:
                    rule_clusters[new_rule_text] = {
                        "occurrence": 0, # Will be incremented shortly
                        "correct_association": 0,
                        "task_types": set(),
                        "embedding": current_rule_embedding if sbert_model else None
                    }
            
            # Increment occurrence for the target cluster/rule
            rule_clusters[target_key_for_stats]["occurrence"] += 1
            rule_clusters[target_key_for_stats]["task_types"].add(task_name)
            if sbert_model and current_rule_embedding is not None and rule_clusters[target_key_for_stats]["embedding"] is None:
                 rule_clusters[target_key_for_stats]["embedding"] = current_rule_embedding


            # Verify the rule (using text_for_verification_prompt)
            ver_prompt = format_rule_verification_abductive_prompt(
                obs1, obs2, hyp1_text, hyp2_text, text_for_verification_prompt, correct_hyp_letter, correct_hyp_text, args.lang_prompt
            )
            llm_ver_output = call_llm_for_induction(ver_prompt, llm_config)
            if llm_ver_output is not None and parse_verification_response(llm_ver_output):
                rule_clusters[target_key_for_stats]["correct_association"] += 1
        
        processed_count += 1

    print(f"\nFinished processing examples. Processed: {processed_count}, Skipped: {skipped_count}")
    print(f"Total unique rule clusters/rules generated before filtering: {len(rule_clusters)}")

    # --- Filter Rules Based on Coverage and Confidence ---
    final_rule_library = []
    print("\nFiltering rules...")
    filtered_out_count = 0
    for rule_text_key, stats in rule_clusters.items(): # Iterate over rule_clusters
        occurrence = stats["occurrence"]
        correct_association = stats["correct_association"]
        confidence = (correct_association / occurrence) if occurrence > 0 else 0

        if occurrence >= args.min_coverage and confidence >= args.min_confidence:
            final_rule_library.append({
                "rule": rule_text_key, # This is the canonical rule text
                "coverage": occurrence,
                "confidence": round(confidence, 4),
                "task_types": sorted(list(stats["task_types"]))
                # Embedding is not typically saved in the final JSON unless needed downstream
            })
        else:
            filtered_out_count += 1
    print(f"Filtered out {filtered_out_count} rules.")
    
    final_rule_library.sort(key=lambda x: (x['confidence'], x['coverage']), reverse=True)

    # --- Save the Final Rule Library (JSON) ---
    try:
        with open(args.output_rule_library_file, 'w', encoding='utf-8') as f:
            json.dump(final_rule_library, f, ensure_ascii=False, indent=4)
        print(f"\nSuccessfully saved {len(final_rule_library)} rules to {args.output_rule_library_file}")
        if not final_rule_library and processed_count > 0:
            print(f"Warning: No rules met the filtering criteria (Min Coverage: {args.min_coverage}, Min Confidence: {args.min_confidence}).")
    except Exception as e:
        print(f"Error saving rule library to {args.output_rule_library_file}: {e}")
    
    # --- Save Tagged Rule Library (XML-like) ---
    # Ensure final_rule_library is used for tagging
    if final_rule_library: # Only proceed if there are rules
        tagged_rule_library_for_xml = organize_rules_with_tags(final_rule_library) # Pass the list of dicts
        tagged_output_file = args.output_rule_library_file.replace(".json", "_tagged.xml")
        save_rule_library_with_tags(tagged_rule_library_for_xml, tagged_output_file) # Pass the list of strings
        print(f"Tagged rule library saved to {tagged_output_file}")
    elif processed_count > 0 : # If rules were processed but none saved
        print(f"No rules to save in tagged XML format as final_rule_library is empty.")


if __name__ == "__main__":
    main()
