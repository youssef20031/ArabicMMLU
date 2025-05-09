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
             print(f"Warning: Invalid label '{label}' found. Expected '0' or '1'. Skipping row.")
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

def save_rule_library_with_tags(rule_library, output_file):
    """
    Saves the rule library with XML tags to a file.
    """
    tagged_rules = organize_rules_with_tags(rule_library)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(tagged_rules))

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
    
    # Processing Control
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum training examples to process (for testing).")

    args = parser.parse_args()

    # Check if Groq library is actually available if needed
    if not GROQ_AVAILABLE:
        print("Error: Groq library is required but not installed. Please run `pip install groq httpx`.")
        return

    # Ensure output directory exists
    os.makedirs(args.output_folder, exist_ok=True)
    # Construct full output path
    args.output_rule_library_file = os.path.join(args.output_folder, os.path.basename(args.output_rule_library_file))

    # Initialize final_rule_library as an empty list
    final_rule_library = []

    # Organize rules with XML tags
    tagged_rule_library = organize_rules_with_tags(final_rule_library)

    # Save the tagged rule library
    tagged_output_file = args.output_rule_library_file.replace(".json", "_tagged.xml")
    save_rule_library_with_tags(tagged_rule_library, tagged_output_file)
    print(f"Tagged rule library saved to {tagged_output_file}")

    # --- Initialize Groq LLM ---
    llm_config = {'type': 'groq', 'model_name': args.groq_model}
    try:
        # Retrieve API key from environment variable
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key: raise ValueError("GROQ_API_KEY environment variable not set.")
        # Configure longer timeouts suitable for potentially long induction process
        timeout_config = httpx.Timeout(60.0, read=300.0) # 60s connect, 300s read timeout
        llm_config['client'] = Groq(api_key=groq_api_key, timeout=timeout_config)
        print(f"Groq client initialized for model: {args.groq_model}")
    except Exception as e:
        print(f"Error initializing Groq: {e}"); return

    # --- Load Training Data ---
    try:
        df_train = pd.read_csv(args.training_data_file)
        # Validate required columns exist in the dataframe
        required_cols = ['observation_1', 'observation_2', 'hypothesis_1', 'hypothesis_2', 'label']
        if not all(col in df_train.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_train.columns]
            print(f"Error: Training data file '{args.training_data_file}' missing required columns: {missing}")
            return

        # Limit examples if specified
        if args.max_examples:
            df_train = df_train.head(args.max_examples)
        print(f"Loaded {len(df_train)} abductive reasoning examples from {args.training_data_file}")
    except FileNotFoundError:
        print(f"Error: Training data file not found at {args.training_data_file}")
        return
    except Exception as e:
        print(f"Error loading training data: {e}"); return

    # --- Rule Induction Process ---
    # Dictionary to store rule statistics: rule_text -> {occurrence, correct_association, task_types}
    rule_stats = defaultdict(lambda: {"occurrence": 0, "correct_association": 0, "task_types": set()})
    task_name = "abductive_reasoning" # Task identifier for the rules

    print("Starting rule induction for abductive reasoning...")
    processed_count = 0
    skipped_count = 0
    # Iterate through training data with a progress bar
    for index, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Processing Examples"):
        # Extract data for the current example
        obs1, obs2, hyp1_text, hyp2_text, correct_hyp_letter, correct_hyp_text = \
            get_abductive_data_from_row(row, args.lang_alpa)

        # Skip row if essential data is missing or invalid
        if not correct_hyp_letter:
            skipped_count += 1
            continue

        # 1. Generate Rules using LLM
        gen_prompt = format_rule_generation_abductive_prompt(
            obs1, obs2, hyp1_text, hyp2_text, correct_hyp_letter, correct_hyp_text, args.lang_prompt
        )
        llm_gen_output = call_llm_for_induction(gen_prompt, llm_config)
        if not llm_gen_output:
            skipped_count += 1
            continue # Skip if LLM fails to generate output

        # Parse the generated rules from the LLM response
        generated_rules = parse_generated_rules(llm_gen_output)
        if not generated_rules:
            # No rules were parsed, still count as processed
            processed_count += 1
            continue

        # 2. Verify Each Generated Rule
        rules_verified_for_example = 0
        for rule_text in generated_rules:
            # Skip empty rules
            if not rule_text.strip(): continue

            # Update occurrence count and associated task type
            rule_stats[rule_text]["occurrence"] += 1
            rule_stats[rule_text]["task_types"].add(task_name)

            # Format the verification prompt
            ver_prompt = format_rule_verification_abductive_prompt(
                obs1, obs2, hyp1_text, hyp2_text, rule_text, correct_hyp_letter, correct_hyp_text, args.lang_prompt
            )
            # Call LLM for verification
            llm_ver_output = call_llm_for_induction(ver_prompt, llm_config)

            # Update correct association count if verification is successful ('Yes')
            if llm_ver_output is not None and parse_verification_response(llm_ver_output):
                rule_stats[rule_text]["correct_association"] += 1
                rules_verified_for_example += 1
            # Optional logging for failed verification
            # else:
            #    print(f"Rule not verified: '{rule_text}' | LLM Output: '{llm_ver_output}'")

        processed_count += 1 # Increment count after processing an example

    # Print summary statistics after processing all examples
    print(f"\nFinished processing examples. Processed: {processed_count}, Skipped: {skipped_count}")
    print(f"Total unique rules generated before filtering: {len(rule_stats)}")

    # --- Filter Rules Based on Coverage and Confidence ---
    final_rule_library = []
    print("\nFiltering rules...")
    filtered_out_count = 0
    for rule_text, stats in rule_stats.items():
        occurrence = stats["occurrence"]
        correct_association = stats["correct_association"]
        # Calculate confidence, handling division by zero
        confidence = (correct_association / occurrence) if occurrence > 0 else 0

        # Apply filtering criteria
        if occurrence >= args.min_coverage and confidence >= args.min_confidence:
            final_rule_library.append({
                "rule": rule_text,
                "coverage": occurrence,
                "confidence": round(confidence, 4),
                "task_types": sorted(list(stats["task_types"])) # Store associated task types
            })
        else:
            filtered_out_count += 1
            # Optional logging for filtered out rules
            # print(f"Filtering out rule: '{rule_text}' (Occ: {occurrence}, Conf: {confidence:.2f})")

    print(f"Filtered out {filtered_out_count} rules.")
    
    # Sort the final library by confidence, then coverage (descending)
    final_rule_library.sort(key=lambda x: (x['confidence'], x['coverage']), reverse=True)

        # --- Save the Final Rule Library ---
    try:
        # Save as JSON file with indentation for readability
        with open(args.output_rule_library_file, 'w', encoding='utf-8') as f:
            json.dump(final_rule_library, f, ensure_ascii=False, indent=4)
        print(f"\nSuccessfully saved {len(final_rule_library)} rules to {args.output_rule_library_file}")
        # Provide a warning if no rules met the criteria
        if not final_rule_library and processed_count > 0:
            print(f"Warning: No rules met the filtering criteria (Min Coverage: {args.min_coverage}, Min Confidence: {args.min_confidence}).")
    except Exception as e:
        print(f"Error saving rule library to {args.output_rule_library_file}: {e}")
    
    # Ensure final_rule_library is initialized as an empty list if no rules were generated
    if 'final_rule_library' not in locals():
        final_rule_library = []
    
    # Organize rules with XML tags
    tagged_rule_library = organize_rules_with_tags(final_rule_library)
    
    # Save the tagged rule library
    tagged_output_file = args.output_rule_library_file.replace(".json", "_tagged.xml")
    save_rule_library_with_tags(tagged_rule_library, tagged_output_file)
    print(f"Tagged rule library saved to {tagged_output_file}")
        


def test_with_rule_library(test_data, rule_library, llm_config):
    """
    Tests the model using the rule library for deduction.
    """
    for example in test_data:
        prompt = format_rule_generation_abductive_prompt(
            example['obs1'], example['obs2'], example['hyp1'], example['hyp2'],
            example['correct_hyp_letter'], example['correct_hyp_text'], 'en'
        )
        augmented_prompt = prepend_rule_library_to_prompt(prompt, rule_library)
        response = call_llm_for_induction(augmented_prompt, llm_config)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()
