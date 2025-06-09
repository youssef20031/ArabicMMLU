# ==============================================================================
# Rule Induction Script (Refactored)
#
# This script is designed to induce a library of reasoning rules from a given
# dataset. It follows a structured, class-based approach inspired by modern
# ML training scripts for better organization and scalability.
#
# The process involves:
# 1. Loading configuration from command-line arguments.
# 2. Setting up a dedicated working directory and logger.
# 3. Instantiating a dataset handler for the specific reasoning task (abductive/deductive).
# 4. Initializing an LLM client (e.g., for Groq).
# 5. Using a PromptController to manage the interaction with the LLM, which:
#    a. Generates potential rules based on data samples.
#    b. Verifies each generated rule for its correctness using the LLM.
# 6. Updating a RuleLibrary with the verified rules, which:
#    a. Clusters semantically similar rules using sentence transformers.
#    b. Tracks statistics like coverage and confidence for each rule cluster.
# 7. Periodically saving the filtered, high-quality rule library to disk.
# ==============================================================================

import argparse
import pandas as pd
import os
import json
import re
import time
import logging
import pprint
import random
from tqdm import tqdm

# --- Dependency Availability Checks ---
# Attempt to import Groq and related errors
try:
    from groq import Groq, RateLimitError, APIError
    import httpx  # Required by Groq client for timeouts
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    # Define dummy classes if Groq is not installed for basic script parsing
    class Groq: pass
    class RateLimitError(Exception): pass
    class APIError(Exception): pass
    class httpx: Timeout = object

# Attempt to import Sentence Transformers and PyTorch
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    # Define dummy classes if not available
    class SentenceTransformer: pass
    class util: pass
    class torch: pass
    print("Warning: 'sentence-transformers' or 'torch' not found. Rule clustering will use exact string matching.")

# --- Prompts ---
PROMPTS = {
    "abductive_generation_en": (
        "Given the following observations and two hypotheses, where one hypothesis is known to be more plausible:\n\n"
        "Observation 1: {observation_1}\n"
        "Observation 2: {observation_2}\n\n"
        "Hypothesis A: {hypothesis_A_text}\n"
        "Hypothesis B: {hypothesis_B_text}\n\n"
        "The more plausible hypothesis is: Hypothesis {correct_hypothesis_letter} ({correct_hypothesis_text})\n\n"
        "What is a general rule or reasoning principle that explains why Hypothesis {correct_hypothesis_letter} is more plausible? "
        "The rule should be concise and broadly applicable. "
        "Output the rule directly, starting with 'Rule: ' and then the rule text on the same line. "
        "If you can identify multiple distinct rules, output each on a new line, each starting with 'Rule: '."
    ),
    "abductive_verification_en": (
        "Consider the following observations and hypotheses:\n\n"
        "Observation 1: {observation_1}\n"
        "Observation 2: {observation_2}\n\n"
        "Hypothesis A: {hypothesis_A_text}\n"
        "Hypothesis B: {hypothesis_B_text}\n\n"
        "The known more plausible hypothesis is: Hypothesis {correct_hypothesis_letter} ({correct_hypothesis_text})\n\n"
        "Now, consider the following rule: \"{rule_to_verify}\"\n\n"
        "If you strictly apply ONLY this rule to the observations and hypotheses, does it help you correctly identify Hypothesis {correct_hypothesis_letter} as the more plausible one? "
        "Answer with only 'Yes' or 'No'."
    ),
    "deductive_generation_en": (
        "Given the following question and options, where one option is the correct answer:\n\n"
        "Question: {question_text}\n"
        "Options:\n{options_formatted_text}\n"
        "The correct answer is: Option {correct_option_letter} ({correct_option_text})\n\n"
        "What is a general rule that explains why Option {correct_option_letter} is the correct answer? "
        "The rule should be concise and broadly applicable. "
        "Output the rule directly, starting with 'Rule: '."
    ),
    "deductive_verification_en": (
        "Consider the following question and options:\n\n"
        "Question: {question_text}\n"
        "Options:\n{options_formatted_text}\n"
        "The known correct answer is: Option {correct_option_letter} ({correct_option_text})\n\n"
        "Now, consider the following rule: \"{rule_to_verify}\"\n\n"
        "If you strictly apply ONLY this rule, does it help you correctly identify Option {correct_option_letter} as the correct answer? "
        "Answer with only 'Yes' or 'No'."
    ),
    "abductive_generation_ar": (
        "بالنظر إلى الملاحظات والفرضيتين التاليتين، حيث من المعروف أن إحدى الفرضيات أكثر قبولاً:\n\n"
        "الملاحظة الأولى: {observation_1}\n"
        "الملاحظة الثانية: {observation_2}\n\n"
        "الفرضية أ: {hypothesis_A_text}\n"
        "الفرضية ب: {hypothesis_B_text}\n\n"
        "الفرضية الأكثر قبولاً هي: الفرضية {correct_hypothesis_letter} ({correct_hypothesis_text})\n\n"
        "ما هي القاعدة العامة أو مبدأ الاستدلال الذي يفسر لماذا الفرضية {correct_hypothesis_letter} هي أكثر قبولاً من الأخرى، بناءً على الملاحظات؟ "
        "يجب أن تكون القاعدة موجزة وقابلة للتطبيق على نطاق واسع إن أمكن. "
        "أخرج القاعدة مباشرة، بادئًا بـ 'Rule: ' ثم نص القاعدة على نفس السطر. "
        "إذا كان بإمكانك تحديد قواعد متعددة ومتميزة، فأخرج كل قاعدة على سطر جديد، تبدأ كل منها بـ 'Rule: '."
    ),
    "abductive_verification_ar": (
        "بالنظر إلى الملاحظات والفرضيات التالية:\n\n"
        "الملاحظة الأولى: {observation_1}\n"
        "الملاحظة الثانية: {observation_2}\n\n"
        "الفرضية أ: {hypothesis_A_text}\n"
        "الفرضية ب: {hypothesis_B_text}\n\n"
        "الفرضية المعروفة الأكثر قبولاً هي: الفرضية {correct_hypothesis_letter} ({correct_hypothesis_text})\n\n"
        "الآن، ضع في اعتبارك القاعدة التالية: \"{rule_to_verify}\"\n\n"
        "إذا طبقت هذه القاعدة فقط بصرامة على الملاحظات والفرضيات، فهل تساعدك في تحديد الفرضية {correct_hypothesis_letter} بشكل صحيح على أنها الأكثر قبولاً؟ "
        "أجب فقط بـ 'نعم' أو 'لا'."
    ),
    "deductive_generation_ar": (
        "بالنظر إلى السؤال والخيارات التالية، حيث يكون أحد الخيارات هو الإجابة الصحيحة:\n\n"
        "السؤال: {question_text}\n"
        "الخيارات:\n{options_formatted_text}\n"
        "الإجابة الصحيحة هي: الخيار {correct_option_letter} ({correct_option_text})\n\n"
        "ما هي القاعدة العامة أو مبدأ الاستدلال الذي يفسر لماذا الخيار {correct_option_letter} هو الإجابة الصحيحة، بناءً على السؤال والخيارات؟ "
        "يجب أن تكون القاعدة موجزة وقابلة للتطبيق على نطاق واسع إن أمكن. "
        "أخرج القاعدة مباشرة، بادئًا بـ 'Rule: ' ثم نص القاعدة على نفس السطر."
    ),
    "deductive_verification_ar": (
        "بالنظر إلى السؤال والخيارات التالية:\n\n"
        "السؤال: {question_text}\n"
        "الخيارات:\n{options_formatted_text}\n"
        "الإجابة الصحيحة المعروفة هي: الخيار {correct_option_letter} ({correct_option_text})\n\n"
        "الآن، ضع في اعتبارك القاعدة التالية: \"{rule_to_verify}\"\n\n"
        "إذا طبقت هذه القاعدة فقط بصرامة على السؤال والخيارات، فهل تساعدك في تحديد الخيار {correct_option_letter} بشكل صحيح على أنه الإجابة الصحيحة؟ "
        "أجب فقط بـ 'نعم' أو 'لا'."
    ),
}

# --- Utility Functions ---
def create_logger(log_dir, name="inducer"):
    """Creates a logger to save output to a file in the working directory."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_file = os.path.join(log_dir, f"{name}_run.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    return logger

# --- Model Classes ---
class GroqModel:
    """A wrapper for the Groq API client."""
    def __init__(self, model_name, max_tokens=1024, max_retries=5, initial_backoff=2):
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library is required. Please run `pip install groq`.")
        try:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            timeout_config = httpx.Timeout(60.0, read=300.0)
            self.client = Groq(api_key=api_key, timeout=timeout_config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {e}")

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    def __call__(self, prompt_text, temperature=0.1, max_tokens_override=None):
        """Calls the Groq API with retry logic."""
        current_attempt = 0
        backoff_time = self.initial_backoff
        while current_attempt < self.max_retries:
            try:
                max_tok = max_tokens_override if max_tokens_override is not None else self.max_tokens
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_text}],
                    model=self.model_name,
                    temperature=temperature,
                    max_tokens=max_tok
                )
                response = chat_completion.choices[0].message.content.strip()
                return response, 0.0
            except (RateLimitError, APIError, Exception) as e:
                current_attempt += 1
                print(f"Groq API Error (Attempt {current_attempt}/{self.max_retries}): {e}. Retrying in {backoff_time}s...")
                if current_attempt >= self.max_retries:
                    print("LLM call failed after all retries.")
                    return None, 0.0
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, 60)
        return None, 0.0

# --- Dataset Classes ---
class BaseDataset:
    """Base class for datasets."""
    def __init__(self, path):
        self.path = path
        self.data = None
    
    def get_split(self):
        raise NotImplementedError
    
    def _load_data(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset file not found at {self.path}")
        self.data = pd.read_csv(self.path)
        # Return python native types
        self.data = self.data.astype(object).where(pd.notnull(self.data), None)

class AbductiveDataset(BaseDataset):
    """Handles abductive reasoning datasets."""
    def get_split(self):
        if self.data is None: self._load_data()
        samples = [row.to_dict() for _, row in self.data.iterrows()]
        return samples

class DeductiveDataset(BaseDataset):
    """Handles deductive reasoning datasets."""
    def get_split(self):
        if self.data is None: self._load_data()
        samples = [row.to_dict() for _, row in self.data.iterrows()]
        return samples

# --- Rule Library and Prompt Controller ---
class RuleLibrary:
    """Manages the collection, clustering, and filtering of rules."""
    def __init__(self, args):
        self.args = args
        self.sbert_model = None
        if SENTENCE_TRANSFORMER_AVAILABLE and args.similarity_threshold < 1.0:
            try:
                self.sbert_model = SentenceTransformer(args.sbert_model_name)
                print(f"Sentence Transformer model '{args.sbert_model_name}' loaded.")
            except Exception as e:
                print(f"Warning: Could not load SBERT model '{args.sbert_model_name}'. Defaulting to exact matching. Error: {e}")
                self.sbert_model = None
        self.clusters = {}

    def _find_matching_rule(self, rule_text, rule_embedding):
        if not self.sbert_model or rule_embedding is None or not self.clusters:
            return rule_text

        candidates = [(k, v["embedding"]) for k, v in self.clusters.items() if v.get("embedding") is not None]
        if not candidates:
            return rule_text
        
        candidate_keys, candidate_embeddings = zip(*candidates)
        cosine_scores = util.pytorch_cos_sim(rule_embedding.unsqueeze(0), torch.stack(list(candidate_embeddings)))[0]
        best_match_idx = cosine_scores.argmax()
        
        if cosine_scores[best_match_idx].item() >= self.args.similarity_threshold:
            return candidate_keys[best_match_idx]
        
        return rule_text

    def update(self, rule_text, is_verified, task_name):
        if not rule_text: return
        rule_embedding = self.sbert_model.encode(rule_text, convert_to_tensor=True) if self.sbert_model else None
        canonical_key = self._find_matching_rule(rule_text, rule_embedding)
        
        if canonical_key not in self.clusters:
            self.clusters[canonical_key] = {"occurrence": 0, "correct_association": 0, "task_types": set(), "embedding": rule_embedding}
        
        self.clusters[canonical_key]["occurrence"] += 1
        self.clusters[canonical_key]["task_types"].add(task_name)
        if is_verified:
            self.clusters[canonical_key]["correct_association"] += 1

    def get_filtered_rules(self):
        final_rules = []
        for rule, stats in self.clusters.items():
            occurrence = stats["occurrence"]
            confidence = (stats["correct_association"] / occurrence) if occurrence > 0 else 0
            if occurrence >= self.args.min_coverage and confidence >= self.args.min_confidence:
                final_rules.append({"rule": rule, "coverage": occurrence, "confidence": round(confidence, 4), "task_types": sorted(list(stats["task_types"]))})
        
        final_rules.sort(key=lambda x: (x['confidence'], x['coverage']), reverse=True)
        return final_rules[:self.args.max_rules] if self.args.max_rules is not None else final_rules

    def save(self, file_path):
        filtered_rules = self.get_filtered_rules()
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_rules, f, ensure_ascii=False, indent=4)
        except Exception as e: print(f"Error saving rule library: {e}")
        return len(filtered_rules)

class PromptController:
    """Orchestrates LLM interactions."""
    def __init__(self, args):
        self.args = args
        self.hyp_map = {'en': {1: 'A', 2: 'B'}, 'ar': {1: 'أ', 2: 'ب'}}
        self.opt_map = {'أ': 'A', 'ب': 'B', 'ج': 'C', 'د': 'D'}
        self.opt_map_ar_display = {'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د'}

    def _parse(self, output, pattern): return re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
    def _parse_verification(self, output): 
        if not output: return False
        return output.strip().lower().startswith("yes") or output.strip().lower().startswith("نعم")

    def __call__(self, model, sample, library, logger):
        prompt_data = self._prepare_prompt_data(sample)
        if not prompt_data:
            logger.warning(f"Skipping sample due to invalid data: {sample}")
            return
        
        gen_prompt_key = f"{self.args.task_type}_generation_{self.args.lang_prompt}"
        gen_prompt = PROMPTS[gen_prompt_key].format(**prompt_data)
        gen_output, _ = model(gen_prompt, temperature=0.2, max_tokens_override=500)
        generated_rules = self._parse(gen_output, r"^\s*Rule:\s*(.*)")
        
        if not generated_rules: return

        for rule_text in generated_rules:
            rule_text = rule_text.strip()
            if not rule_text: continue
            
            ver_prompt_key = f"{self.args.task_type}_verification_{self.args.lang_prompt}"
            ver_prompt = PROMPTS[ver_prompt_key].format(**prompt_data, rule_to_verify=rule_text)
            ver_output, _ = model(ver_prompt, temperature=0.0, max_tokens_override=50)
            is_verified = self._parse_verification(ver_output)
            library.update(rule_text, is_verified, self.args.task_type)
            logger.info(f"Rule: '{rule_text}' | Verified: {is_verified}")

    def _prepare_prompt_data(self, sample):
        try:
            if self.args.task_type == "abductive":
                hyp1_text, hyp2_text = str(sample['hypothesis_1']), str(sample['hypothesis_2'])
                correct_idx = int(sample['label'])
                correct_hyp_letter = self.hyp_map[self.args.lang_alpa][correct_idx]
                return {
                    "observation_1": str(sample['observation_1']), "observation_2": str(sample['observation_2']),
                    "hypothesis_A_text": hyp1_text, "hypothesis_B_text": hyp2_text,
                    "correct_hypothesis_letter": correct_hyp_letter,
                    "correct_hypothesis_text": hyp1_text if correct_idx == 1 else hyp2_text
                }
            elif self.args.task_type == "deductive":
                options = { 'A': str(sample['option_a']), 'B': str(sample['option_b']), 'C': str(sample['option_c']), 'D': str(sample['option_d']) }
                answer_letter = str(sample['answer']).strip()
                norm_letter = self.opt_map.get(answer_letter, answer_letter.upper())
                
                display_options = []
                if self.args.lang_prompt == 'ar':
                    for k, v in options.items(): display_options.append(f"{self.opt_map_ar_display[k]}: {v}")
                else:
                    for k, v in options.items(): display_options.append(f"{k}: {v}")

                correct_option_letter_display = self.opt_map_ar_display.get(norm_letter, norm_letter) if self.args.lang_prompt == 'ar' else norm_letter

                return {
                    "question_text": str(sample['question']),
                    "options_formatted_text": "\n".join(display_options),
                    "correct_option_letter": correct_option_letter_display,
                    "correct_option_text": options[norm_letter]
                }
        except (KeyError, ValueError) as e:
            print(f"Error preparing prompt data for sample {sample}: {e}")
            return None
        return None

# --- Main Execution ---
def main():
    """Main function to run the rule induction process."""
    random.seed(0)
    parser = argparse.ArgumentParser(description="Induce a rule library for Reasoning using H->T (Groq Focused).")
    parser.add_argument("--task_type", type=str, default="abductive", choices=["abductive", "deductive"], help="Type of reasoning task.")
    parser.add_argument("--abductive_data_file", type=str, help="Path to CSV for abductive reasoning.")
    parser.add_argument("--deductive_data_file", type=str, help="Path to CSV for deductive reasoning.")
    parser.add_argument("--output_folder", type=str, default="results", help="Folder to save the rule library.")
    parser.add_argument("--output_rule_library_file", type=str, default="rule_library.json", help="Filename for the induced rule library.")
    parser.add_argument("--groq_model", type=str, default="llama3-70b-8192", help="Groq model ID.")
    parser.add_argument("--lang_prompt", type=str, default="en", choices=["en", "ar"], help="Language of the prompts.")
    parser.add_argument("--lang_alpa", type=str, default="en", choices=["en", "ar"], help="Language of hypothesis labels (A/B vs أ/ب).")
    parser.add_argument("--min_coverage", type=int, default=2, help="Minimum number of times a rule must occur.")
    parser.add_argument("--min_confidence", type=float, default=0.75, help="Minimum confidence for a rule.")
    parser.add_argument("--similarity_threshold", type=float, default=0.9, help="Cosine similarity threshold for grouping rules.")
    parser.add_argument("--num_iterations", type=int, default=500, help="Total number of processing iterations.")
    parser.add_argument("--save_interval", type=int, default=100, help="Save library every N iterations.")
    parser.add_argument("--sbert_model_name", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model name.")
    parser.add_argument("--max_rules", type=int, default=None, help="Maximum number of rules in the final library.")
    parser.add_argument("--focus_only_deductive", action="store_true", help="Overrides task_type to 'deductive'.")
    args = parser.parse_args()

    if args.focus_only_deductive: args.task_type = 'deductive'
    os.makedirs(args.output_folder, exist_ok=True)
    logger = create_logger(args.output_folder)
    logger.info("--- Configuration ---\n" + pprint.pformat(vars(args)) + "\n---------------------\n")
    
    data_file = args.deductive_data_file if args.task_type == "deductive" else args.abductive_data_file
    if not data_file: raise ValueError(f"--{args.task_type}_data_file is required.")
    
    dataset = DeductiveDataset(data_file) if args.task_type == "deductive" else AbductiveDataset(data_file)
    model = GroqModel(args.groq_model)
    controller = PromptController(args)
    library = RuleLibrary(args)

    initial_train_set = dataset.get_split()
    logger.info(f"Loaded {len(initial_train_set)} unique examples for {args.task_type} induction.")

    if len(initial_train_set) == 0:
        logger.warning("Input data file is empty. Exiting.")
        return

    # Create the processing list based on num_iterations (epochs)
    num_epoch = args.num_iterations // len(initial_train_set)
    remainder = args.num_iterations % len(initial_train_set)
    train_set = initial_train_set * num_epoch + random.sample(initial_train_set, remainder)
    logger.info(f"Processing a total of {len(train_set)} samples ({num_epoch} epochs and {remainder} random samples).")

    for i, sample in enumerate(tqdm(train_set, desc=f"Inducing {args.task_type.capitalize()} Rules")):
        controller(model, sample, library, logger)
        
        num_processed = i + 1
        if num_processed % args.save_interval == 0 or num_processed == len(train_set):
            output_path = os.path.join(args.output_folder, f"{os.path.splitext(args.output_rule_library_file)[0]}_{num_processed}.json")
            num_saved_rules = library.save(output_path)
            logger.info(f"\nSaved {num_saved_rules} rules at iteration {num_processed} to {output_path}")

    final_output_path = os.path.join(args.output_folder, args.output_rule_library_file)
    num_saved_rules = library.save(final_output_path)
    logger.info(f"\nRule induction process completed. Final library with {num_saved_rules} rules saved to {final_output_path}")

if __name__ == "__main__":
    main()
