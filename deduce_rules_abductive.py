import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import time
import re
import torch
import xml.etree.ElementTree as ET # Added for XML parsing

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

# For Deductive (A/B/C/D)
hyp_alpa_deductive_en = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
hyp_alpa_deductive_ar = {0: 'أ', 1: 'ب', 2: 'ج', 3: 'د'}

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

DEDUCTION_PROMPT_DEDUCTIVE_EN = """
You are an expert in deductive reasoning using a rule library.
Below is a library of reasoning rules:
Rule Library:
---
{rules_text}
---

Consider the following question and options:
Question: {question_text}
Option A: {option_A_text}
Option B: {option_B_text}
Option C: {option_C_text}
Option D: {option_D_text}

Based *strictly* on the provided Rule Library, which option is the correct answer?
Conclude with your final answer in the format: 'Final Answer: X' where X is the single letter of the correct option (A, B, C, or D). Do not add any other text after this line.
"""

DEDUCTION_PROMPT_DEDUCTIVE_AR = """
أنت خبير في الاستدلال الاستنباطي باستخدام مكتبة قواعد.
فيما يلي مكتبة من قواعد الاستدلال:
مكتبة القواعد:
---
{rules_text}
---

ضع في اعتبارك السؤال والخيارات التالية:
السؤال: {question_text}
الخيار أ: {option_A_text}
الخيار ب: {option_B_text}
الخيار ج: {option_C_text}
الخيار د: {option_D_text}

بناءً *فقط* على مكتبة القواعد المقدمة، أي خيار هو الإجابة الصحيحة؟
اختتم بإجابتك النهائية بالتنسيق: 'الإجابة النهائية: X' حيث X هو الحرف الوحيد للخيار الصحيح (أ، ب، ج، أو د). لا تضف أي نص آخر بعد هذا السطر.
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

COT_DEDUCTION_PROMPT_DEDUCTIVE_EN = """
You are an expert in deductive reasoning using a rule library.
Below is a library of reasoning rules:
Rule Library:
---
{rules_text}
---

Consider the following question and options:
Question: {question_text}
Option A: {option_A_text}
Option B: {option_B_text}
Option C: {option_C_text}
Option D: {option_D_text}

Based *strictly* on the provided Rule Library, reason step-by-step to determine which option is the correct answer.
1. Evaluate Option A against the rules.
2. Evaluate Option B against the rules.
3. Evaluate Option C against the rules.
4. Evaluate Option D against the rules.
5. Conclude which option is best supported by the rules.

Conclude with your final answer in the format: 'Final Answer: X' where X is the single letter of the correct option (A, B, C, or D). Do not add any other text after this line.
"""

COT_DEDUCTION_PROMPT_DEDUCTIVE_AR = """
أنت خبير في الاستدلال الاستنباطي باستخدام مكتبة قواعد.
فيما يلي مكتبة من قواعد الاستدلال:
مكتبة القواعد:
---
{rules_text}
---

ضع في اعتبارك السؤال والخيارات التالية:
السؤال: {question_text}
الخيار أ: {option_A_text}
الخيار ب: {option_B_text}
الخيار ج: {option_C_text}
الخيار د: {option_D_text}

بناءً *فقط* على مكتبة القواعد المقدمة، استدل خطوة بخطوة لتحديد أي خيار هو الإجابة الصحيحة.
١. قيّم الخيار أ في ضوء القواعد.
٢. قيّم الخيار ب في ضوء القواعد.
٣. قيّم الخيار ج في ضوء القواعد.
٤. قيّم الخيار د في ضوء القواعد.
٥. استنتج أي خيار هو الأفضل دعمًا بالقواعد.

اختتم بإجابتك النهائية بالتنسيق: 'الإجابة النهائية: X' حيث X هو الحرف الوحيد للخيار الصحيح (أ، ب، ج، أو د). لا تضف أي نص آخر بعد هذا السطر.
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

TOT_DEDUCTION_PROMPT_DEDUCTIVE_EN = """
You are an expert in deductive reasoning using a rule library.
Below is a library of reasoning rules:
Rule Library:
---
{rules_text}
---

Consider the following question and options:
Question: {question_text}
Option A: {option_A_text}
Option B: {option_B_text}
Option C: {option_C_text}
Option D: {option_D_text}

Based *strictly* on the provided Rule Library, use a Tree of Thought approach to determine the correct option:
1.  Initial Rule Brainstorm & Option Connection:
    *   For Option A: Identify rules that support or contradict it in relation to the question.
    *   For Option B: Identify rules that support or contradict it in relation to the question.
    *   For Option C: Identify rules that support or contradict it in relation to the question.
    *   For Option D: Identify rules that support or contradict it in relation to the question.
2.  Evaluate Plausibility Paths for each option based on rule connections.
3.  Compare Paths and Decide: Based *only* on the rule-based evaluation, which option (A, B, C, or D) is most strongly supported by the Rule Library as the answer to the question?

Conclude with your final answer in the format: 'Final Answer: X' where X is the single letter of the correct option (A, B, C, or D). Do not add any other text after this line.
"""

TOT_DEDUCTION_PROMPT_DEDUCTIVE_AR = """
أنت خبير في الاستدلال الاستنباطي باستخدام مكتبة قواعد.
فيما يلي مكتبة من قواعد الاستدلال:
مكتبة القواعد:
---
{rules_text}
---

ضع في اعتبارك السؤال والخيارات التالية:
السؤال: {question_text}
الخيار أ: {option_A_text}
الخيار ب: {option_B_text}
الخيار ج: {option_C_text}
الخيار د: {option_D_text}

بناءً *فقط* على مكتبة القواعد المقدمة، استخدم نهج شجرة الأفكار لتحديد الخيار الصحيح:
١. عصف ذهني أولي للقواعد وربط الخيارات:
    *   بالنسبة للخيار أ: حدد القواعد التي تدعمه أو تتعارض معه فيما يتعلق بالسؤال.
    *   بالنسبة للخيار ب: حدد القواعد التي تدعمه أو تتعارض معه فيما يتعلق بالسؤال.
    *   بالنسبة للخيار ج: حدد القواعد التي تدعمه أو تتعارض معه فيما يتعلق بالسؤال.
    *   بالنسبة للخيار د: حدد القواعد التي تدعمه أو تتعارض معه فيما يتعلق بالسؤال.
٢. تقييم مسارات القبول لكل خيار بناءً على ارتباطات القواعد.
٣. قارن المسارات وقرر: بناءً *فقط* على التقييم المستند إلى القواعد، أي خيار (أ، ب، ج، أو د) هو الأكثر دعمًا قويًا بواسطة مكتبة القواعد كإجابة على السؤال؟

اختتم بإجابتك النهائية بالتنسيق: 'الإجابة النهائية: X' حيث X هو الحرف الوحيد للخيار الصحيح (أ، ب، ج، أو د). لا تضف أي نص آخر بعد هذا السطر.
"""

# --- Helper Functions ---

def load_rule_library(file_path: str) -> list[str]:
    """Loads rules from a JSON file, expecting a list of dicts with a 'rule' key."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(item, dict) and 'rule' in item for item in data):
            return [item['rule'] for item in data if item['rule'].strip()]
        elif isinstance(data, list) and all(isinstance(item, str) for item in data): # Handle list of strings
             return [item for item in data if item.strip()]
        else:
            print(f"Warning: Rule library JSON structure not as expected in {file_path}. Expected list of dicts with 'rule' key or list of strings.")
            return []
    except FileNotFoundError:
        print(f"Error: Abductive rule JSON file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not parse abductive rule JSON file: {file_path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading abductive JSON rules from {file_path}: {e}")
        return []

def load_deductive_xml_rules(file_path: str) -> list[str]:
    """Loads deductive rules from an XML file, expecting <rules><rule>text</rule></rules> structure."""
    rules = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        # Assuming rules are direct children <rule> tags under the root <rules> tag
        # Or <rule> tags anywhere if the structure is flatter. Using .//rule to be more flexible.
        for rule_element in root.findall('.//rule'): 
            if rule_element.text:
                rules.append(rule_element.text.strip())
        if not rules:
            print(f"Warning: No rules found or extracted from deductive XML file: {file_path}")
    except FileNotFoundError:
        print(f"Error: Deductive rule XML file not found: {file_path}")
        return []
    except ET.ParseError:
        print(f"Error: Could not parse deductive rule XML file: {file_path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading deductive XML rules from {file_path}: {e}")
        return []
    return rules

def get_abductive_data_from_row(row: pd.Series, lang_alpa: str) -> dict | None:
    """
    Extracts observations, hypotheses, and gold answer letter from a DataFrame row
    for abductive reasoning (2 hypotheses).
    """
    obs1 = str(row.get('observation_1', '')).strip()
    obs2 = str(row.get('observation_2', '')).strip()
    hyp1_text = str(row.get('hypothesis_1', '')).strip()
    hyp2_text = str(row.get('hypothesis_2', '')).strip()
    label_str = str(row.get('label', '')).strip()

    if not all([obs1, obs2, hyp1_text, hyp2_text, label_str]):
        print(f"Warning: Missing one or more required fields for abductive data in row: {row.to_dict()}")
        return None

    current_prompt_alpa_map = hyp_alpa_ar if lang_alpa == 'ar' else hyp_alpa_en
    gold_answer_letter = None

    # Assuming CSV label '1' means hypothesis_1 (A/أ) is correct, and '2' means hypothesis_2 (B/ب) is correct.
    if label_str == '1':
        gold_answer_letter = current_prompt_alpa_map[0]
    elif label_str == '2':
        gold_answer_letter = current_prompt_alpa_map[1]
    elif label_str == '0': # If '0' is used for the first hypothesis
        gold_answer_letter = current_prompt_alpa_map[0]
    else: # Check for direct letter labels 'A'/'B' or 'أ'/'ب'
        temp_label_upper = label_str.upper() # Ensure consistent casing for A/B
        if temp_label_upper == current_prompt_alpa_map[0].upper():
            gold_answer_letter = current_prompt_alpa_map[0]
        elif temp_label_upper == current_prompt_alpa_map[1].upper():
            gold_answer_letter = current_prompt_alpa_map[1]
        else:
            print(f"Warning: Unrecognized label '{label_str}' for abductive data in row: {row.to_dict()}")

    if gold_answer_letter is None:
        return None

    return {
        "obs1": obs1, "obs2": obs2,
        "hyp1_text": hyp1_text, "hyp2_text": hyp2_text,
        "gold_answer_letter": gold_answer_letter,
        "type": "abductive" # Add type identifier
    }

def get_deductive_data_from_row(row: pd.Series, lang_alpa: str) -> dict | None:
    """
    Extracts question, options (A,B,C,D), and gold answer letter from a DataFrame row
    for deductive reasoning.
    """
    question = str(row.get('question', '')).strip()
    opt_a = str(row.get('option_a', '')).strip()
    opt_b = str(row.get('option_b', '')).strip()
    opt_c = str(row.get('option_c', '')).strip()
    opt_d = str(row.get('option_d', '')).strip()
    answer_str = str(row.get('answer', '')).strip() # This is the gold answer letter (A,B,C,D or أ,ب,ج,د)

    if not all([question, opt_a, opt_b, opt_c, opt_d, answer_str]):
        print(f"Warning: Missing one or more required fields for deductive data in row: {row.to_dict()}")
        return None

    current_deductive_alpa_map = hyp_alpa_deductive_ar if lang_alpa == 'ar' else hyp_alpa_deductive_en
    gold_answer_letter = None

    # The 'answer' column directly contains the letter of the correct option.
    # Normalize to upper for English to match map keys if necessary, Arabic keys are as-is.
    normalized_answer_str = answer_str.upper() if lang_alpa == 'en' else answer_str

    # Check if the answer_str is one of the valid letters in the current deductive map
    if normalized_answer_str in current_deductive_alpa_map.values():
        gold_answer_letter = normalized_answer_str
    else:
        # Attempt to map from 0-indexed or 1-indexed numeric answers if direct letter match fails
        # This is a fallback, ideally the 'answer' column is already A/B/C/D or أ/ب/ج/د
        try:
            answer_idx = -1
            if answer_str == '0' or answer_str.upper() == 'A' or answer_str == 'أ': answer_idx = 0
            elif answer_str == '1' or answer_str.upper() == 'B' or answer_str == 'ب': answer_idx = 1
            elif answer_str == '2' or answer_str.upper() == 'C' or answer_str == 'ج': answer_idx = 2
            elif answer_str == '3' or answer_str.upper() == 'D' or answer_str == 'د': answer_idx = 3
            
            if answer_idx != -1 and answer_idx in current_deductive_alpa_map:
                 gold_answer_letter = current_deductive_alpa_map[answer_idx]
            else:
                print(f"Warning: Unrecognized answer label '{answer_str}' for deductive data in row: {row.to_dict()}")
        except ValueError:
            print(f"Warning: Could not parse answer label '{answer_str}' for deductive data in row: {row.to_dict()}")

    if gold_answer_letter is None:
        return None

    return {
        "question_text": question,
        "option_A_text": opt_a,
        "option_B_text": opt_b,
        "option_C_text": opt_c,
        "option_D_text": opt_d,
        "gold_answer_letter": gold_answer_letter,
        "type": "deductive" # Add type identifier
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
        # Ensure DEDUCTION_PROMPT_ABDUCTIVE_AR and DEDUCTION_PROMPT_ABDUCTIVE_EN are defined
        if lang_prompt == 'ar':
            template = globals().get('DEDUCTION_PROMPT_ABDUCTIVE_AR', COT_DEDUCTION_PROMPT_ABDUCTIVE_AR) # Fallback to CoT if zero-shot not defined
        else:
            template = globals().get('DEDUCTION_PROMPT_ABDUCTIVE_EN', COT_DEDUCTION_PROMPT_ABDUCTIVE_EN) # Fallback to CoT if zero-shot not defined
    
    return template.format(
        rules_text=rules_str,
        observation_1=obs1, observation_2=obs2,
        hypothesis_A_text=hyp_A_text, hypothesis_B_text=hyp_B_text
    )

def format_deductive_prompt(rules_text_list: list[str], question_text: str, 
                            option_A_text: str, option_B_text: str, option_C_text: str, option_D_text: str, 
                            lang_prompt: str, use_cot: bool = False, use_tot: bool = False) -> str:
    rules_str = "\n".join(f"- {rule.strip()}" for rule in rules_text_list if rule.strip())

    if use_tot:
        template = TOT_DEDUCTION_PROMPT_DEDUCTIVE_AR if lang_prompt == 'ar' else TOT_DEDUCTION_PROMPT_DEDUCTIVE_EN
    elif use_cot:
        template = COT_DEDUCTION_PROMPT_DEDUCTIVE_AR if lang_prompt == 'ar' else COT_DEDUCTION_PROMPT_DEDUCTIVE_EN
    else: # Zero-shot
        template = DEDUCTION_PROMPT_DEDUCTIVE_AR if lang_prompt == 'ar' else DEDUCTION_PROMPT_DEDUCTIVE_EN
        
    return template.format(
        rules_text=rules_str,
        question_text=question_text,
        option_A_text=option_A_text,
        option_B_text=option_B_text,
        option_C_text=option_C_text,
        option_D_text=option_D_text
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

def parse_deduction_response(llm_output: str | None, lang_alpa: str, 
                             is_cot_or_tot: bool = False, 
                             reasoning_type: str = "abductive") -> str | None: # Added reasoning_type
    if not llm_output: # Handles None or empty string
        return None
    
    cleaned_output_for_final_answer_check = llm_output.strip() # Keep original case for regex
    
    current_alpa_map = {}
    if reasoning_type == "deductive":
        current_alpa_map = hyp_alpa_deductive_ar if lang_alpa == 'ar' else hyp_alpa_deductive_en
    else: # abductive (default)
        current_alpa_map = hyp_alpa_ar if lang_alpa == 'ar' else hyp_alpa_en
        
    valid_choices_set = set(current_alpa_map.values()) # e.g. {'A', 'B'} or {'أ', 'ب', 'ج', 'د'}
    
    # For CoT/ToT, primarily look for "Final Answer: X"
    if is_cot_or_tot:
        # Dynamically create character class for regex from valid_choices_set
        char_class_chars_list = sorted(list(valid_choices_set))
        char_class_regex_part = "".join(char_class_chars_list)
        
        if lang_alpa == 'ar':
            # For Arabic, ensure the marker is exact. The choices are already specific.
            pattern_str_final_answer = rf"الإجابة النهائية:\s*\[?\[?([{char_class_regex_part}])\]?\]?\.?$"
        else: # English
            # For English, "Final Answer:" can be case-insensitive. The choice letter will be normalized.
            pattern_str_final_answer = rf"Final Answer:\s*\[?\[?([{char_class_regex_part}])\]?\]?\.?$"

        final_answer_letter = None
        # Apply re.IGNORECASE for English marker, not for Arabic.
        # The character class itself ([ABCD] or [أبجد]) will match case-sensitively unless the IGNORECASE flag makes it not so.
        # For English, we want to match "Final Answer: a" and normalize 'a' to 'A'.
        match = re.search(pattern_str_final_answer, cleaned_output_for_final_answer_check, re.IGNORECASE if lang_alpa == 'en' else 0)
        if match:
            letter = match.group(1)
            if lang_alpa == 'en': # Normalize case for English choice
                letter = letter.upper()
            
            if letter in valid_choices_set: # Check if the (normalized) extracted letter is a valid choice
                final_answer_letter = letter
        
        if final_answer_letter:
            return final_answer_letter
        # If CoT/ToT was used but "Final Answer: X" not found, fall through to simpler parsing.

    # Standard parsing (Zero-shot or fallback if CoT/ToT pattern fails)
    text_to_parse = llm_output.strip()
    if lang_alpa == 'en': # Uppercase for English to match 'A', 'B', 'C', 'D'
        text_to_parse = text_to_parse.upper()

    if not text_to_parse: # If stripping results in an empty string
        return None

    # Fallback 1: Exact match (single character that is a valid choice after cleaning)
    if text_to_parse in valid_choices_set:
        return text_to_parse

    # Fallback 2: Single unique valid character in the entire cleaned output
    # (handles "A.", "(A)", "Answer: A", etc.)
    present_valid_choices = set()
    for char_token in text_to_parse: 
        # char_token is already uppercased if lang_alpa == 'en'
        if char_token in valid_choices_set: 
            present_valid_choices.add(char_token)
    
    if len(present_valid_choices) == 1:
        return list(present_valid_choices)[0]
    
    # Fallback 3: Check if the last "word" (token) of the output is a valid choice.
    # This is for cases like "The answer is clearly A" or "أعتقد أن الإجابة هي ب"
    # Split by whitespace and common punctuation that might separate words.
    # For Arabic, \w includes Arabic letters.
    words = re.split(r'\s+|[^\w\s]', text_to_parse) 
    words = [word for word in words if word] # Filter out empty strings

    if words:
        last_word = words[-1]
        # last_word is already uppercased if lang_alpa == 'en' because text_to_parse was.
        if last_word in valid_choices_set:
            return last_word

    return None



# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Deduce hypothesis plausibility or solve deductive problems using a rule library and LLMs.")
    parser.add_argument("--eval_data_file", type=str, required=True, help="Path to evaluation CSV.")
    parser.add_argument("--rule_library_file", type=str, required=True, help="Path to JSON rule library (abductive/general).")
    parser.add_argument("--deductive_rule_file", type=str, default=None, help="Path to XML rule library for deductive reasoning (optional).")
    parser.add_argument("--output_results_file", type=str, default="reasoning_results.csv", help="Filename for results CSV.")
    parser.add_argument("--output_folder", type=str, default="results_reasoning", help="Folder for results CSV.")
    parser.add_argument("--lang_prompt", type=str, default="en", choices=["en", "ar"], help="Prompt language.")
    parser.add_argument("--lang_alpa", type=str, default="en", choices=["en", "ar"], help="Hypothesis/Option label language (A/B or A/B/C/D).")
    parser.add_argument("--max_examples", type=int, default=None, help="Max evaluation examples (for testing).")
    parser.add_argument("--reasoning_type", type=str, default="abductive", choices=["abductive", "deductive"], help="Type of reasoning to perform (abductive or deductive).")

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
    if args.reasoning_type == "deductive":
        model_identifier_suffix += "_deductive"
    else:
        model_identifier_suffix += "_abductive"


    is_cot_or_tot_enabled = args.use_tot or args.use_cot # ToT takes precedence if both are somehow true
    llm_config["is_cot_or_tot"] = is_cot_or_tot_enabled

    # Define max tokens based on prompting strategy and LLM type
    MAX_TOKENS_COT_TOT_HF = 4096 
    MAX_TOKENS_ZERO_SHOT_HF = 4096  # Increased slightly for safety
    MAX_TOKENS_COT_TOT_GROQ = 4096
    MAX_TOKENS_ZERO_SHOT_GROQ = 4096 # Increased for Groq zero-shot, as it might add small phrases

    if args.use_hf_model:
        llm_config["type"] = "hf"
        llm_config["model_path"] = args.hf_model_path
        llm_config["lora_weights"] = args.lora_weights
        llm_config["load_8bit"] = args.load_8bit
        llm_config["max_length"] = args.hf_max_length # Tokenizer max length
        model_name_for_file = os.path.basename(args.hf_model_path) if args.hf_model_path else "hf_default"
        # Sanitize model_name_for_file further for any other problematic characters if necessary
        model_name_for_file = model_name_for_file.replace("/", "_").replace(":", "_")
        model_identifier_suffix += f"_hf_{model_name_for_file}"
        if is_cot_or_tot_enabled:
            llm_config["max_tokens_to_generate"] = MAX_TOKENS_COT_TOT_HF
        else:
            llm_config["max_tokens_to_generate"] = MAX_TOKENS_ZERO_SHOT_HF
        
        if not HF_AVAILABLE:
            print("Error: --use_hf_model selected, but Hugging Face Transformers library is not available. Please install it.")
            return
        if not args.hf_model_path:
            print("Error: --use_hf_model selected, but --hf_model_path not provided.")
            return
        
        print(f"Loading Hugging Face model: {args.hf_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
        model_load_params = {"trust_remote_code": True}
        if args.load_8bit:
            model_load_params["load_in_8bit"] = True
        
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_path, **model_load_params)

        if args.lora_weights:
            print(f"Loading LoRA weights from {args.lora_weights}...")
            model = PeftModel.from_pretrained(model, args.lora_weights)
        
        model.to(device)
        model.eval()
        llm_config["model"] = model
        llm_config["tokenizer"] = tokenizer
        print("Hugging Face model loaded.")

    else: # Groq LLM
        llm_config["type"] = "groq"
        sanitized_groq_model_name = args.groq_model.replace("/", "_").replace(":", "_")
        llm_config["model_name"] = args.groq_model # Keep original for API call
        model_identifier_suffix += f"_groq_{sanitized_groq_model_name}" # Use sanitized for filename
        if is_cot_or_tot_enabled:
            llm_config["max_tokens_to_generate"] = MAX_TOKENS_COT_TOT_GROQ
        else:
            llm_config["max_tokens_to_generate"] = MAX_TOKENS_ZERO_SHOT_GROQ

        if not GROQ_AVAILABLE:
            print("Error: Groq LLM selected, but Groq library is not available. Please install it.")
            return
        try:
            llm_config['client'] = Groq()
            print(f"Using Groq LLM with model: {args.groq_model}")
        except Exception as e:
            print(f"Error initializing Groq client: {e}")
            return

    # Determine prompting strategy for filename
    prompt_strategy_suffix = ""
    if args.use_tot:
        prompt_strategy_suffix = "_tot"
    elif args.use_cot:
        prompt_strategy_suffix = "_cot"
    else:
        prompt_strategy_suffix = "_zeroshot" # Explicitly state zero-shot

    # Construct output filename
    base_output_filename = f"deduction_results_{args.lang_prompt}{prompt_strategy_suffix}{model_identifier_suffix}.csv"
    output_file_path = os.path.join(args.output_folder, base_output_filename)

    abductive_rule_library_texts = load_rule_library(args.rule_library_file)
    if not abductive_rule_library_texts:
        print(f"Warning: Main rule library from {args.rule_library_file} is empty or failed to load.")
    
    deductive_xml_rules = []
    if args.deductive_rule_file:
        deductive_xml_rules = load_deductive_xml_rules(args.deductive_rule_file)
        if not deductive_xml_rules:
            print(f"Warning: Deductive XML rule library from {args.deductive_rule_file} is empty or failed to load.")

    all_rules_for_prompt = abductive_rule_library_texts + deductive_xml_rules
    
    if not all_rules_for_prompt:
        print("Error: No rules were loaded from any source. Exiting, as rules are required for reasoning.")
        return

    print(f"Loaded {len(abductive_rule_library_texts)} rules from JSON library.")
    if args.deductive_rule_file:
        print(f"Loaded {len(deductive_xml_rules)} rules from XML library.")
    print(f"Total {len(all_rules_for_prompt)} rules will be used for prompting.")

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

    print(f"Starting {args.reasoning_type} reasoning process...")
    for index, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"{args.reasoning_type.capitalize()} Reasoning"):
        prompt = None
        data_item = None
        # expected_options_str = "" # For logging/error messages (can be re-added if needed)

        if args.reasoning_type == "abductive":
            data_item = get_abductive_data_from_row(row, args.lang_alpa)
            if data_item:
                prompt = format_deduction_abductive_prompt(
                    all_rules_for_prompt, data_item["obs1"], data_item["obs2"],
                    data_item["hyp1_text"], data_item["hyp2_text"], args.lang_prompt,
                    use_cot=args.use_cot, use_tot=args.use_tot
                )
        elif args.reasoning_type == "deductive":
            data_item = get_deductive_data_from_row(row, args.lang_alpa)
            if data_item:
                prompt = format_deductive_prompt(
                    all_rules_for_prompt,
                    data_item["question_text"],
                    data_item["option_A_text"], data_item["option_B_text"],
                    data_item["option_C_text"], data_item["option_D_text"],
                    args.lang_prompt,
                    use_cot=args.use_cot, use_tot=args.use_tot
                )
        else: # Should not happen due to choices in argparse
            print(f"Critical Error: Unknown reasoning type '{args.reasoning_type}'. Exiting.")
            # Consider `parser.error()` or `sys.exit(1)` if this path is reachable
            return

        if not data_item or not prompt:
            data_errors += 1
            error_log_item = {k: row.get(k, '') for k in row.index}
            error_log_item.update({
                "gold_label_original": row.get('label', row.get('answer', '')), # Handle both abductive/deductive original label
                "gold_label_letter": "DATA_ERROR",
                "predicted_label_letter": "DATA_ERROR", "is_correct": False,
                "llm_raw_response": "Skipped due to data error or prompt formatting issue.",
                "reasoning_type": args.reasoning_type
            })
            results_data.append(error_log_item)
            continue

        llm_response = call_llm_for_deduction(prompt, llm_config)
        
        predicted_letter_for_csv = "ERROR" # Default
        is_correct = False

        if llm_response is None:
            llm_call_errors +=1
            predicted_letter_for_csv = "LLM_CALL_ERROR"
        else:
            predicted_letter = parse_deduction_response(
                llm_response, 
                args.lang_alpa, 
                is_cot_or_tot=is_cot_or_tot_enabled,
                reasoning_type=args.reasoning_type # Pass the reasoning type
            )
            if predicted_letter is None:
                parse_errors += 1
                predicted_letter_for_csv = "PARSE_ERROR"
            elif data_item.get("gold_answer_letter"): # Check if gold_answer_letter exists
                is_correct = (predicted_letter == data_item["gold_answer_letter"])
                if is_correct:
                    correct_predictions += 1
                predicted_letter_for_csv = predicted_letter
            else: # Parsed successfully but no gold answer (should not happen if data_item is valid)
                predicted_letter_for_csv = predicted_letter 
                print(f"Warning: Parsed LLM response to '{predicted_letter}' but no gold_answer_letter in data_item for row {index}.")
        
        total_processed +=1

        current_result = {
            "gold_label_original": row.get('label', row.get('answer', '')), # original label from CSV
            "gold_label_letter": data_item.get("gold_answer_letter", "N/A_IF_DATA_ERROR"), # Use .get for safety
            "predicted_label_letter": predicted_letter_for_csv,
            "is_correct": is_correct,
            "llm_raw_response": llm_response if llm_response else "LLM_ERROR_NO_RESPONSE",
            "reasoning_type": args.reasoning_type
        }
        if args.reasoning_type == "abductive":
            current_result.update({
                "obs1": data_item.get("obs1",""), "obs2": data_item.get("obs2",""), # Use .get for safety
                "hyp1": data_item.get("hyp1_text",""), "hyp2": data_item.get("hyp2_text",""),
            })
        else: # deductive
            current_result.update({
                "question": data_item.get("question_text",""),
                "option_a": data_item.get("option_A_text",""),
                "option_b": data_item.get("option_B_text",""),
                "option_c": data_item.get("option_C_text",""),
                "option_d": data_item.get("option_D_text",""),
            })
        results_data.append(current_result)

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f"\nReasoning results saved to: {output_file_path}")

    if data_errors > 0:
        print(f"Note: {data_errors} rows skipped due to input data errors.")
    if llm_call_errors > 0:
        print(f"Note: {llm_call_errors} LLM calls failed or returned no response.")
    if parse_errors > 0: 
        print(f"Note: {parse_errors} LLM responses could not be parsed into a valid choice (excluding LLM call failures).")
        
    if total_processed > 0:
        accuracy = (correct_predictions / total_processed) * 100
        print(f"Reasoning Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_processed} correctly predicted examples)")
    else:
        print("No examples were successfully processed to calculate accuracy.")


if __name__ == "__main__":
    main()