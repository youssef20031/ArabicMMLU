import pandas as pd
import os # Import os for path operations if needed later

# --- Alphabet Mappings ---
alpa_ar = {
    0: 'أ',
    1: 'ب',
    2: 'ج',
    3: 'د',
    4: 'ه'
}
alpa_en = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E'
}

# --- Metadata Mappings (Optional, keep if used elsewhere, but not directly in new format_question) ---
level_en = {
        'Primary': 'primary school',
        'Middle': 'middle school',
        'High': 'high school',
        'Univ': 'university',
        'Prof': 'professional'
}
level_ar = {
    'Primary': 'للمرحلة الابتدائية',
    'Middle': 'للمرحلة المتوسطة',
    'High': 'للمرحلة الثانوية',
    'Univ': 'للجامعات',
    'Prof': 'للمحترفين'
}
country_ar = {
    'UAE': 'في دولة الإمارات العربية المتحدة',
    'Egypt': 'في مصر',
    'Lebanon': 'في لبنان',
    'Jordan': 'في الأردن',
    'Kuwait': 'في الكويت',
    'KSA': 'في المملكة العربية السعودية',
    'Palestine': 'في فلسطين',
    'Morocco': 'في المغرب',
}
subject_ar = {
    'Islamic Studies': 'في الدراسات الإسلامية',
    'Driving Test': 'اختبار القيادة',
    'Natural Science': 'في العلوم الطبيعية',
    'History': 'في التاريخ',
    'General Knowledge': 'في المعلومات العامة',
    'Law': 'في القانون',
    'Physics': 'في الفيزياء',
    'Social Science': 'في العلوم الاجتماعية',
    'Management': 'في الإدارة',
    'Arabic Language': 'في اللغة العربية',
    'Political Science': ' في العلوم السياسية',
    'Philosophy': 'في الفلسفة',
    'Accounting': 'في المحاسبة',
    'Computer Science': 'في علوم الحاسوب',
    'Geography': 'في الجغرافيا',
    'Math': 'في الرياضيات',
    'Biology': 'في علم الأحياء',
    'Economics': 'في الاقتصاد',
    'Arabic Language (General)': 'في اللغة العربية (عام)',
    'Arabic Language (Grammar)': 'في اللغة العربية (قواعد النحو)',
    'Civics': 'في التربية المدنية',
}
# --- End Metadata Mappings ---


# --- Define Prompt Templates ---

# Zero-Shot Prompt (Simplified - focuses on question and options)
ZERO_SHOT_PROMPT_EN = (
    "The following is a multiple choice question about {subject}.\n\n"
    "{question}\n"
    "{options}\n\n"
    "Please answer with only the single letter corresponding to the correct option." # Instruction added
)
ZERO_SHOT_PROMPT_AR = (
    "السؤال التالي هو سؤال متعدد الخيارات حول {subject}.\n\n"
    "{question}\n"
    "{options}\n\n"
    "الرجاء الإجابة فقط بحرف الخيار الصحيح الموافق للإجابة الصحيحة." # Instruction added
)

# Chain-of-Thought Prompt
COT_PROMPT_EN = (
    "You are an expert in {subject}.\n"
    "Analyze the given multiple-choice question and provide the correct answer using this approach:\n\n"
    "1. Carefully read the question and options.\n"
    "2. Identify core {subject} concepts and required knowledge.\n"
    "3. Analyze each option for relevance, accuracy, and consistency.\n"
    "4. Consider {subject}-specific context and factors.\n"
    "5. Use elimination and comparative analysis.\n"
    "6. Select the most accurate answer.\n"
    "7. Maintain objectivity, consider {subject}-specific sensitivities, and base your decision on verifiable facts and sound logical reasoning within {subject}.\n\n"
    "Question:\n{question}\n"
    "{options}\n\n"
    "Provide your step-by-step reasoning and conclude with the final answer in the format 'Final Answer: The final answer is [[X]]' where X is the correct letter choice."
)
COT_PROMPT_AR = (
    "أنت خبير في {subject}.\n"
    "حلل السؤال متعدد الخيارات التالي وقدم الإجابة الصحيحة باستخدام النهج التالي:\n\n"
    "١. اقرأ السؤال والخيارات بعناية.\n"
    "٢. حدد مفاهيم {subject} الأساسية والمعرفة المطلوبة.\n"
    "٣. حلل كل خيار من حيث الصلة والدقة والاتساق.\n"
    "٤. ضع في اعتبارك السياق والعوامل الخاصة بـ {subject}.\n"
    "٥. استخدم الاستبعاد والتحليل المقارن.\n"
    "٦. اختر الإجابة الأكثر دقة.\n"
    "٧. حافظ على الموضوعية، ضع في اعتبارك الحساسيات الخاصة بـ {subject}، واستند في قرارك إلى الحقائق التي يمكن التحقق منها والتفكير المنطقي السليم ضمن {subject}.\n\n"
    "السؤال:\n{question}\n"
    "{options}\n\n"
    "قدم تفكيرك خطوة بخطوة واختتم بالإجابة النهائية بالتنسيق 'الإجابة النهائية: الإجابة النهائية هي [[X]]' حيث X هو حرف الخيار الصحيح."
)

# Tree-of-Thought Prompt (Experimental Simulation)
TOT_PROMPT_EN = (
    "You are an expert in {subject}.\n"
    "Solve the following multiple-choice question using a Tree of Thought approach:\n\n"
    "1. Deconstruct the Question: Identify the core problem and constraints.\n"
    "2. Generate Potential Thoughts/Paths: Brainstorm multiple initial approaches or interpretations to solve the problem.\n"
    "3. Evaluate Thoughts: For each path, analyze its validity, potential pitfalls, and likelihood of leading to the correct answer based on {subject} knowledge.\n"
    "4. Explore Promising Paths: Elaborate on the most promising paths, generating intermediate reasoning steps.\n"
    "5. Self-Reflect and Prune: Review the explored paths. Discard paths that are clearly incorrect or less likely. Refine remaining paths.\n"
    "6. Synthesize and Decide: Based on the evaluation and exploration, determine the most logical and well-supported answer among the options.\n\n"
    "Question:\n{question}\n"
    "{options}\n\n"
    "Provide your detailed reasoning simulating the Tree of Thought process described above. Conclude with the final answer in the format 'Final Answer: The final answer is [[X]]' where X is the correct letter choice."
)
TOT_PROMPT_AR = (
    "أنت خبير في {subject}.\n"
    "حل السؤال متعدد الخيارات التالي باستخدام نهج شجرة الأفكار:\n\n"
    "١. فكك السؤال: حدد المشكلة الأساسية والقيود.\n"
    "٢. ولّد أفكارًا/مسارات محتملة: اطرح أفكارًا متعددة للمقاربات الأولية أو التفسيرات لحل المشكلة.\n"
    "٣. قيّم الأفكار: لكل مسار، حلل صلاحيته، والمزالق المحتملة، واحتمالية أن يؤدي إلى الإجابة الصحيحة بناءً على معرفة {subject}.\n"
    "٤. استكشف المسارات الواعدة: توسع في المسارات الواعدة، مولداً خطوات تفكير وسيطة.\n"
    "٥. تأمل ذاتيًا وقلم: راجع المسارات المستكشفة. تجاهل المسارات غير الصحيحة بوضوح أو الأقل احتمالاً. نقح المسارات المتبقية.\n"
    "٦. ركب وقرر: بناءً على التقييم والاستكشاف، حدد الإجابة الأكثر منطقية ومدعومة جيدًا بين الخيارات.\n\n"
    "السؤال:\n{question}\n"
    "{options}\n\n"
    "قدم تفكيرك المفصل محاكيًا عملية شجرة الأفكار الموضحة أعلاه. اختتم بالإجابة النهائية بالتنسيق 'الإجابة النهائية: الإجابة النهائية هي [[X]]' حيث X هو حرف الخيار الصحيح."
)


# --- New Abductive Reasoning Prompts ---
# These prompts expect a {question_text} placeholder which will be filled with observations and hypotheses.
# And a {subject} placeholder, which can be generic like "reasoning".

ZERO_SHOT_PROMPT_ABDUCTIVE_EN = (
    "{question_text}\n\n"
    "Instructions: Answer with only the single letter (A or B) corresponding to the most plausible hypothesis."
)
ZERO_SHOT_PROMPT_ABDUCTIVE_AR = (
    "{question_text}\n\n"
    "التعليمات: أجب فقط بالحرف (أ أو ب) الموافق للفرضية الأكثر قبولاً."
)

COT_PROMPT_ABDUCTIVE_EN = (
    "You are an expert in {subject}.\n"
    "Analyze the following observations and hypotheses to determine the most plausible hypothesis.\n\n"
    "{question_text}\n\n"
    "Provide your step-by-step reasoning and conclude with the final answer in the format 'Final Answer: The final answer is [[X]]' where X is the correct letter choice (A or B)."
)
COT_PROMPT_ABDUCTIVE_AR = (
    "أنت خبير في {subject}.\n"
    "حلل الملاحظات والفرضيات التالية لتحديد الفرضية الأكثر قبولاً.\n\n"
    "{question_text}\n\n"
    "قدم تفكيرك خطوة بخطوة واختتم بالإجابة النهائية بالتنسيق 'الإجابة النهائية: الإجابة النهائية هي [[X]]' حيث X هو حرف الخيار الصحيح (أ أو ب)."
)

TOT_PROMPT_ABDUCTIVE_EN = (
    "You are an expert in {subject}.\n"
    "Solve the following abductive reasoning problem using a Tree of Thought approach:\n\n"
    "{question_text}\n\n"
    "1. Deconstruct the Observations: Identify key information in each observation.\n"
    "2. Analyze Hypotheses: For each hypothesis, evaluate how well it explains the observations.\n"
    "3. Generate Arguments: Develop arguments for and against each hypothesis based on the observations.\n"
    "4. Compare and Contrast: Weigh the evidence for each hypothesis.\n"
    "5. Synthesize and Decide: Determine the most plausible hypothesis.\n\n"
    "Provide your detailed reasoning simulating the Tree of Thought process described above. Conclude with the final answer in the format 'Final Answer: The final answer is [[X]]' where X is the correct letter choice (A or B)."
)
TOT_PROMPT_ABDUCTIVE_AR = (
    "أنت خبير في {subject}.\n"
    "حل مشكلة الاستدلال الافتراضي التالية باستخدام نهج شجرة الأفكار:\n\n"
    "{question_text}\n\n"
    "١. تحليل الملاحظات: حدد المعلومات الأساسية في كل ملاحظة.\n"
    "٢. تحليل الفرضيات: لكل فرضية، قم بتقييم مدى شرحها للملاحظات.\n"
    "٣. توليد الحجج: طور حججًا مؤيدة ومعارضة لكل فرضية بناءً على الملاحظات.\n"
    "٤. المقارنة والمقابلة: وازن الأدلة لكل فرضية.\n"
    "٥. الت综合 والاستنتاج: حدد الفرضية الأكثر قبولاً.\n\n"
    "قدم تفكيرك المفصل محاكيًا عملية شجرة الأفكار الموضحة أعلاه. اختتم بالإجابة النهائية بالتنسيق 'الإجابة النهائية: الإجابة النهائية هي [[X]]' حيث X هو حرف الخيار الصحيح (أ أو ب)."
)

# Mapping for abductive task labels to choice letters
# Assuming label '0' corresponds to hypothesis_1 (Choice A)
# and label '1' corresponds to hypothesis_2 (Choice B)
ABDUCTIVE_LABEL_TO_CHOICE_LETTER = {'1': 'A', '2': 'B'}
ALPA_EN_TO_AR_MAP = {alpa_en[k]: alpa_ar[k] for k in alpa_en if k in alpa_ar}


# --- End Prompt Templates ---

def format_abductive_question_text(obs1, obs2, hyp1, hyp2, lang_prompt):
    if lang_prompt == 'ar':
        return (
            f"بالنظر إلى الملاحظات التالية:\n"
            f"الملاحظة الأولى: {obs1}\n"
            f"الملاحظة الثانية: {obs2}\n\n"
            f"أي من الفرضيات التالية هي الأكثر قبولاً؟\n"
            f"(أ) {hyp1}\n"
            f"(ب) {hyp2}"
        )
    else:  # Default to English
        return (
            f"Given the following observations:\n"
            f"Observation 1: {obs1}\n"
            f"Observation 2: {obs2}\n\n"
            f"Which of the following hypotheses is more plausible?\n"
            f"(A) {hyp1}\n"
            f"(B) {hyp2}"
        )
    
def format_question(row, lang_prompt, lang_alpa, use_chain_of_thought, use_tree_of_thought):
    """
    Formats a single question row into the desired prompt based on arguments.

    Args:
        row (pd.Series): A row from the input DataFrame.
        lang_prompt (str): 'ar' or 'en' for the prompt language.
        lang_alpa (str): 'ar' or 'en' for the answer choice alphabet.
        use_chain_of_thought (bool): Whether to use CoT prompting.
        use_tree_of_thought (bool): Whether to use ToT prompting (overrides CoT).

    Returns:
        tuple: (formatted_prompt_text, list_of_option_labels)
               Returns (None, None) if essential data is missing.
    """
    try:
        alpa = alpa_ar if lang_alpa == 'ar' else alpa_en
        subject = row.get('Subject', 'the topic') # Default subject if missing

        # Extract options and create labels list
        labels = []
        options_parts = []
        for i, opt_key in enumerate(['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5']):
            option_text = row.get(opt_key)
            if pd.isna(option_text):
                break
            option_text = str(option_text).strip()
            letter = alpa.get(i)
            if letter is None: # Should not happen with defined alpa dicts
                 print(f"Warning: Missing alphabet mapping for index {i}")
                 continue
            labels.append(option_text) # Store the raw option text
            options_parts.append(f"{letter}- {option_text}") # Format for prompt display

        if not options_parts:
             print(f"Warning: No valid options found for row index {row.name if hasattr(row, 'name') else 'unknown'}. Skipping.")
             return None, None

        options_str = "\n".join(options_parts)

        # --- Construct Question Part ---
        backstory_text = str(row.get('BackStory', '')).strip()
        context_text = str(row.get('Context', '')).strip()
        question_text = str(row.get('Question', '')).strip()

        # Combine parts, handling potential empty strings and adding separators
        full_question_parts = []
        if backstory_text:
            full_question_parts.append(f"BackStory: {backstory_text}" if lang_prompt == 'en' else f"الخلفية: {backstory_text}")
        if context_text:
            full_question_parts.append(f"Context: {context_text}" if lang_prompt == 'en' else f"السياق: {context_text}")
        if question_text:
            full_question_parts.append(f"Question: {question_text}" if lang_prompt == 'en' else f"السؤال: {question_text}")
        else:
            # If Question field is empty, it's likely an error in the data
            print(f"Warning: Missing 'Question' field for row index {row.name if hasattr(row, 'name') else 'unknown'}. Skipping.")
            return None, None

        full_question_str = "\n\n".join(full_question_parts)
        # --- End Construct Question Part ---


        # --- Select and Format Prompt ---
        if lang_prompt == 'ar':
            if use_tree_of_thought:
                prompt_template = TOT_PROMPT_AR
            elif use_chain_of_thought:
                prompt_template = COT_PROMPT_AR
            else:
                prompt_template = ZERO_SHOT_PROMPT_AR
            # Translate subject for Arabic prompts if needed (using the mapping)
            subject_display = subject_ar.get(subject, subject) # Use original if no translation
            input_text = prompt_template.format(subject=subject_display, question=full_question_str, options=options_str)

        else: # lang_prompt == 'en'
            if use_tree_of_thought:
                prompt_template = TOT_PROMPT_EN
            elif use_chain_of_thought:
                prompt_template = COT_PROMPT_EN
            else:
                prompt_template = ZERO_SHOT_PROMPT_EN
            input_text = prompt_template.format(subject=subject, question=full_question_str, options=options_str)
        # --- End Select and Format Prompt ---

        return input_text, labels # Return formatted prompt and the list of raw option texts

    except Exception as e:
        print(f"Error formatting row index {row.name if hasattr(row, 'name') else 'unknown'}: {e}")
        return None, None


def load_and_format_data(args):
    """
    Loads data from the specified CSV file and formats questions based on arguments.
    Handles both 'mmlu' and 'abductive' task types.

    Args:
        args (argparse.Namespace): Parsed command-line arguments including
                                   task_type, data_file, lang_prompt, lang_alpa,
                                   chain_of_thought, tree_of_thought.

    Returns:
        tuple: (prompts, golds, labels_list, subjects, indices, abilities)
               Returns empty lists if data loading fails or no valid prompts generated.
    """
    prompts = []
    golds = []  # Gold standard answer letter (e.g., 'A', 'B', 'أ', 'ب')
    labels_list = []  # List of lists, each inner list contains the raw option/hypothesis texts
    subjects = []
    indices = []
    abilities = []

    # Determine data file path
    data_file_path = args.data_file
    if not data_file_path:
        if args.task_type == "mmlu":
            data_file_path = 'data/cleaned_output3.csv'  # Default for MMLU
        elif args.task_type == "abductive":
            data_file_path = 'data/abductive_data2.csv'  # Default for abductive
        else:
            print(f"Error: Unknown task_type '{args.task_type}' and no data_file provided.")
            return [], [], [], [], [], []

    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
        return [], [], [], [], [], []

    try:
        data_df = pd.read_csv(data_file_path, engine='python', on_bad_lines='warn')
        print(f"Read {len(data_df)} rows from {data_file_path} for task '{args.task_type}'.")
    except Exception as e:
        print(f"Error reading data file {data_file_path}: {e}")
        return [], [], [], [], [], []

    if args.task_type == "mmlu":
        # Filter out few-shot examples if necessary (specific to MMLU data structure)
        if 'is_few_shot' in data_df.columns:
            initial_count = len(data_df)
            data_df = data_df[data_df['is_few_shot'] == 0].copy()
            print(f"Filtered out {initial_count - len(data_df)} few-shot examples for MMLU task.")
        # else:
        #     print("Warning: 'is_few_shot' column not found in MMLU data. Assuming all are zero-shot.")

        # Mapping from answer key letter ('A', 'B', ...) to index (0, 1, ...)
        # This is used to find the correct letter from alpa_en or alpa_ar
        answer_key_to_index_map = {letter: idx for idx, letter in alpa_en.items()} # e.g. {'A':0, 'B':1}

        print("Formatting MMLU prompts...")
        for idx, row in data_df.iterrows():
            prompt, current_labels = format_question(
                row,
                args.lang_prompt,
                args.lang_alpa,
                args.chain_of_thought,
                args.tree_of_thought
            )

            if prompt is None or current_labels is None:
                # format_question prints its own warnings
                continue

            answer_key_original = row.get('Answer Key')
            if pd.isna(answer_key_original):
                print(f"Warning: MMLU task - Missing 'Answer Key' for row at original CSV index {idx}. Skipping.")
                continue

            answer_key_original = str(answer_key_original).strip().upper() # Normalize (e.g., 'A', 'B')
            
            gold_idx = answer_key_to_index_map.get(answer_key_original)

            if gold_idx is None:
                print(f"Warning: MMLU task - Invalid 'Answer Key' ('{answer_key_original}') found for row at original CSV index {idx}. Expected A, B, C, D, or E. Skipping.")
                continue
            
            # Determine the gold letter based on lang_alpa
            current_alpa = alpa_ar if args.lang_alpa == 'ar' else alpa_en
            gold_letter = current_alpa.get(gold_idx)

            if gold_letter is None: # Should not happen if gold_idx is valid
                print(f"Warning: MMLU task - Could not map gold index {gold_idx} to a letter for lang_alpa '{args.lang_alpa}'. Skipping row {idx}.")
                continue

            prompts.append(prompt)
            labels_list.append(current_labels)
            golds.append(gold_letter)
            subjects.append(row.get('Subject', 'Unknown_MMLU_Subject'))
            indices.append(row.get('INDEX', idx)) # Use 'INDEX' column if exists, else df index
            abilities.append(row.get('ABILITY', 'Unknown_MMLU_Ability'))

    elif args.task_type == "abductive":
        required_cols = ['observation_1', 'observation_2', 'hypothesis_1', 'hypothesis_2', 'label']
        if not all(col in data_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            print(f"Error: Abductive task - Missing required columns in {data_file_path}: {missing_cols}")
            return [], [], [], [], [], []

        print("Formatting abductive reasoning prompts...")
        for idx, row in data_df.iterrows():
            obs1 = str(row['observation_1'])
            obs2 = str(row['observation_2'])
            hyp1_text = str(row['hypothesis_1'])
            hyp2_text = str(row['hypothesis_2'])
            original_label = str(row['label']).strip() # '0' or '1'

            # Get the English gold letter ('A' or 'B')
            gold_choice_letter_en = ABDUCTIVE_LABEL_TO_CHOICE_LETTER.get(original_label)

            if gold_choice_letter_en is None:
                print(f"Warning: Abductive task - Unknown label '{original_label}' in row {idx} of {data_file_path}. Expected '0' or '1'. Skipping.")
                continue
            
            # Convert to target alphabet if needed
            if args.lang_alpa == 'ar':
            # Validate the mapping from English to Arabic
                assert 'A' in ALPA_EN_TO_AR_MAP and ALPA_EN_TO_AR_MAP['A'] == 'أ', "Mapping for 'A' to 'أ' is incorrect."
                assert 'B' in ALPA_EN_TO_AR_MAP and ALPA_EN_TO_AR_MAP['B'] == 'ب', "Mapping for 'B' to 'ب' is incorrect."

                gold_letter_final = ALPA_EN_TO_AR_MAP.get(gold_choice_letter_en)
                if gold_letter_final is None:  # Should not happen for 'A' or 'B'
                    print(f"Warning: Abductive task - Could not map English letter '{gold_choice_letter_en}' to Arabic. Skipping row {idx}.")
                    continue
            else:  # 'en' or other (defaults to English letter)
                # Validate the English mapping
                assert ABDUCTIVE_LABEL_TO_CHOICE_LETTER['1'] == 'A', "Label '1' should map to 'A'."
                assert ABDUCTIVE_LABEL_TO_CHOICE_LETTER['2'] == 'B', "Label '2' should map to 'B'."
                gold_letter_final = gold_choice_letter_en
            
            golds.append(gold_letter_final)
            
            current_hypotheses_texts = [hyp1_text, hyp2_text]
            labels_list.append(current_hypotheses_texts)

            # Format the question part of the prompt
            # Note: format_abductive_question_text already includes A/B or أ/ب based on lang_prompt
            # The lang_alpa argument for the prediction function will handle the expected output format.
            question_text_for_prompt = format_abductive_question_text(obs1, obs2, hyp1_text, hyp2_text, args.lang_prompt)
            
            current_subject = "abductive reasoning" # Generic subject for this task

            if args.tree_of_thought:
                PROMPT_TEMPLATE = TOT_PROMPT_ABDUCTIVE_AR if args.lang_prompt == 'ar' else TOT_PROMPT_ABDUCTIVE_EN
            elif args.chain_of_thought:
                PROMPT_TEMPLATE = COT_PROMPT_ABDUCTIVE_AR if args.lang_prompt == 'ar' else COT_PROMPT_ABDUCTIVE_EN
            else: # Zero-shot
                PROMPT_TEMPLATE = ZERO_SHOT_PROMPT_ABDUCTIVE_AR if args.lang_prompt == 'ar' else ZERO_SHOT_PROMPT_ABDUCTIVE_EN
            
            final_prompt = PROMPT_TEMPLATE.format(question_text=question_text_for_prompt, subject=current_subject)
            prompts.append(final_prompt)
            
            subjects.append(current_subject)
            abilities.append("reasoning_ability") # Generic ability for this task
            indices.append(idx) # Use DataFrame index for abductive task

    else:
        print(f"Error: Unknown task_type '{args.task_type}' in load_and_format_data.")
        return [], [], [], [], [], []

    if not prompts:
        print(f"Warning: No valid prompts were generated for task '{args.task_type}' from file '{data_file_path}'.")
    else:
        print(f"Successfully formatted {len(prompts)} prompts for task '{args.task_type}'.")
        
    return prompts, golds, labels_list, subjects, indices, abilities


# --- Deprecated Functions (Keep commented out or remove) ---
# def prepare_data_en(args):
#     # ... old implementation ...
#     pass

# def prepare_data_ar(args):
#     # ... old implementation ...
#     pass

# def prepare_data(args): # This function is now replaced by load_and_format_data
#     if args.lang_prompt == 'en':
#         return prepare_data_en(args)
#     elif args.lang_prompt == 'ar':
#         return prepare_data_ar(args)
# --- End Deprecated Functions ---