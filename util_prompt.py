import pandas as pd
import os # Ensure os is imported
import ast # Keep existing import

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
    "Please answer with only the single letter corresponding to the correct option."
)
ZERO_SHOT_PROMPT_AR = (
    "السؤال التالي هو سؤال متعدد الخيارات حول {subject}.\n\n"
    "{question}\n"
    "{options}\n\n"
    "الرجاء الإجابة فقط بحرف الخيار الصحيح الموافق للإجابة الصحيحة."
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

# --- Abductive Reasoning Prompts ---
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
# --- End Abductive Reasoning Prompts ---

# --- Deductive Reasoning Prompts (Multiple Choice Format) ---
ZERO_SHOT_PROMPT_DEDUCTIVE_EN = (
    "The following is a deductive reasoning question about {subject}.\n\n"
    "{question}\n"
    "{options}\n\n"
    "Please answer with only the single letter corresponding to the most logically sound conclusion."
)
ZERO_SHOT_PROMPT_DEDUCTIVE_AR = (
    "السؤال التالي هو سؤال استدلال استنباطي حول {subject}.\n\n"
    "{question}\n"
    "{options}\n\n"
    "الرجاء الإجابة فقط بحرف الخيار الموافق للاستنتاج الأكثر سلامة منطقيًا."
)

COT_PROMPT_DEDUCTIVE_EN = (
    "You are an expert in {subject}.\n"
    "Analyze the given deductive reasoning problem and determine which conclusion logically follows from the premises.\n\n"
    "Premises/Question:\n{question}\n\n"
    "Potential Conclusions (Options):\n{options}\n\n"
    "Instructions:\n"
    "1. Carefully read the premises (in the question) and each potential conclusion (in the options).\n"
    "2. For each option, analyze its logical connection to the premises. Determine if it necessarily follows, assuming the premises are true.\n"
    "3. Identify any logical fallacies or sound deductive steps for each option.\n"
    "4. Provide your step-by-step reasoning for evaluating each option against the premises.\n"
    "5. Conclude with the final answer in the format 'Final Answer: The final answer is [[X]]' where X is the letter of the correct conclusion."
)
COT_PROMPT_DEDUCTIVE_AR = (
    "أنت خبير في {subject}.\n"
    "حلل مشكلة الاستدلال الاستنباطي المعطاة وحدد أي استنتاج يتبع منطقيًا من المقدمات.\n\n"
    "المقدمات/السؤال:\n{question}\n\n"
    "الاستنتاجات المحتملة (الخيارات):\n{options}\n\n"
    "التعليمات:\n"
    "١. اقرأ بعناية المقدمات (في السؤال) وكل استنتاج محتمل (في الخيارات).\n"
    "٢. لكل خيار، حلل ارتباطه المنطقي بالمقدمات. حدد ما إذا كان يتبع بالضرورة، بافتراض أن المقدمات صحيحة.\n"
    "٣. حدد أي مغالطات منطقية أو خطوات استنباطية سليمة لكل خيار.\n"
    "٤. قدم تفكيرك خطوة بخطوة لتقييم كل خيار مقابل المقدمات.\n"
    "٥. اختتم بالإجابة النهائية بالتنسيق 'الإجابة النهائية: الإجابة النهائية هي [[X]]' حيث X هو حرف الاستنتاج الصحيح."
)

TOT_PROMPT_DEDUCTIVE_EN = (
    "You are an expert in {subject}.\n"
    "Solve the following deductive reasoning problem using a Tree of Thought approach to determine which conclusion logically follows from the premises.\n\n"
    "Premises/Question:\n{question}\n\n"
    "Potential Conclusions (Options):\n{options}\n\n"
    "Instructions for Tree of Thought Simulation:\n"
    "1. Deconstruct the Problem: Clearly identify the core premises provided in the question.\n"
    "2. For each Option (Potential Conclusion): Generate lines of reasoning (thought paths) to evaluate if it logically and necessarily follows from the given premises.\n"
    "3. Evaluate Each Path: Assess the validity of each reasoning path. Does it adhere to rules of deductive logic? Are there counterexamples or fallacies?\n"
    "4. Explore Promising Options: Elaborate on why certain options appear to be logically sound conclusions based on the premises. Detail the deductive steps.\n"
    "5. Self-Reflect and Prune: Review all evaluated options. Discard options that are demonstrably not supported by the premises or involve logical fallacies. Refine the arguments for plausible options.\n"
    "6. Synthesize and Decide: Based on the most robust and logically sound evaluation, determine which option is the correct conclusion that necessarily follows from the premises.\n\n"
    "Provide your detailed reasoning simulating the Tree of Thought process. Conclude with the final answer in the format 'Final Answer: The final answer is [[X]]' where X is the letter of the correct conclusion."
)
TOT_PROMPT_DEDUCTIVE_AR = (
    "أنت خبير في {subject}.\n"
    "حل مشكلة الاستدلال الاستنباطي التالية باستخدام نهج شجرة الأفكار لتحديد أي استنتاج يتبع منطقيًا من المقدمات.\n\n"
    "المقدمات/السؤال:\n{question}\n\n"
    "الاستنتاجات المحتملة (الخيارات):\n{options}\n\n"
    "تعليمات لمحاكاة شجرة الأفكار:\n"
    "١. فكك المشكلة: حدد بوضوح المقدمات الأساسية الواردة في السؤال.\n"
    "٢. لكل خيار (استنتاج محتمل): ولّد خطوط تفكير (مسارات فكرية) لتقييم ما إذا كان يتبع منطقيًا وبالضرورة من المقدمات المعطاة.\n"
    "٣. قيّم كل مسار: قيّم مدى صحة كل مسار تفكير. هل يلتزم بقواعد المنطق الاستنباطي؟ هل هناك أمثلة مضادة أو مغالطات؟\n"
    "٤. استكشف الخيارات الواعدة: توسع في شرح لماذا تبدو بعض الخيارات استنتاجات سليمة منطقيًا بناءً على المقدمات. فصل الخطوات الاستنباطية.\n"
    "٥. تأمل ذاتيًا وقلم: راجع جميع الخيارات التي تم تقييمها. تجاهل الخيارات التي ثبت أنها غير مدعومة بالمقدمات أو تنطوي على مغالطات منطقية. نقح الحجج للخيارات المعقولة.\n"
    "٦. ركب وقرر: بناءً على التقييم الأكثر قوة وسليمة منطقيًا، حدد أي خيار هو الاستنتاج الصحيح الذي يتبع بالضرورة من المقدمات.\n\n"
    "قدم تفكيرك المفصل محاكيًا عملية شجرة الأفكار. اختتم بالإجابة النهائية بالتنسيق 'الإجابة النهائية: الإجابة النهائية هي [[X]]' حيث X هو حرف الاستنتاج الصحيح."
)
# --- End Deductive Reasoning Prompts ---

# Mapping for abductive task labels to choice letters
ABDUCTIVE_LABEL_TO_CHOICE_LETTER = {'1': 'A', '2': 'B'}

# Define English and Arabic alphabets for options
alpa_en = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
alpa_ar = {0: 'أ', 1: 'ب', 2: 'ج', 3: 'د', 4: 'ه'}

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
    
def format_question(row, lang_prompt, lang_alpa, task_type="mmlu", use_cot=False, use_tot=False):
    """
    Formats a single question row into the desired prompt based on arguments.

    Args:
        row (pd.Series): A row from the input DataFrame.
        lang_prompt (str): 'ar' or 'en' for the prompt language.
        lang_alpa (str): 'ar' or 'en' for the answer choice alphabet.
        task_type (str): Type of task ('mmlu', 'abductive', 'deductive').
        use_cot (bool): Whether to use CoT prompting.
        use_tot (bool): Whether to use ToT prompting (overrides CoT).

    Returns:
        tuple: (formatted_prompt_text, list_of_option_labels)
               Returns (None, None) if essential data is missing.
    """
    try:
        alpa = alpa_ar if lang_alpa == 'ar' else alpa_en
        subject = row.get('Subject', 'the topic') # Default subject if missing

        if task_type == "mmlu":
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
                if use_tot:
                    prompt_template = TOT_PROMPT_AR
                elif use_cot:
                    prompt_template = COT_PROMPT_AR
                else:
                    prompt_template = ZERO_SHOT_PROMPT_AR
                # Translate subject for Arabic prompts if needed (using the mapping)
                subject_display = subject_ar.get(subject, subject) # Use original if no translation
                input_text = prompt_template.format(subject=subject_display, question=full_question_str, options=options_str)

            else: # lang_prompt == 'en'
                if use_tot:
                    prompt_template = TOT_PROMPT_EN
                elif use_cot:
                    prompt_template = COT_PROMPT_EN
                else:
                    prompt_template = ZERO_SHOT_PROMPT_EN
                input_text = prompt_template.format(subject=subject, question=full_question_str, options=options_str)
            # --- End Select and Format Prompt ---

            return input_text, labels # Return formatted prompt and the list of raw option texts

        elif task_type == "abductive":
            question_text = row['question'] # Assuming 'question' column holds the full abductive problem text
            subject = row.get('subject', 'logical reasoning') 
            if lang_prompt == 'ar':
                prompt_template = ZERO_SHOT_PROMPT_ABDUCTIVE_AR
            else:
                prompt_template = ZERO_SHOT_PROMPT_ABDUCTIVE_EN
            
            formatted_prompt = prompt_template.format(question_text=question_text, subject=subject)
            options_list = [row['hyp1'], row['hyp2']] 
            gold_index = row['gold_index'] 
            return formatted_prompt, options_list, gold_index, subject

        elif task_type == "deductive":
            main_question_text = str(row.get('question', '')).strip()
            if not main_question_text:
                print(f"Warning: Deductive task - Missing 'question' field for row index {row.name if hasattr(row, 'name') else 'unknown'}. Skipping.")
                return None, None, None

            subject = str(row.get('subject', 'logical reasoning')).strip() # Default subject

            # Define the keys for options in the CSV for deductive task
            option_col_keys = {
                0: 'option_a',
                1: 'option_b',
                2: 'option_c',
                3: 'option_d',
                4: 'option_e' # Assuming up to 5 options like MMLU
            }

            labels = [] # List to store the text of the options
            options_parts = [] # List to store formatted options for the prompt string

            for i in range(len(option_col_keys)): # Iterate up to the number of defined option keys
                opt_csv_key = option_col_keys.get(i)
                if not opt_csv_key: continue # Should not happen with current dict

                option_text = row.get(opt_csv_key)
                if pd.isna(option_text): # Stop if an option is missing (e.g. only A, B, C are present)
                    break 
                
                option_text = str(option_text).strip()
                letter = alpa.get(i) # Get 'A', 'B', 'C' or 'أ', 'ب', 'ج'
                
                if letter is None:
                     print(f"Warning: Deductive task - Missing alphabet mapping for option index {i}. Skipping option.")
                     continue
                
                labels.append(option_text) # Store the raw option text
                options_parts.append(f"{letter}- {option_text}") # Format for prompt display (e.g., "A- Conclusion 1")

            if not labels: # If no valid options were found
                 print(f"Warning: Deductive task - No valid options (option_a, option_b, etc.) found for row index {row.name if hasattr(row, 'name') else 'unknown'}. Skipping.")
                 return None, None, None
            
            options_str = "\n".join(options_parts)

            # Select and Format Prompt for Deductive
            if lang_prompt == 'ar':
                if use_tot:
                    prompt_template = TOT_PROMPT_DEDUCTIVE_AR
                elif use_cot:
                    prompt_template = COT_PROMPT_DEDUCTIVE_AR
                else:
                    prompt_template = ZERO_SHOT_PROMPT_DEDUCTIVE_AR
                subject_display = subject_ar.get(subject, subject) if subject_ar else subject # Translate subject if map exists
            else: # lang_prompt == 'en'
                if use_tot:
                    prompt_template = TOT_PROMPT_DEDUCTIVE_EN
                elif use_cot:
                    prompt_template = COT_PROMPT_DEDUCTIVE_EN
                else:
                    prompt_template = ZERO_SHOT_PROMPT_DEDUCTIVE_EN
                subject_display = subject # No translation for English subject needed here
            
            input_text = prompt_template.format(subject=subject_display, question=main_question_text, options=options_str)
            
            # For deductive, format_question returns (input_text, labels, subject)
            # gold_index is handled in load_and_format_data based on 'answer' column
            return input_text, labels, subject_display

        else:
            print(f"Error: Unknown task_type '{task_type}' in format_question.")
            return None, None

    except Exception as e:
        print(f"Error formatting row index {row.name if hasattr(row, 'name') else 'unknown'}: {e}")
        return None, None


def load_and_format_data(args):
    """
    Loads data from the specified CSV file and formats questions based on arguments.
    Handles 'mmlu', 'abductive', and 'deductive' task types.

    Args:
        args (argparse.Namespace): Parsed command-line arguments including
                                   task_type, data_file, lang_prompt, lang_alpa,
                                   chain_of_thought, tree_of_thought.

    Returns:
        tuple: (prompts, golds, labels_list, subjects, indices, abilities)
               Returns empty lists if data loading fails or no valid prompts generated.
    """
    prompts = []
    golds = []  
    labels_list = []  
    subjects = []
    indices = []
    abilities = []

    data_file_path = args.data_file
    if not data_file_path:
        if args.task_type == "mmlu":
            data_file_path = 'data/cleaned_output3.csv'  
        elif args.task_type == "abductive":
            data_file_path = 'data/abductive_data2.csv'  
        elif args.task_type == "deductive":
            data_file_path = 'data/deductive_data.csv'  
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
        if 'is_few_shot' in data_df.columns:
            initial_count = len(data_df)
            data_df = data_df[data_df['is_few_shot'] == 0].copy()
            print(f"Filtered out {initial_count - len(data_df)} few-shot examples for MMLU task.")

        answer_key_to_index_map = {letter: idx for idx, letter in alpa_en.items()} 

        print("Formatting MMLU prompts...")
        for idx, row in data_df.iterrows():
            prompt, current_labels = format_question(
                row,
                args.lang_prompt,
                args.lang_alpa,
                task_type="mmlu",
                use_cot=args.chain_of_thought,
                use_tot=args.tree_of_thought
            )

            if prompt is None or current_labels is None:
                continue

            answer_key_original = row.get('Answer Key')
            if pd.isna(answer_key_original):
                print(f"Warning: MMLU task - Missing 'Answer Key' for row at original CSV index {idx}. Skipping.")
                continue

            answer_key_original = str(answer_key_original).strip().upper() 
            
            gold_idx = answer_key_to_index_map.get(answer_key_original)

            if gold_idx is None:
                print(f"Warning: MMLU task - Invalid 'Answer Key' ('{answer_key_original}') found for row at original CSV index {idx}. Expected A, B, C, D, or E. Skipping.")
                continue
            
            current_alpa = alpa_ar if args.lang_alpa == 'ar' else alpa_en
            gold_letter = current_alpa.get(gold_idx)

            if gold_letter is None: 
                print(f"Warning: MMLU task - Could not map gold index {gold_idx} to a letter for lang_alpa '{args.lang_alpa}'. Skipping row {idx}.")
                continue

            prompts.append(prompt)
            labels_list.append(current_labels)
            golds.append(gold_letter)
            subjects.append(row.get('Subject', 'Unknown_MMLU_Subject'))
            indices.append(row.get('INDEX', idx)) 
            abilities.append(row.get('ABILITY', 'Unknown_MMLU_Ability'))

    elif args.task_type == "abductive":
        required_cols = ['question', 'hyp1', 'hyp2', 'gold_index']
        if not all(col in data_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            print(f"Error: Abductive task - Missing required columns in {data_file_path}: {missing_cols}")
            return [], [], [], [], [], []

        print("Formatting abductive reasoning prompts...")
        for idx, row in data_df.iterrows():
            formatted_prompt, options, gold_idx, subj = format_question(
                row,
                args.lang_prompt,
                args.lang_alpa,
                task_type="abductive",
                use_cot=args.chain_of_thought, 
                use_tot=args.tree_of_thought
            )
            prompts.append(formatted_prompt)
            golds.append(gold_idx)
            labels_list.append(options)
            subjects.append(subj)
            indices.append(idx) 
            abilities.append("reasoning_ability") 

    elif args.task_type == "deductive":
        # Define required columns based on the new structure
        required_cols = ['question', 'option_a', 'option_b', 'answer'] 
        # Check for essential columns
        if not all(col in data_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data_df.columns]
            print(f"Error: Deductive task - Missing essential columns in {data_file_path}: {missing_cols}. Need at least 'question', 'option_a', 'option_b', 'answer'.")
            return [], [], [], [], [], []

        # Map for converting answer letter (A, B, C, D) to numeric index (0, 1, 2, 3)
        alpa_en_to_idx_map = {letter: idx for idx, letter in alpa_en.items()}
        alpa_ar_to_idx_map = {letter: idx for idx, letter in alpa_ar.items()}


        print("Formatting deductive reasoning prompts...")
        for idx, row in data_df.iterrows():
            # format_question for deductive now returns (input_text, labels, subject)
            formatted_prompt, current_labels, subj = format_question(
                row,
                args.lang_prompt,
                args.lang_alpa,
                task_type="deductive",
                use_cot=args.chain_of_thought,
                use_tot=args.tree_of_thought
            )

            if formatted_prompt is None or current_labels is None: # Check if format_question skipped the row
                continue

            answer_key_original = row.get('answer') # Get the gold answer letter (e.g., 'A', 'B', 'أ', 'ب')
            if pd.isna(answer_key_original):
                print(f"Warning: Deductive task - Missing 'answer' for row at original CSV index {idx}. Skipping.")
                continue
            
            answer_key_original = str(answer_key_original).strip() # Normalize

            # Convert the gold answer letter to a numeric index
            gold_idx_numeric = alpa_ar_to_idx_map.get(answer_key_original) # Try Arabic first
            if gold_idx_numeric is None:
                gold_idx_numeric = alpa_en_to_idx_map.get(answer_key_original.upper()) # Then try English (uppercase)


            if gold_idx_numeric is None:
                print(f"Warning: Deductive task - Invalid 'answer' key ('{answer_key_original}') found for row at original CSV index {idx}. Expected A-E or أ-ه. Skipping.")
                continue
            
            # Ensure the gold index is within the range of available options for this row
            if not (0 <= gold_idx_numeric < len(current_labels)):
                print(f"Warning: Deductive task - 'answer' key ('{answer_key_original}' -> index {gold_idx_numeric}) is out of range for the number of options ({len(current_labels)}) found for row {idx}. Options: {current_labels}. Skipping.")
                continue

            # Determine the gold letter based on lang_alpa (e.g., 'A' or 'أ')
            current_alpa_map = alpa_ar if args.lang_alpa == 'ar' else alpa_en
            gold_letter_for_output = current_alpa_map.get(gold_idx_numeric)

            if gold_letter_for_output is None:
                print(f"Warning: Deductive task - Could not map gold index {gold_idx_numeric} to a letter for lang_alpa '{args.lang_alpa}'. Skipping row {idx}.")
                continue

            prompts.append(formatted_prompt)
            labels_list.append(current_labels) # List of option texts
            golds.append(gold_letter_for_output) # Gold letter ('A', 'أ', etc.)
            subjects.append(subj)
            indices.append(row.get('INDEX', idx)) # Use 'INDEX' if exists, else DataFrame index
            abilities.append(row.get('ABILITY', 'deductive_reasoning')) # Or derive from data

    else:
        print(f"Error: Unknown task_type '{args.task_type}' in load_and_format_data.")
        return [], [], [], [], [], []

    if not prompts:
        print(f"Warning: No valid prompts were generated for task '{args.task_type}' from file '{data_file_path}'.")
    else:
        print(f"Successfully formatted {len(prompts)} prompts for task '{args.task_type}'.")
        
    return prompts, golds, labels_list, subjects, indices, abilities