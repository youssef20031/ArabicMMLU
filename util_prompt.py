import pandas as pd


alpa_en = ['A.', 'B.', 'C.', 'D.', 'E.']

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

alpa_ar = ['أ-',
           'ب-',
           'ج-',
           'د-', 
           'ه-']


def prepare_data_en(args):
    if args.chain_of_thought:
        PROMPT = (
            "You are an expert in {subject}\n"
            "Analyze the given multiple-choice question and\n"
            "provide the correct answer using this approach:\n\n"
            "Carefully read the question and options\n"
            "Identify core {subject} concepts and required\n"
            "knowledge\n"
            "Analyze each option for relevance, accuracy,\n"
            "and consistency\n"
            "Consider {subject}-specific context and factors\n"
            "Use elimination and comparative analysis\n"
            "Select the most accurate answer\n"
            "Maintain objectivity, consider {subject}-specific\n"
            "sensitivities, and base your decision on verifiable\n"
            "facts and sound logical reasoning within {subject}\n"
            "Question:\n"
            "{question}\n"
            "{options}\n"
            "When asked to choose from options like 'A','B','C','D' your response must be only the single character representing your choice. Do not include any introductory phrases (e.g., 'Here's the answer:', 'I choose:'), explanations, or any other text before or after the selected character. For example, if the correct answer is  'B', your entire output should be just  'B'"
            "Correct option number is:"
        )
    else:
         # Updated prompt structure as requested
        PROMPT = (
            "Below is a multiple-choice question with a story and serveral answer options. Based on the content of the story and the given\n"
            "question, please infer the most likely answer and output the answer index.\n"
            "Note:\n"
            "(1) Please only output the most likely answer index in the format: [[Answer Index]], for example, if the most likely answer\n"
            "option is ‘أ. Handbag’, then output ‘[[أ]]’;\n"
            "(2) You must choose one of the given answer options ‘أ, ب, ج, د’ as the most likely answer, regardless of whether the story\n"
            "provides enough information. If you think there is not enough information in the story to choose an answer, please randomly\n"
            "output one of “[[أ]]”, “[[ب]]”, “[[ج]]”, or “[[د]]”;\n"
            "(3) Please only output the most likely answer index based on the given information, and do not output any other content.\n"
            "[Story]\n"
            "{Story}\n"
            "[Question]\n"
            "{Questions}\n"
            "[Candidate Answers]\n"
            "ا. {Option_a} ب. {Option_b} ج. {Option _c} د. {Option_d}" # Adjusted order to A, B, C, D
        )

    alpa = alpa_ar
    if args.lang_alpa == 'en':
        alpa = alpa_en

    inputs = []
    outputs = []
    outputs_options = []
    subjects = []  # added subjects list
    indices = [] # New list for INDEX
    abilities = [] # New list for ABILITY
    data = pd.read_csv('data/cleaned_output3.csv', engine='python', on_bad_lines='skip')
    data = data[data['is_few_shot'] == 0]

    for idx, row in data.iterrows():
        subject = row['Subject']
        subjects.append(subject)  # store subject for each question
        indices.append(row['INDEX']) # Store INDEX
        abilities.append(row['ABILITY']) # Store ABILITY
        level = level_en[row['Level']] if not pd.isna(row['Level']) else 'unknown'

        if args.chain_of_thought:
            backstory_text = f"BackStory: {str(row['BackStory']).strip()}\n\n" if not pd.isna(row['BackStory']) else ""
            context_text = f"Context: {str(row['Context']).strip()}\n\n" if not pd.isna(row['Context']) else ""
            question_text = f"{backstory_text}{context_text}Question: {str(row['Question']).strip()}"
    
            options_list = []
            for i, opt in enumerate(['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5']):
                if pd.isna(row[opt]):
                    break
                options_list.append(f"{alpa[i]} {row[opt]}")
            options_text = "\n".join(options_list)
    
            prompt_text = PROMPT.format(
                subject=subject,
                level=level,
                question=question_text,
                options=options_text
            )
        else:
            # Combine BackStory and Context for the Story part
            story_parts = []
            if not pd.isna(row['BackStory']):
                story_parts.append(str(row['BackStory']).strip())
            if not pd.isna(row['Context']):
                 story_parts.append(str(row['Context']).strip())
            story_text = "\n".join(story_parts) if story_parts else "N/A" # Handle case with no story/context

            question_field = str(row['Question']).strip()

            # Extract options, assuming at least 4 options based on the new prompt
            # Handle potential missing options gracefully (e.g., replace with "N/A" or similar)
            option_a = str(row['Option 1']).strip() if not pd.isna(row['Option 1']) else "N/A"
            option_b = str(row['Option 2']).strip() if not pd.isna(row['Option 2']) else "N/A"
            option_c = str(row['Option 3']).strip() if not pd.isna(row['Option 3']) else "N/A"
            option_d = str(row['Option 4']).strip() if not pd.isna(row['Option 4']) else "N/A"

            # Format the new prompt
            prompt_text = PROMPT.format(
                Story=story_text,
                Questions=question_field,
                Option_a=option_a,
                Option_b=option_b,
                Option_c=option_c, # Mapped Option 3 to C
                Option_d=option_d  # Mapped Option 4 to D
            )

        inputs.append(prompt_text)
        # Ensure Answer Key is stripped of whitespace before lookup
        answer_key = str(row['Answer Key']).strip()
        idx_label = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}.get(answer_key, -1) # Use .get for safety, default to -1 if key not found
        if idx_label == -1:
            print(f"Warning: Invalid Answer Key '{answer_key}' found at CSV index {idx}. Skipping row or assigning default.")
            # Decide how to handle invalid keys: skip row, assign default, etc.
            # For now, let's append a placeholder, but you might want to skip.
            outputs.append(-1) # Or some other indicator
            outputs_options.append([])
            continue # Or adjust as needed

        outputs.append(idx_label)

        options_list = []
        for i, opt_key in enumerate(['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5']):
             if pd.isna(row[opt_key]):
                 break
             options_list.append(f"{alpa[i]} {row[opt_key]}")
        outputs_options.append(options_list)

    # Return the new lists as well
    return inputs, outputs, outputs_options, subjects, indices, abilities



def prepare_data_ar(args):
    PROMPT = 'هذا سؤال [MAIN_META_DATA]. اختر الإجابة الصحيحة!\n\nسؤال: [INPUT]\n[OPTION]'
    if args.lora_weights == "x":
        PROMPT = f'{PROMPT}\n\nإجابة: '
    else:
        PROMPT = f'### Input:{PROMPT}\n\n### Output:\n'
        
    alpa = alpa_ar
    if args.lang_alpa == 'en':
        alpa = alpa_en

    inputs = []
    outputs = []
    outputs_options = []
    subjects = []  # added subjects list
    data = pd.read_csv('data/ArabicMMLUSS.csv')
    data = data[data['is_few_shot'] == 0]

    for idx, row in data.iterrows():
        # Get subject for each question and store it
        subject_value = subject_ar[row['Subject']]
        subjects.append(subject_value)
        level = "" if pd.isna(row['Level']) else ' ' + level_ar[row['Level']]
        country = "" if pd.isna(row['Country']) else ' ' + country_ar[row['Country']]
        main_meta_data = f"{subject_value}{level}{country}"
        
        backstory_text = f"الخلفية: {str(row['BackStory']).strip()}\n\n" if not pd.isna(row['BackStory']) else ""
        context_text = f"السياق: {str(row['Context']).strip()}\n\n" if not pd.isna(row['Context']) else ""
        question_text = f"{backstory_text}{context_text}السؤال: {str(row['Question']).strip()}"

        options = []
        for i, opt in enumerate(['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5']):
            if pd.isna(row[opt]):
                break
            options.append(f"{alpa[i]} {row[opt]}")
        inputs.append(
            PROMPT.replace('[MAIN_META_DATA]', main_meta_data)\
                  .replace('[INPUT]', question_text)\
                  .replace('[OPTION]', '\n'.join(options))
        )
        idx_label = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}[row['Answer Key']]
        outputs.append(idx_label)
        outputs_options.append(options)
    return inputs, outputs, outputs_options, subjects


def prepare_data(args):
    if args.lang_prompt == 'en':
        return prepare_data_en(args)
    elif args.lang_prompt == 'ar':
        return prepare_data_ar(args)
