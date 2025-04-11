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
            "You are an expert in {subject} at the {level} level.\n"
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
            "at the {level}. Question:\n"
            "{question}\n"
            "{options}\n"
            "Correct option number is:"
        )
    else:
        prefix = (
            "I give you a phrase of a dialogue between agents. I will reveal more parts of it later. "
            "At the end, I will give you a question you must answer. For each phrase, you must:\n"
            "# 1. Write down a succinct description of what each agent knows about the environment and about the other agents. "
            "Keep the description short and do not produce redundant information. \n"
            "Here's the dialogue:\n"
        )
        final_ask = (
            "This is the end of the dialogue. Now, answer the following question.\n"
            "Question: {question}{options}\n"
            "Think step by step, answer with one word and provide the answer between <answer></answer> tags.\n"
            "For example, reply with <answer>vase</answer>."
        )

    alpa = alpa_ar
    if args.lang_alpa == 'en':
        alpa = alpa_en

    inputs = []
    outputs = []
    outputs_options = []
    subjects = []  # added subjects list
    data = pd.read_csv('data/ArabicMMLUSS.csv', engine='python', on_bad_lines='skip')
    data = data[data['is_few_shot'] == 0]

    for idx, row in data.iterrows():
        subject = row['Subject']
        subjects.append(subject)  # store subject for each question
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
            if not pd.isna(row['BackStory']):
                task_text = f"BackStory: {str(row['BackStory']).strip()}"
            else:
                task_text = ""
            context_text = ""
            if not pd.isna(row['Context']):
                context_text = f"Context: {str(row['Context']).strip()}"
            
            dialogue = task_text
            if context_text:
                dialogue += "\n" + context_text

            question_field = str(row['Question']).strip()
            
            options_list = []
            for i, opt in enumerate(['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5']):
                if pd.isna(row[opt]):
                    break
                options_list.append(f"{alpa[i]} {row[opt]}")
            options_text = ""
            if options_list:
                options_text = "\nOptions:\n" + "\n".join(options_list)
    
            prompt_text = prefix + "\n" + dialogue + "\n\n" + final_ask.format(question=question_field, options=options_text)
    
        inputs.append(prompt_text)
        idx_label = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}[row['Answer Key']]
        outputs.append(idx_label)
        outputs_options.append(options_list)
    return inputs, outputs, outputs_options, subjects


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
