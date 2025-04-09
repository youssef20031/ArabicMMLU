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
        PROMPT = (
            "You are an expert in {subject} at the {level} level.\n"
            "Question:\n"
            "{question}\n"
            "{options}\n"
            "Correct option number is:"
        )

    alpa = alpa_ar
    if args.lang_alpa == 'en':
        alpa = alpa_en

    inputs = []
    outputs = []
    outputs_options = []
    data = pd.read_csv('data/ArabicMMLUSS.csv', engine='python', on_bad_lines='skip')
    data = data[data['is_few_shot'] == 0]

    for idx, row in data.iterrows():
        subject = row['Subject']
        level = level_en[row['Level']] if not pd.isna(row['Level']) else 'unknown'

        # Build BackStory and Context texts separately
        backstory_text = f"BackStory: {str(row['BackStory']).strip()}\n\n" if not pd.isna(row['BackStory']) else ""
        context_text = f"Context: {str(row['Context']).strip()}\n\n" if not pd.isna(row['Context']) else ""
        question_text = f"{backstory_text}{context_text}Question: {str(row['Question']).strip()}"

        options_list = []
        for i, opt in enumerate(['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5']):
            if pd.isna(row[opt]):
                break
            options_list.append(f"{alpa[i]} {row[opt]}")
        options_text = '\n'.join(options_list)

        prompt_text = PROMPT.format(
            subject=subject,
            level=level,
            question=question_text,
            options=options_text
        )
        inputs.append(prompt_text)
        idx_label = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}[row['Answer Key']]
        outputs.append(idx_label)
        outputs_options.append(options_list)
    return inputs, outputs, outputs_options


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
    data = pd.read_csv('data/ArabicMMLUSS.csv')
    data = data[data['is_few_shot'] == 0]

    for idx, row in data.iterrows():
        subject = subject_ar[row['Subject']]
        level = "" if pd.isna(row['Level']) else ' ' + level_ar[row['Level']]
        country = "" if pd.isna(row['Country']) else ' ' + country_ar[row['Country']]
        main_meta_data = f"{subject}{level}{country}"
        
        # Build BackStory and Context texts separately (labels can be adjusted for Arabic as needed)
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
    return inputs, outputs, outputs_options


def prepare_data(args):
    if args.lang_prompt == 'en':
        return prepare_data_en(args)
    elif args.lang_prompt == 'ar':
        return prepare_data_ar(args)

