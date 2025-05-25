import os
import csv
from groq import Groq, APIError # Import APIError for specific error handling
import time # For potential rate limiting with large files and retry logic
import sys # For checking if running in Colab
from tqdm import tqdm # For progress bars

# --- Configuration ---
INPUT_DATA_FOLDER = 'tobetranslated6'  # Folder containing input CSV files
OUTPUT_DATA_FOLDER = 'tobetranslated6' # Folder to save translated CSV files

# Ensure your GROQ_API_KEY is set as an environment variable.
# In Google Colab, you can set this in the "Secrets" tab (recommended)
# API_KEY = "YOUR_GROQ_API_KEY_HERE" 
API_KEY = os.environ.get("GROQ_API_KEY")

MODEL_NAME = "llama-3.1-8b-instant" # Groq model for translation

# --- Retry Configuration ---
MAX_RETRIES = 5  # Maximum number of retries for API calls
INITIAL_BACKOFF_SECONDS = 1  # Initial wait time in seconds for retries
MAX_BACKOFF_SECONDS = 60 # Maximum wait time for a single retry

# --- Groq Client Initialization ---
client = None # Initialize client to None
if API_KEY:
    try:
        client = Groq(api_key=API_KEY)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        print("Please ensure your API key is correct and valid.")
else:
    print("Warning: GROQ_API_KEY environment variable not set.")
    print("Please set it in your environment or Colab Secrets for the script to function.")


def translate_texts_groq(texts_to_translate: list[str]) -> list[str]:
    """
    Translates a list of English texts to Arabic using the Groq API, with retry logic.

    Args:
        texts_to_translate (list[str]): A list of English strings to translate.
    Returns:
        list of str: A list of translated Arabic strings or error messages.
    """
    if not client:
        return [f"ERROR_GROQ_CLIENT_NOT_INITIALIZED_FOR_ROW" for _ in texts_to_translate]

    if not texts_to_translate:
        return []

    final_translations_template = [""] * len(texts_to_translate)
    non_empty_texts_for_api = []
    indices_of_non_empty_texts = [] # To map translations back to original list structure if it had empty strings

    for i, text in enumerate(texts_to_translate):
        if text and text.strip(): # Only send non-empty, non-whitespace text to API
            non_empty_texts_for_api.append(text.strip())
            indices_of_non_empty_texts.append(i)
        else:
            final_translations_template[i] = "" # Preserve original empty/whitespace strings

    if not non_empty_texts_for_api: # All original texts were empty or whitespace
        return final_translations_template

    system_prompt = (
        "You are an expert English to Arabic translator. "
        "Translate the following English texts to Arabic. "
        "Each input text is provided on a new line. "
        "Provide each corresponding Arabic translation on a new line, in the exact same order as the input. "
        "Do not add any extra text, numbers, bullet points, or explanations before or after the translations. "
        "If an input text is just a placeholder or seems untranslatable in context, try to return a sensible Arabic equivalent or the original placeholder. For numbers, return the Arabic numeral if possible, or the original number."
    )
    
    user_content = "\n".join(non_empty_texts_for_api)

    current_retry = 0
    backoff_time = INITIAL_BACKOFF_SECONDS
    
    # This will hold translations for non_empty_texts_for_api
    api_call_successful_translations = [f"ERROR_TRANSLATION_ATTEMPT_FAILED_PER_ITEM" for _ in non_empty_texts_for_api]


    while current_retry < MAX_RETRIES:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                model=MODEL_NAME,
                temperature=0.1,
                top_p=0.9,
                max_tokens=int(len(user_content) * 3.0) + 200 
            )
            response_content = chat_completion.choices[0].message.content.strip()
            api_translated_lines = response_content.split('\n')
            cleaned_api_translations = [line.strip() for line in api_translated_lines if line.strip() or line == ""]

            if len(cleaned_api_translations) == len(non_empty_texts_for_api):
                api_call_successful_translations = cleaned_api_translations # Store successful translations
                break # Successful translation, exit retry loop
            else:
                tqdm.write(f"      Warning: Mismatch in translated lines on attempt {current_retry + 1}. Expected {len(non_empty_texts_for_api)}, got {len(cleaned_api_translations)}.")
                if current_retry == MAX_RETRIES -1: 
                    # Fill with error for the last attempt if mismatch
                    for i in range(len(non_empty_texts_for_api)):
                        api_call_successful_translations[i] = "TRANSLATION_ERROR_LINE_COUNT_MISMATCH_FINAL"
                    break # Exit loop
                # Continue to retry for mismatch if not last attempt
        
        except APIError as e: 
            tqdm.write(f"      Groq APIError on attempt {current_retry + 1}/{MAX_RETRIES}: Status {e.status_code}, Message: {str(e.message)[:100]}...")
            if e.status_code in [500, 502, 503, 504, 429]: 
                if current_retry < MAX_RETRIES - 1:
                    actual_backoff_time = backoff_time
                    if e.status_code == 429 and e.headers and isinstance(e.headers.get("retry-after"), (str, int)):
                        try:
                            actual_backoff_time += int(e.headers.get("retry-after"))
                        except ValueError:
                            pass # Could not parse retry-after, use standard backoff
                    
                    actual_backoff_time = min(actual_backoff_time, MAX_BACKOFF_SECONDS)
                    tqdm.write(f"      Retrying in {actual_backoff_time:.2f} seconds...")
                    time.sleep(actual_backoff_time)
                    backoff_time = min(backoff_time * 2, MAX_BACKOFF_SECONDS) 
                else: # Max retries reached for this specific error type
                    for i in range(len(non_empty_texts_for_api)):
                         api_call_successful_translations[i] = f"TRANSLATION_ERROR_API_MAX_RETRIES:{e.status_code}"
                    break # Exit loop
            else: 
                for i in range(len(non_empty_texts_for_api)):
                     api_call_successful_translations[i] = f"TRANSLATION_ERROR_API_CLIENT_SIDE:{e.status_code}"
                break # Exit loop, non-retryable
        except Exception as e: 
            tqdm.write(f"      Unexpected error during Groq API call on attempt {current_retry + 1}/{MAX_RETRIES}: {str(e)[:100]}...")
            if current_retry < MAX_RETRIES - 1:
                tqdm.write(f"      Retrying in {backoff_time} seconds due to unexpected error...")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, MAX_BACKOFF_SECONDS)
            else: # Max retries for unexpected error
                for i in range(len(non_empty_texts_for_api)):
                    api_call_successful_translations[i] = f"TRANSLATION_ERROR_UNEXPECTED_MAX_RETRIES:{str(e)[:50]}"
                break # Exit loop
        
        current_retry += 1 # Increment retry counter here after handling exceptions or mismatch

    # Populate the final_translations_template with results (or errors) from api_call_successful_translations
    for i, original_idx in enumerate(indices_of_non_empty_texts):
        if i < len(api_call_successful_translations): # Ensure we don't go out of bounds
            final_translations_template[original_idx] = api_call_successful_translations[i]
        # else: it implies an earlier logic error, should remain as initialized error string

    return final_translations_template


def create_sample_csv(file_path):
    """Creates a sample CSV file if it doesn't exist."""
    print(f"Creating a sample CSV file at: {file_path}")
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f_dummy:
            writer = csv.writer(f_dummy)
            writer.writerow(['observation_1', 'observation_2', 'hypothesis_1', 'hypothesis_2', 'label'])
            writer.writerow(['The sky is blue.', 'The sun is bright.', 'Perhaps it is daytime.', 'It is sunny.', 'positive outlook'])
            writer.writerow(['Rain is falling.', 'Clouds are dark.', 'A storm might be coming.', 'It is cloudy.', 'negative weather'])
            writer.writerow(['The cat sat on the mat.', '', 'The cat is comfortable.', 'The mat is soft.', 'neutral observation'])
        print(f"Sample file '{os.path.basename(file_path)}' created in '{os.path.dirname(file_path)}'.")
    except IOError as e:
        print(f"Error creating sample CSV file '{file_path}': {e}")


def process_single_csv(input_file_path, output_file_path):
    """Reads a single CSV, translates its content (skipping 'label'), and writes to a new CSV."""
    translated_rows_data = []
    header = []
    file_processed_successfully = True
    rows_processed_count = 0
    rows_with_translation_issues = 0
    total_rows_in_file = 0 

    try:
        try:
            with open(input_file_path, 'r', newline='', encoding='utf-8-sig') as infile_count:
                reader_count = csv.reader(infile_count)
                header_temp = next(reader_count, None)
                if header_temp:
                    total_rows_in_file = sum(1 for _ in reader_count) # Use _ if row content isn't needed
        except Exception as e_count:
            print(f"  Warning: Could not pre-count rows in {os.path.basename(input_file_path)} due to {e_count}. Progress bar for rows might be indeterminate.")
            total_rows_in_file = None 

        with open(input_file_path, 'r', newline='', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            try:
                header = next(reader)
            except StopIteration:
                print(f"  Error: Input file '{os.path.basename(input_file_path)}' is empty or has no header.")
                with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile_err:
                    writer_err = csv.writer(outfile_err)
                    writer_err.writerow([f"Error: Source file '{os.path.basename(input_file_path)}' was empty or had no header."])
                return False
            
            translated_rows_data.append(header)
            
            label_column_index = -1
            if 'label' in header:
                try:
                    label_column_index = header.index('label')
                except ValueError:
                    tqdm.write("      Warning: 'label' column specified in header but not found. Translating all columns.")
            else:
                tqdm.write("      Warning: 'label' column not found in header. Translating all columns.")

            row_iterator = tqdm(reader, total=total_rows_in_file, desc=f"  Translating rows in {os.path.basename(input_file_path)}", unit="row", leave=False)
            
            for i, row in enumerate(row_iterator):
                rows_processed_count += 1
                if len(row) != len(header):
                    tqdm.write(f"      Warning: Malformed row {rows_processed_count}. Expected {len(header)} columns, got {len(row)}: {str(row)[:100]}... Writing as is with error marker.")
                    error_row = list(row) + ["MALFORMED_ROW_STRUCTURE"] * (len(header) - len(row)) if len(row) < len(header) else list(row[:len(header)])
                    if "MALFORMED_ROW_STRUCTURE" not in error_row[-1] and len(error_row) == len(header): # Add marker if not already there due to padding
                         error_row[-1] += " (POTENTIALLY_TRUNCATED_OR_EXTENDED_MALFORMED_ROW)"
                    translated_rows_data.append(error_row)
                    rows_with_translation_issues +=1
                    continue

                original_label_value = None
                cells_to_translate_this_row = []

                if label_column_index != -1 and label_column_index < len(row):
                    original_label_value = row[label_column_index]
                    for idx, cell_content in enumerate(row):
                        if idx != label_column_index:
                            cells_to_translate_this_row.append(cell_content)
                else: # label column not found, or row too short for it: translate all cells
                    cells_to_translate_this_row = list(row) 
                    # original_label_value remains None, label_column_index effectively ignored for assembly

                # If all cells designated for translation are empty, skip API call
                if not any(text.strip() for text in cells_to_translate_this_row if isinstance(text, str)):
                    final_row_for_csv = list(row) # Start with original row
                    if label_column_index != -1 and original_label_value is not None and label_column_index < len(final_row_for_csv):
                        # Ensure original label is preserved, other cells are as they were (empty)
                        final_row_for_csv[label_column_index] = original_label_value 
                    # If label_column_index was -1, all cells were empty, final_row_for_csv is already correct (all original empty cells)
                    translated_rows_data.append(final_row_for_csv)
                    continue
                
                translated_cell_parts = translate_texts_groq(cells_to_translate_this_row)
                
                # Assemble the final row
                final_row_for_csv = [""] * len(header)
                current_translated_idx = 0
                has_issue_in_row = False

                if len(translated_cell_parts) != len(cells_to_translate_this_row):
                    tqdm.write(f"      Row {rows_processed_count}: Critical mismatch in translated parts count. Expected {len(cells_to_translate_this_row)}, got {len(translated_cell_parts)}. Marking row with error.")
                    final_row_for_csv = [f"ERROR_TRANSLATION_PARTS_COUNT_R{rows_processed_count}"] * len(header)
                    has_issue_in_row = True
                else: # Counts match, proceed with assembly
                    for col_idx in range(len(header)):
                        if col_idx == label_column_index:
                            final_row_for_csv[col_idx] = original_label_value if original_label_value is not None else ""
                        else:
                            if current_translated_idx < len(translated_cell_parts):
                                final_row_for_csv[col_idx] = translated_cell_parts[current_translated_idx]
                                if "ERROR" in str(translated_cell_parts[current_translated_idx]) or "FAILED" in str(translated_cell_parts[current_translated_idx]):
                                    has_issue_in_row = True
                                current_translated_idx += 1
                            else:
                                # Should not happen if counts matched and logic is correct
                                tqdm.write(f"      Row {rows_processed_count}: Logic error during row assembly for column {col_idx}. Using empty.")
                                final_row_for_csv[col_idx] = "ERROR_ASSEMBLY_LOGIC"
                                has_issue_in_row = True
                
                if has_issue_in_row:
                    rows_with_translation_issues += 1
                translated_rows_data.append(final_row_for_csv)

    except FileNotFoundError:
        print(f"  Error: Input file '{input_file_path}' not found.")
        return False
    except Exception as e:
        print(f"  An unexpected error during CSV processing for '{os.path.basename(input_file_path)}': {e}")
        try:
            with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile_err:
                writer_err = csv.writer(outfile_err)
                if header: writer_err.writerow(header)
                writer_err.writerow([f"Error during processing: {e}"])
        except Exception as write_e:
            print(f"    Additionally, failed to write error state to output file: {write_e}")
        return False

    if not translated_rows_data or len(translated_rows_data) <= 1:
        print(f"  No data rows were processed or translated for '{os.path.basename(input_file_path)}'.")
        file_processed_successfully = False 
    
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            if translated_rows_data: 
                 writer.writerows(translated_rows_data)
            else:
                 writer.writerow(["Error: No data to write or file was empty."])
        
        summary_msg = f"  Finished processing '{os.path.basename(input_file_path)}'. {rows_processed_count} data rows processed."
        if rows_with_translation_issues > 0:
            summary_msg += f" {rows_with_translation_issues} row(s) had translation issues."
        
        if file_processed_successfully and rows_processed_count > 0:
            print(summary_msg + f" Output: '{output_file_path}'")
        elif not file_processed_successfully and rows_processed_count == 0 :
             print(f"  Processing for '{os.path.basename(input_file_path)}' resulted in no data rows. Output file '{output_file_path}' created.")
        return True

    except Exception as e:
        print(f"  Error writing to output file '{output_file_path}': {e}")
        return False


def main():
    global client, API_KEY
    if not API_KEY and 'google.colab' in sys.modules:
        try:
            from google.colab import userdata
            colab_api_key = userdata.get('GROQ_API_KEY')
            if colab_api_key:
                API_KEY = colab_api_key
                os.environ['GROQ_API_KEY'] = colab_api_key
                print("Successfully loaded GROQ_API_KEY from Colab Secrets.")
                try:
                    client = Groq(api_key=API_KEY)
                except Exception as e:
                    print(f"Error re-initializing Groq client with Colab key: {e}")
                    client = None
            else:
                print("GROQ_API_KEY not found in Colab Secrets.")
        except ImportError:
            print("Could not import Colab userdata. Are you running in a Colab environment?")
        except Exception as e:
            print(f"Error accessing Colab userdata: {e}")
    
    if not client:
        print("Groq client is not initialized (API key likely missing or invalid). Aborting.")
        return

    if not os.path.exists(INPUT_DATA_FOLDER):
        print(f"Input folder '{INPUT_DATA_FOLDER}' not found. Creating it.")
        try:
            os.makedirs(INPUT_DATA_FOLDER)
            create_sample_csv(os.path.join(INPUT_DATA_FOLDER, "sample_for_translation.csv"))
            print(f"Please add your CSV files to '{INPUT_DATA_FOLDER}' and re-run.")
        except OSError as e:
            print(f"Error creating input folder '{INPUT_DATA_FOLDER}': {e}.")
        return 

    try:
        all_files_in_input_dir = os.listdir(INPUT_DATA_FOLDER)
    except OSError as e:
        print(f"Error reading input folder '{INPUT_DATA_FOLDER}': {e}.")
        return
        
    csv_files_to_process = [f for f in all_files_in_input_dir if f.lower().endswith('.csv') and os.path.isfile(os.path.join(INPUT_DATA_FOLDER, f))]

    if not csv_files_to_process:
        print(f"No CSV files found in '{INPUT_DATA_FOLDER}'.")
        if not any(f.lower().endswith('.csv') for f in all_files_in_input_dir):
            create_sample_csv(os.path.join(INPUT_DATA_FOLDER, "sample_for_translation.csv"))
        print(f"Ensure CSV files are in '{INPUT_DATA_FOLDER}' with a '.csv' extension.")
        return

    if not os.path.exists(OUTPUT_DATA_FOLDER): 
        print(f"Output folder '{OUTPUT_DATA_FOLDER}' not found. Creating it.")
        try:
            os.makedirs(OUTPUT_DATA_FOLDER)
        except OSError as e:
            print(f"Error creating output folder '{OUTPUT_DATA_FOLDER}': {e}.")
            return

    print(f"\nFound {len(csv_files_to_process)} CSV file(s) in '{INPUT_DATA_FOLDER}'. Starting processing...\n")
    
    successful_files_count = 0
    skipped_files_count = 0
    
    file_iterator = tqdm(csv_files_to_process, total=len(csv_files_to_process), desc="Processing files", unit="file")

    for file_name in file_iterator:
        file_iterator.set_description(f"Processing file: {file_name}")
        
        input_file_path = os.path.join(INPUT_DATA_FOLDER, file_name)
        base_name, ext = os.path.splitext(file_name)

        if base_name.lower().endswith("_arabic"):
            tqdm.write(f"  Skipping already translated file: {file_name}") 
            skipped_files_count += 1
            continue

        output_file_name = f"{base_name}_arabic{ext}"
        output_file_path = os.path.join(OUTPUT_DATA_FOLDER, output_file_name)

        if process_single_csv(input_file_path, output_file_path):
            successful_files_count += 1

    print("\n--- Translation Summary ---")
    actual_files_attempted_translation = len(csv_files_to_process) - skipped_files_count
    print(f"Total CSV files found: {len(csv_files_to_process)}")
    if skipped_files_count > 0:
        print(f"Files skipped (already translated): {skipped_files_count}")
    print(f"Files attempted for translation: {actual_files_attempted_translation}")
    print(f"Successfully processed and translated: {successful_files_count}")
    
    errors_or_issues = actual_files_attempted_translation - successful_files_count
    if errors_or_issues > 0:
        print(f"Files with errors or issues during processing: {errors_or_issues}")

    print(f"Translated files are located in: '{os.path.abspath(OUTPUT_DATA_FOLDER)}'")
    print("--- End of Script ---")

if __name__ == '__main__':
    main()
