import os
import csv
from groq import Groq, APIError # Import APIError for specific error handling
import time # For potential rate limiting with large files and retry logic
import sys # For checking if running in Colab
from tqdm import tqdm # For progress bars

# --- Configuration ---
INPUT_DATA_FOLDER = 'tobetranslated4'  # Folder containing input CSV files
OUTPUT_DATA_FOLDER = 'tobetranslated4' # Folder to save translated CSV files

# Ensure your GROQ_API_KEY is set as an environment variable.
# In Google Colab, you can set this in the "Secrets" tab (recommended)
API_KEY = os.environ.get("GROQ_API_KEY")

MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct" # Groq model for translation

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


def translate_single_text_groq(text_to_translate: str) -> str:
    """
    Translates a single string of English text to Arabic using the Groq API,
    with retry logic and handling for internal newlines.
    Joins multi-line API responses for a single input into one string.

    Args:
        text_to_translate (str): The English string to translate.
    Returns:
        str: The translated Arabic string or an error message.
    """
    if not client:
        return "ERROR_GROQ_CLIENT_NOT_INITIALIZED"
    
    if not text_to_translate or not text_to_translate.strip():
        return "" # Return empty if input is empty or just whitespace

    INTERNAL_NEWLINE_PLACEHOLDER = "||NEWLINE_PLACEHOLDER||"
    
    # Replace internal newlines with a placeholder
    processed_text = text_to_translate.replace('\r\n', INTERNAL_NEWLINE_PLACEHOLDER).replace('\n', INTERNAL_NEWLINE_PLACEHOLDER).strip()

    if not processed_text: # If after stripping and placeholder replacement, it's empty
        return ""

    system_prompt = (
        "You are an expert English to Arabic translator. "
        "Translate the following single English text to Arabic. "
        "Return only the Arabic translation. "
        "If the input text contains '||NEWLINE_PLACEHOLDER||', preserve it in the corresponding position in your Arabic translation. "
        "Do not add any extra text, numbers, bullet points, or explanations before or after the translation. "
        "If the input text is just a placeholder or seems untranslatable, try to return a sensible Arabic equivalent or the original placeholder. For numbers, return the Arabic numeral if possible, or the original number."
    )

    user_content = processed_text # Single text item

    current_retry = 0
    backoff_time = INITIAL_BACKOFF_SECONDS
    translated_text_or_error = f"ERROR_TRANSLATION_ATTEMPT_FAILED_FOR_TEXT"

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
                # Adjust max_tokens based on the single processed text length
                max_tokens=int(len(processed_text) * 3.5) + 250 
            )
            # The entire response content is the translation for the single input text
            response_content = chat_completion.choices[0].message.content.strip()
            
            # Restore the internal newline placeholders
            translated_text_or_error = response_content.replace(INTERNAL_NEWLINE_PLACEHOLDER, '\n')
            break # Successful translation

        except APIError as e:
            tqdm.write(f"       Groq APIError on text (attempt {current_retry + 1}/{MAX_RETRIES}): '{str(text_to_translate)[:50]}...' Status {e.status_code}, Msg: {str(e.message)[:100]}...")
            is_retryable = e.status_code in [500, 502, 503, 504, 429]
            if current_retry < MAX_RETRIES - 1 and is_retryable:
                actual_backoff_time = backoff_time
                if e.status_code == 429 and e.headers and isinstance(e.headers.get("retry-after"), (str, int)):
                    try:
                        actual_backoff_time += int(e.headers.get("retry-after"))
                    except ValueError:
                        pass
                actual_backoff_time = min(actual_backoff_time, MAX_BACKOFF_SECONDS)
                tqdm.write(f"       Retrying text in {actual_backoff_time:.2f} seconds...")
                time.sleep(actual_backoff_time)
                backoff_time = min(backoff_time * 2, MAX_BACKOFF_SECONDS)
            else: 
                translated_text_or_error = f"TRANSLATION_ERROR_API_{'MAX_RETRIES' if is_retryable else 'CLIENT_SIDE'}:{e.status_code}_FOR_TEXT"
                break 
        except Exception as e:
            tqdm.write(f"       Unexpected error on text (attempt {current_retry + 1}/{MAX_RETRIES}): '{str(text_to_translate)[:50]}...' Error: {str(e)[:100]}...")
            if current_retry < MAX_RETRIES - 1:
                tqdm.write(f"       Retrying text in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, MAX_BACKOFF_SECONDS)
            else: 
                translated_text_or_error = f"TRANSLATION_ERROR_UNEXPECTED_MAX_RETRIES_FOR_TEXT:{str(e)[:50]}"
                break
        current_retry += 1
    
    return translated_text_or_error


def create_sample_csv(file_path):
    """Creates a sample CSV file with the new structure if it doesn't exist."""
    print(f"Creating a sample CSV file for the new structure at: {file_path}")
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f_dummy:
            writer = csv.writer(f_dummy)
            writer.writerow(['question', 'option_a', 'option_b', 'option_c', 'option_d', 'answer'])
            writer.writerow(['What is the capital of France?\nIt is a famous city.', 'Berlin', 'Paris', 'Rome', 'Madrid', 'Paris'])
            writer.writerow(['Which programming language is this script written in?', 'Java', 'C++', 'Python', 'Ruby', 'Python'])
            writer.writerow(['What is 2 + 2?', '3', '4', '5', '6', '4'])
            writer.writerow(['An empty field example for a question:\n(Second line of question)', '', 'Option B here\n(with newline)', '', 'Option D here', ''])
            writer.writerow(['This is a sample question.', 'Sample Option A', 'Sample Option B', 'Sample Option C', 'Sample Option D', 'Sample Option B'])
        print(f"Sample file '{os.path.basename(file_path)}' created in '{os.path.dirname(file_path)}'.")
    except IOError as e:
        print(f"Error creating sample CSV file '{file_path}': {e}")


def process_single_csv(input_file_path, output_file_path):
    """Reads a single CSV, translates its content cell by cell, and writes to a new CSV."""
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
                    total_rows_in_file = sum(1 for _ in reader_count)
        except Exception as e_count:
            print(f"   Warning: Could not pre-count rows in {os.path.basename(input_file_path)} due to {e_count}. Progress bar for rows might be indeterminate.")
            total_rows_in_file = None

        with open(input_file_path, 'r', newline='', encoding='utf-8-sig') as infile:
            reader = csv.reader(infile)
            try:
                header = next(reader)
            except StopIteration:
                print(f"   Error: Input file '{os.path.basename(input_file_path)}' is empty or has no header.")
                with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile_err:
                    writer_err = csv.writer(outfile_err)
                    writer_err.writerow([f"Error: Source file '{os.path.basename(input_file_path)}' was empty or had no header."])
                return False

            translated_rows_data.append(header)

            row_iterator = tqdm(reader, total=total_rows_in_file, desc=f"   Translating rows in {os.path.basename(input_file_path)}", unit="row", leave=False)

            for i, row in enumerate(row_iterator):
                rows_processed_count += 1
                if len(row) != len(header):
                    tqdm.write(f"       Warning: Malformed row {rows_processed_count}. Expected {len(header)} columns, got {len(row)}: {str(row)[:100]}... Writing as is with error marker.")
                    error_row = list(row) + ["MALFORMED_ROW_STRUCTURE"] * (len(header) - len(row)) if len(row) < len(header) else list(row[:len(header)])
                    if "MALFORMED_ROW_STRUCTURE" not in error_row[-1] and len(error_row) == len(header):
                        error_row[-1] += " (POTENTIALLY_TRUNCATED_OR_EXTENDED_MALFORMED_ROW)"
                    translated_rows_data.append(error_row)
                    rows_with_translation_issues +=1
                    continue

                translated_cells_for_this_row = []
                has_issue_in_row = False
                for cell_text in row:
                    translated_cell = translate_single_text_groq(cell_text)
                    translated_cells_for_this_row.append(translated_cell)
                    if "ERROR" in translated_cell or "FAILED" in translated_cell:
                        has_issue_in_row = True # Mark if any cell in the row had a translation error
                
                if has_issue_in_row:
                    rows_with_translation_issues += 1
                translated_rows_data.append(translated_cells_for_this_row)

    except FileNotFoundError:
        print(f"   Error: Input file '{input_file_path}' not found.")
        return False
    except Exception as e:
        print(f"   An unexpected error during CSV processing for '{os.path.basename(input_file_path)}': {e}")
        try:
            with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile_err:
                writer_err = csv.writer(outfile_err)
                if header: writer_err.writerow(header)
                writer_err.writerow([f"Error during processing: {e}"])
        except Exception as write_e:
            print(f"       Additionally, failed to write error state to output file: {write_e}")
        return False

    if not translated_rows_data or len(translated_rows_data) <= 1:
        print(f"   No data rows were processed or translated for '{os.path.basename(input_file_path)}'.")
        file_processed_successfully = False

    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            if translated_rows_data:
                writer.writerows(translated_rows_data)
            else:
                writer.writerow(["Error: No data to write or file was empty."])

        summary_msg = f"   Finished processing '{os.path.basename(input_file_path)}'. {rows_processed_count} data rows processed."
        if rows_with_translation_issues > 0:
            summary_msg += f" {rows_with_translation_issues} row(s) had cells with translation issues." # Clarified message

        if file_processed_successfully and rows_processed_count > 0 :
            print(summary_msg + f" Output: '{output_file_path}'")
        elif not file_processed_successfully and rows_processed_count == 0 and header :
             print(f"   Processing for '{os.path.basename(input_file_path)}' resulted in no data rows. Output file '{output_file_path}' created (contains header).")
        return True

    except Exception as e:
        print(f"   Error writing to output file '{output_file_path}': {e}")
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
            pass # Not in Colab or userdata not available
        except Exception as e:
            print(f"Error accessing Colab userdata: {e}")

    if not client:
        print("Groq client is not initialized (API key likely missing or invalid). Aborting.")
        print("Please ensure GROQ_API_KEY is set as an environment variable or in Colab Secrets.")
        return

    if not os.path.exists(INPUT_DATA_FOLDER):
        print(f"Input folder '{INPUT_DATA_FOLDER}' not found. Creating it.")
        try:
            os.makedirs(INPUT_DATA_FOLDER)
            create_sample_csv(os.path.join(INPUT_DATA_FOLDER, "sample_quiz_for_translation.csv"))
            print(f"Please add your CSV files (with columns: question,option_a,option_b,option_c,option_d,answer) to '{INPUT_DATA_FOLDER}' and re-run.")
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
             create_sample_csv(os.path.join(INPUT_DATA_FOLDER, "sample_quiz_for_translation.csv"))
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
            tqdm.write(f"   Skipping already translated file: {file_name}")
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
        print(f"Files skipped (already marked as translated): {skipped_files_count}")
    print(f"Files attempted for translation: {actual_files_attempted_translation}")
    print(f"Successfully processed and translated (or attempted): {successful_files_count}")

    errors_or_issues = actual_files_attempted_translation - successful_files_count
    if errors_or_issues > 0:
        print(f"Files with errors or issues during processing: {errors_or_issues}")

    print(f"Translated files are located in: '{os.path.abspath(OUTPUT_DATA_FOLDER)}'")
    print("--- End of Script ---")

if __name__ == '__main__':
    main()
