import torch
import numpy as np
import os
import time  # <--- Add this import
# import google.generativeai as genai
from google.api_core import exceptions as google_exceptions # <-- Import google exceptions
import math # <-- Add math import for ceil
import re
import json # <-- Add json import for parsing error details

try:
    from groq import Groq, RateLimitError, APIError # Import specific Groq errors
except ImportError:
    Groq = None # Define Groq as None if import fails
    RateLimitError = Exception # Define dummy exceptions if Groq not installed
    APIError = Exception
# --- End Groq imports ---


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


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax


def predict_classification_causal_by_letter(model, tokenizer, input_text, labels, device, lang_alpa):
    if not labels:
        print(f"Warning: Received empty labels list for input. Skipping prediction.")
        # Return values indicating failure/skip
        return None, None

    alpa = alpa_ar
    if lang_alpa == 'en':
        alpa = alpa_en

    choices = list(alpa.values())[:len(labels)]
    choice_ids = [tokenizer.encode(choice)[-1] for choice in choices]
    with torch.no_grad():

        if model.config._name_or_path in ['core42/jais-30b-v3', 'core42/jais-30b-chat-v3', 'abdo-Mansour/jais-adapted-7b-chat-BNB-4bit']:
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        elif model.config._name_or_path in ['aubmindlab/aragpt2-mega']:
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        else:
            inputs = tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        if model.config._name_or_path in ['FreedomIntelligence/AceGPT-13B', 'FreedomIntelligence/AceGPT-7B', 'FreedomIntelligence/AceGPT-7B-chat', 'FreedomIntelligence/AceGPT-13B-chat', 'abdo-Mansour/jais-adapted-7b-chat-BNB-4bit']:
            inputs.pop("token_type_ids")
        outputs = model(**inputs, labels=input_ids)
        last_token_logits = outputs.logits[:, -1, :]
        choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
        conf = softmax(choice_logits[0])
        pred = alpa[np.argmax(choice_logits[0])]
    return conf, pred


def predict_classification_mt0_by_letter(model, tokenizer, input_text, labels, device, lang_alpa):
    alpa = alpa_ar
    if lang_alpa == 'en':
        alpa = alpa_en

    choices = list(alpa.values())[:len(labels)]
    choice_ids = [tokenizer.encode(choice)[0] for choice in choices]
    if not choice_ids:
        print(f"Warning: Generated empty choice_ids list. Skipping prediction.")
        return None, None
    with torch.no_grad():
        start_token = tokenizer('<pad>', return_tensors="pt").to(device)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model(**inputs, decoder_input_ids=start_token['input_ids'])
        last_token_logits = outputs.logits[:, -1, :]
        choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
        if not choice_ids:
            print(f"Warning: Generated empty choice_ids list. Skipping prediction.")
            return None, None
        conf = softmax(choice_logits[0])
        pred = alpa[np.argmax(choice_logits[0])]
    return conf, pred

# --- New Gemini Functionality ---
# --- Gemini Functionality Globals ---
_primary_api_key = None
_secondary_api_key = None
_active_api_key = None
_current_key_type = 'primary' # 'primary' or 'secondary'
_gemini_configured = False
_gemini_model_cache = {}
_rate_limit_consecutive_failures = 0 # Track consecutive rate limit errors across calls
_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES = 5

def configure_gemini():
    """Configures the Gemini API key if not already configured, using the active key."""
    global _gemini_configured, _primary_api_key, _secondary_api_key, _active_api_key, _current_key_type
    if not _gemini_configured:
        _primary_api_key = os.getenv("GEMINI_API_KEY")
        _secondary_api_key = os.getenv("GEMINI_API_KEY_SECONDARY") # Read secondary key

        if not _primary_api_key:
            raise ValueError("Primary Gemini API key not found. Set the GEMINI_API_KEY environment variable.")

        # Determine active key based on current state
        if _current_key_type == 'secondary' and _secondary_api_key:
            _active_api_key = _secondary_api_key
            print("Configuring Gemini API with SECONDARY key.")
        else:
            _active_api_key = _primary_api_key
            _current_key_type = 'primary' # Ensure it's primary if secondary doesn't exist or wasn't active
            print("Configuring Gemini API with PRIMARY key.")

        if not _active_api_key: # Should not happen if primary exists, but safety check
             raise ValueError("No active Gemini API key could be determined.")

        try:
            # Dynamically import genai only when needed
            global genai
            import google.generativeai as genai
            genai.configure(api_key=_active_api_key)
            _gemini_configured = True
            print(f"Gemini API configured successfully with {_current_key_type.upper()} key.")
        except ImportError:
             print("Error: google.generativeai package not found. Cannot use Gemini models.")
             _gemini_configured = False # Mark as not configured
             # Do not raise here, let the calling function handle the lack of configuration
        except Exception as e:
            print(f"Error configuring Gemini API with {_current_key_type.upper()} key: {e}")
            # Reset configured status so it retries configuration next time
            _gemini_configured = False
            raise

def switch_gemini_key():
    """Switches to the secondary API key if available and not already active."""
    global _gemini_configured, _secondary_api_key, _active_api_key, _current_key_type, _gemini_model_cache
    if _current_key_type == 'primary' and _secondary_api_key:
        print("Attempting to switch to SECONDARY Gemini API key...")
        _current_key_type = 'secondary'
        _active_api_key = _secondary_api_key
        _gemini_configured = False # Force reconfiguration on next call
        _gemini_model_cache.clear() # Clear model cache as configuration changed
        print("Switched to SECONDARY Gemini API key. Model cache cleared. Reconfiguration needed.")
        return True
    elif _current_key_type == 'secondary':
        print("Already using SECONDARY key. Cannot switch further.")
        return False
    else:
        print("SECONDARY key not available or already primary. Cannot switch.")
        return False

def get_gemini_model(model_name):
    """Initializes and caches Gemini models. Ensures API is configured."""
    global _gemini_model_cache, _gemini_configured
    # Ensure API is configured *before* checking cache or initializing
    if not _gemini_configured:
         try:
             configure_gemini()
             # Check again after attempting configuration
             if not _gemini_configured:
                 raise RuntimeError("Gemini API could not be configured (package missing or key error).")
         except (ValueError, RuntimeError) as e:
             print(f"Skipping Gemini model initialization due to configuration error: {e}")
             raise # Re-raise to signal failure to the caller

    if model_name not in _gemini_model_cache:
        try:
            # Configuration should already be done by the check above
            print(f"Initializing Gemini model: {model_name} using {_current_key_type.upper()} key.")
            # Safety settings moved inside predict function, only need model name here
            # Ensure genai is imported (it should be by configure_gemini)
            global genai
            if 'genai' not in globals():
                 import google.generativeai as genai # Import if somehow missed
            model = genai.GenerativeModel(model_name)
            _gemini_model_cache[model_name] = model
            print(f"Successfully initialized Gemini model: {model_name}")
        except Exception as e:
            print(f"Error initializing Gemini model {model_name} with {_current_key_type.upper()} key: {e}")
            # Clear cache entry if initialization failed
            if model_name in _gemini_model_cache:
                del _gemini_model_cache[model_name]
            raise # Re-raise the exception to be handled by the caller
    return _gemini_model_cache[model_name]

def predict_classification_gemini(model_name, input_text, labels, lang_alpa):
    """
    Predicts the classification using a Gemini model with robust retry logic
    and secondary key fallback on repeated rate limits.
    Retries indefinitely on rate limits (with key switching), limited retries for other errors.

    Args:
        model_name (str): The specific Gemini model to use (e.g., 'gemini-1.5-flash-latest').
        input_text (str): The formatted input prompt containing the question and choices.
        labels (list): The list of possible label strings (used to determine number of choices).
        lang_alpa (str): 'ar' or 'en' to select the alphabet for choices.

    Returns:
        tuple: (confidence, prediction) - Confidence is None for Gemini, prediction is the predicted letter or None on failure.
    """
    global _rate_limit_consecutive_failures # Use the global counter

    try:
        # Get model will handle configuration if needed
        model = get_gemini_model(model_name) # This can raise if configuration fails

        alpa = alpa_ar if lang_alpa == 'ar' else alpa_en
        choices = list(alpa.values())[:len(labels)]

        # Construct the prompt for Gemini
        if lang_alpa == 'ar':
             prompt = f"{input_text}\n\nالرجاء الإجابة فقط بحرف الخيار الصحيح الموافق للإجابة الصحيحة من الخيارات {', '.join(choices)}."
        else: # lang_alpa == 'en'
             prompt = f"{input_text}\n\nPlease answer with only the single letter corresponding to the correct option from the choices {', '.join(choices)}."

        # Ensure genai is available for types
        global genai
        if 'genai' not in globals():
             import google.generativeai as genai

        # Generation Configuration
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            max_output_tokens=5, # Limit output to a few tokens (enough for a letter)
            temperature=0.0 # For deterministic output
        )

        # Safety Settings (adjust thresholds as needed)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        max_general_retries = 3 # Max retries for non-rate-limit errors
        general_attempt = 0
        backoff_factor = 1 # Initial backoff factor for general errors

        while True: # Loop indefinitely until success or max general retries exceeded
            try:
                # Ensure model is up-to-date if key switched
                model = get_gemini_model(model_name)

                print(f"Attempting Gemini API call (Consecutive Rate Limit Failures: {_rate_limit_consecutive_failures})...")
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                # --- Process successful response ---
                _rate_limit_consecutive_failures = 0 # Reset rate limit counter on success
                try:
                    # Extract the predicted letter (handle potential errors)
                    predicted_text = response.text.strip()
                    # Basic validation: check if it's a single character and within the expected choices
                    if len(predicted_text) == 1 and predicted_text in choices:
                        print(f"Gemini Prediction Successful: {predicted_text}")
                        return None, predicted_text # Confidence is None for Gemini
                    else:
                        # Handle cases where the output is not a single valid letter
                        raise ValueError(f"Gemini returned unexpected text: '{predicted_text}'. Expected one of {choices}.")

                except ValueError as ve:
                     # Handle cases where accessing response.text fails (e.g., blocked content)
                     # or the format is wrong as caught above
                     print(f"Warning: Could not process Gemini response text or unexpected format. Response: {response}. Error: {ve}")
                     # Check for block reason - this is likely unrecoverable by retrying
                     if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                         print(f"Reason: Blocked due to {response.prompt_feedback.block_reason}. Failing permanently for this request.")
                         return None, None # Indicate permanent failure for this request
                     else:
                         # Treat other value errors (like unexpected format) as general errors
                         # Re-raise to be caught by the general Exception handler below
                         raise Exception(f"Response processing error: {ve}")

                except Exception as resp_e:
                     print(f"Error processing Gemini response: {resp_e}")
                     # Treat as a general error, raise to be caught below
                     raise Exception(f"Response processing error: {resp_e}")
                # --- End of Process successful response ---

                # Should not be reached if processing returns or raises
                break # Exit while loop on successful processing and return

            except google_exceptions.ResourceExhausted as e: # <-- Catch specific rate limit error
                _rate_limit_consecutive_failures += 1
                print(f"Gemini API rate limit exceeded (Failure {_rate_limit_consecutive_failures}/{_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES}): {e}")

                # Reset general attempt counter and backoff factor as we are specifically handling rate limits
                general_attempt = 0
                backoff_factor = 1

                if _rate_limit_consecutive_failures >= _MAX_CONSECUTIVE_RATE_LIMIT_FAILURES:
                    print(f"Reached {_MAX_CONSECUTIVE_RATE_LIMIT_FAILURES} consecutive rate limit failures.")
                    if switch_gemini_key():
                        # Key switched successfully, reset counter and retry immediately
                        _rate_limit_consecutive_failures = 0
                        print("Retrying immediately with the new key...")
                        continue # Continue the loop to retry with the new key configuration
                    else:
                        # Could not switch key (no secondary or already secondary)
                        print("Unable to switch to a secondary key. Continuing to wait and retry with current key.")
                        # Fall through to the standard wait logic below

                # Standard wait logic for rate limits (used if < 5 failures or if switch failed)
                wait_time = 60 # Default wait time (seconds)
                # Check if the error object has metadata with retry delay
                suggested_wait = None
                if hasattr(e, 'metadata') and e.metadata:
                    for item in e.metadata:
                        if item.key == 'retry-delay':
                            # Assuming the value is like '60s' or similar, parse it
                            try:
                                delay_str = item.value
                                if delay_str.endswith('s'):
                                    suggested_wait = float(delay_str[:-1])
                                else:
                                    suggested_wait = float(delay_str) # Assume seconds if no unit
                                break # Found the delay
                            except ValueError:
                                print(f"Could not parse retry-delay value: {item.value}")

                if suggested_wait is not None:
                     wait_time = max(1, math.ceil(suggested_wait)) + 1 # Add 1s buffer
                     print(f"Using suggested retry delay + 1 second: {wait_time:.0f} seconds...")
                else:
                     print(f"No specific retry delay provided by API. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                # Continue the while loop to retry

            except Exception as e:
                _rate_limit_consecutive_failures = 0 # Reset rate limit counter on any other error
                general_attempt += 1
                print(f"Gemini API call failed (general attempt {general_attempt}/{max_general_retries}): {e}")

                if general_attempt >= max_general_retries:
                    print(f"Gemini API call failed after {max_general_retries} general attempts.")
                    return None, None # Indicate failure after max general retries

                # Exponential backoff for general errors
                wait_time = backoff_factor * 2
                print(f"Retrying general error in {wait_time} seconds...")
                time.sleep(wait_time)
                backoff_factor = wait_time # Increase backoff for the next potential general error
                # Continue the while loop to retry

    except (ImportError, RuntimeError, ValueError) as e:
        # Catch errors during setup (import, configuration, key issues)
        print(f"Error during Gemini prediction setup: {e}")
        return None, None # Indicate failure
    except Exception as e:
        # Catch other unexpected issues outside the loop
        print(f"An unexpected error occurred during Gemini prediction: {e}")
        return None, None # Indicate failure

    # This part should ideally not be reached
    print("Warning: Reached end of Gemini prediction function unexpectedly.")
    return None, None

# --- End of New Gemini Functionality ---


# --- New Groq Prediction Function ---
def predict_classification_groq(client, model_name, input_text, labels, lang_alpa):
    """
    Predicts the classification using the Groq API.

    Args:
        client: Initialized Groq client.
        model_name: The Groq model ID (e.g., 'llama-3.1-8b-instant').
        input_text: The formatted prompt containing the question and options.
        labels: The list of possible answer labels (e.g., ['A', 'B', 'C', 'D']).
        lang_alpa: 'ar' or 'en' to determine expected output format.

    Returns:
        A tuple (predicted_label, raw_response_content).
        predicted_label is the single character prediction (e.g., 'A' or 'ب').
        Returns (None, None) on failure.
    """
    if client is None:
        print("Error: Groq client not initialized.")
        return None, None

    # Determine the expected answer format based on lang_alpa
    alpa = alpa_ar if lang_alpa == 'ar' else alpa_en
    expected_labels = list(alpa.values())[:len(labels)] # Use the actual labels passed
    expected_labels_str = ", ".join(f"'{label}'" for label in expected_labels)

    # Simple instruction for the system prompt
    system_prompt = f"You are an assistant helping with multiple-choice questions. Please answer the following question. Your response must be only the single character representing your choice from {expected_labels_str}. Do not include any other text, explanations, or introductory phrases."

    max_retries = 5 # Increased retries for rate limits
    attempt = 0
    backoff_time = 1 # Initial seconds for exponential backoff if no specific time given

    while attempt < max_retries:
        try:
            print(f"Attempting Groq API call (Attempt {attempt + 1}/{max_retries})...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": input_text # Assuming input_text is the full prompt
                    }
                ],
                model=model_name,
                temperature=0, # Set low for deterministic output
                max_tokens=10, # Allow a few tokens in case of slight deviation
            )

            raw_response = chat_completion.choices[0].message.content.strip()
            print(f"Groq Raw Response: '{raw_response}'")

            # --- Parse the response ---
            predicted_label = None
            expected_chars_set = set(expected_labels) # Use a set for faster lookup

            # 1. Try direct match first (cleanest case)
            if raw_response in expected_chars_set:
                predicted_label = raw_response
                print(f"Groq Parsed Prediction (direct match): '{predicted_label}'")
                return predicted_label, raw_response

            # 2. Try stripping common bracket/formatting issues and checking length 1
            # Define characters to strip more aggressively, including brackets
            strip_chars = '[]{}()\'"` *.:-' # Brackets, quotes, spaces, asterisks, dots, colons, hyphens
            # Apply stripping multiple times or use replace for nested brackets
            cleaned_response = raw_response.strip()
            # Remove brackets specifically
            cleaned_response = cleaned_response.replace('[', '').replace(']', '')
            # Strip other unwanted characters
            cleaned_response = cleaned_response.strip(strip_chars)

            # …existing code…
            if len(cleaned_response) == 1:
                 predicted_label = cleaned_response
                 # Updated print message to reflect the change
                 print(f"Groq Parsed Prediction (single char after cleaning '{raw_response}'): '{predicted_label}'")
                 return predicted_label, raw_response


            # 3. Try regex search as a fallback (might catch labels embedded in longer text)
            expected_chars_pattern = "".join(expected_labels)

            # Search for the first occurrence of an expected character
            match = re.search(f"[{re.escape(expected_chars_pattern)}]", raw_response)
            if match:
                potential_label = match.group(0)
                # Check if this is the *only* potential label character in the cleaned string
                # to avoid matching random letters in explanations.
                temp_cleaned = raw_response.strip().strip(strip_chars)
                all_expected_in_temp = re.findall(f"[{re.escape(expected_chars_pattern)}]", temp_cleaned)
                if len(all_expected_in_temp) == 1 and all_expected_in_temp[0] == potential_label:
                     predicted_label = potential_label
                     print(f"Groq Parsed Prediction (regex fallback on '{raw_response}'): '{predicted_label}'")
                     return predicted_label, raw_response
                else:
                     # If regex match is ambiguous after cleaning, prefer failing over guessing
                     print(f"Warning: Regex matched '{potential_label}' in '{raw_response}', but cleaning resulted in ambiguity ('{temp_cleaned}').")
                     # Fall through to error

            # 4. If still not found, raise the error
            print(f"Warning: Could not parse expected label from Groq response: '{raw_response}'. Expected one of {expected_labels_str}.")
            raise ValueError(f"Failed to parse expected label from response: {raw_response}")


        except RateLimitError as e:
            attempt += 1 # Count as a retry attempt
            wait_time = backoff_time # Default wait time

            # Try to parse the suggested wait time from the error message
            try:
                # The error body might be JSON parsable
                error_body = e.body
                if error_body and 'error' in error_body and 'message' in error_body['error']:
                    message = error_body['error']['message']
                    # Use regex to find "try again in Xs" or "try again in XmYs"
                    match_seconds = re.search(r"try again in ([\d.]+)\s*s", message, re.IGNORECASE)
                    match_minutes_seconds = re.search(r"try again in ([\d.]+)\s*m([\d.]+)\s*s", message, re.IGNORECASE)

                    if match_minutes_seconds:
                        minutes = float(match_minutes_seconds.group(1))
                        seconds = float(match_minutes_seconds.group(2))
                        suggested_wait = (minutes * 60) + seconds
                        wait_time = max(1, math.ceil(suggested_wait)) + 2 # Add 2s buffer
                        print(f"Groq Rate Limit Error: {e}. Using suggested wait time + 2s: {wait_time} seconds.")
                    elif match_seconds:
                        suggested_wait = float(match_seconds.group(1))
                        wait_time = max(1, math.ceil(suggested_wait)) + 2 # Add 2s buffer
                        print(f"Groq Rate Limit Error: {e}. Using suggested wait time + 2s: {wait_time} seconds.")
                    else:
                         # Fallback to exponential backoff if time not found in message
                         print(f"Groq Rate Limit Error: {e}. Could not parse suggested wait time. Retrying in {wait_time} seconds (exponential backoff)...")
                         backoff_time *= 2 # Exponential backoff only if no specific time given
                else:
                    # Fallback if error body structure is unexpected
                    print(f"Groq Rate Limit Error: {e}. Could not parse error details. Retrying in {wait_time} seconds (exponential backoff)...")
                    backoff_time *= 2
            except Exception as parse_e:
                # Fallback if any parsing error occurs
                print(f"Groq Rate Limit Error: {e}. Error parsing details ({parse_e}). Retrying in {wait_time} seconds (exponential backoff)...")
                backoff_time *= 2

            if attempt < max_retries:
                time.sleep(wait_time)
            else:
                print(f"Groq Rate Limit Error: Failed after {max_retries} attempts.")
                return None, None # Failed after retries

        except APIError as e:
            attempt += 1
            print(f"Groq API Error: {e}. Retrying in {backoff_time} seconds...")
            if attempt < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                print(f"Groq API Error: Failed after {max_retries} attempts.")
                return None, None # Failed after retries

        except ValueError as e: # Catch the parsing failure raised above
             attempt += 1
             print(f"Groq Response Parsing Error: {e}. Retrying in {backoff_time} seconds...")
             if attempt < max_retries:
                 time.sleep(backoff_time)
                 backoff_time *= 2 # Apply backoff for parsing errors too
             else:
                 print(f"Groq Response Parsing Error: Failed after {max_retries} attempts.")
                 # Return the last raw response for debugging even if parsing failed
                 # Need to get raw_response from the last failed attempt if possible, tricky scope.
                 # Let's return None, None for simplicity.
                 return None, None

        except Exception as e:
            print(f"An unexpected error occurred during Groq API call: {e}")
            # Treat unexpected errors as non-retryable for this call
            return None, None # Indicate failure

    # This point is reached if the loop completes without returning (i.e., max retries exceeded)
    print(f"Groq API call failed after {max_retries} attempts (loop ended).")
    return None, None # Failed after retries

# --- End of New Groq Prediction Function ---