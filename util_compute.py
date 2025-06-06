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
# --- Add OpenAI imports ---
try:
    from openai import OpenAI, APIError as OpenAI_APIError, RateLimitError as OpenAI_RateLimitError
except ImportError:
    OpenAI = None # Define OpenAI as None if import fails
    OpenAI_APIError = Exception
    OpenAI_RateLimitError = Exception
# --- End OpenAI imports ---


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
    # Handle offline Chain-of-Thought or Tree-of-Thought: generate full reasoning and extract final answer
    if 'Final Answer' in input_text or 'الإجابة النهائية' in input_text:
        import re
        # Generate reasoning text
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        raw_output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # Extract predicted label
        alpa = alpa_ar if lang_alpa=='ar' else alpa_en
        expected = ''.join(re.escape(l) for l in list(alpa.values())[:len(labels)])
        pat_en = rf"Final Answer:.*?\[\[?([{expected}])\]?"
        pat_ar = rf"الإجابة النهائية:.*?\[\[?([{expected}])\]?"
        match = re.search(pat_en, raw_output) or re.search(pat_ar, raw_output)
        pred = match.group(1) if match else raw_output.strip()[-1]
        return pred, raw_output

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
        # Move all input tensors to the specified device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Prepare labels on the same device
        input_ids = inputs["input_ids"]
        if model.config._name_or_path in ['FreedomIntelligence/AceGPT-13B', 'FreedomIntelligence/AceGPT-7B', 'FreedomIntelligence/AceGPT-7B-chat', 'FreedomIntelligence/AceGPT-13B-chat', 'abdo-Mansour/jais-adapted-7b-chat-BNB-4bit']:
            inputs.pop("token_type_ids", None)
        outputs = model(**inputs, labels=input_ids)
        last_token_logits = outputs.logits[:, -1, :]
        # Ensure choice_ids tensor is created on the same device as logits
        choice_ids_tensor = torch.tensor(choice_ids, device=last_token_logits.device)
        choice_logits = last_token_logits[:, choice_ids_tensor].detach().cpu().numpy()
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
        # Prepare start_token on the same device
        start_token = tokenizer('<pad>', return_tensors="pt")
        start_token = {k: v.to(device) for k, v in start_token.items()}
        # Prepare model inputs on the same device
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, decoder_input_ids=start_token['input_ids'])
        last_token_logits = outputs.logits[:, -1, :]
        # Ensure choice_ids tensor is on the same device as logits
        choice_ids_tensor = torch.tensor(choice_ids, device=last_token_logits.device)
        choice_logits = last_token_logits[:, choice_ids_tensor].detach().cpu().numpy()
        if not choice_ids: # This check might be redundant now or could be choice_ids_tensor
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
            import google.genai as genai
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
                 import google.genai as genai # Import if somehow missed
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
            import google.genai as genai

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
# --- Groq Prediction Function ---
def predict_classification_groq(client, model_name, input_text, labels, lang_alpa):
    """
    Predicts the classification using a Groq model with retry logic.

    Args:
        client (Groq): The initialized Groq client.
        model_name (str): The specific Groq model to use (e.g., 'llama3-8b-8192').
        input_text (str): The formatted input prompt containing the question and choices.
        labels (list): The list of possible label strings (used to determine number of choices).
        lang_alpa (str): 'ar' or 'en' to select the alphabet for choices.

    Returns:
        tuple: (prediction, raw_response) - prediction is the predicted letter ('A', 'B', 'أ', 'ب', etc.) or None on failure.
                                            raw_response is the full text output from the API.
    """
    if not client:
        print("Error: Groq client is not initialized.")
        return None, None
    if not labels:
        print(f"Warning: Received empty labels list for input. Skipping prediction.")
        return None, None

    alpa = alpa_ar if lang_alpa == 'ar' else alpa_en
    expected_labels = list(alpa.values())[:len(labels)]
    expected_labels_str = ", ".join(f"'{label}'" for label in expected_labels)
    expected_chars_set = set(expected_labels) # Use a set for faster lookup

    # System prompt instructing the model on the desired output format
    # Note: For CoT/ToT, the user prompt overrides this, but it's good practice for zero-shot.
    system_prompt = f"You are an assistant helping with multiple-choice questions. Please answer the following question. Your response must be only the single character representing your choice from {expected_labels_str}. Do not include any other text, explanations, or introductory phrases."

    max_retries = 5
    attempt = 0
    backoff_time = 1 # Initial backoff time in seconds

    while attempt < max_retries:
        try:
            print(f"Attempting Groq API call (Attempt {attempt + 1}/{max_retries})...")
            chat_completion = client.chat.completions.create(
                messages=[
                    # System prompt might be less effective if user prompt is very detailed (like CoT/ToT)
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                model=model_name,
                temperature=0, # For deterministic output
                max_tokens=8192 # Increased max_tokens for potentially verbose CoT/ToT reasoning
                # top_p=1, # Default is usually 1
                # stop=None, # Default is usually fine
                # stream=False # Default is False
            )

            raw_response = chat_completion.choices[0].message.content.strip()
            print(f"Groq Raw Response: '{raw_response}'")

            # --- Parse the response ---
            predicted_label = None

            # 1. Try regex for "Final Answer: ... is X" pattern first (MODIFIED REGEX)
            # Build character class dynamically from expected labels
            expected_chars_pattern_str = "".join(re.escape(l) for l in expected_labels)

            # Improved English pattern: look for 'Final Answer' then 'is' or ':' and optional [[ ]]
            final_answer_pattern = rf"(?:##\s*)?Final Answer.*?(?:is|:)\s*(?:\[\[)?\s*([{expected_chars_pattern_str}])\s*(?:\]\])?"

            # Improved Arabic pattern: look for 'الإجابة النهائية' then 'هي' and optional [[ ]]
            final_answer_pattern_ar = rf"(?:##\s*)?الإجابة النهائية.*?هي\s*(?:\[\[)?\s*([{expected_chars_pattern_str}])\s*(?:\]\])?"

            # Try matching English then Arabic patterns
            match = re.search(final_answer_pattern, raw_response, re.IGNORECASE | re.DOTALL)
            if not match:
                match = re.search(final_answer_pattern_ar, raw_response, re.IGNORECASE | re.DOTALL)

            if match:
                predicted_label = match.group(1).strip() # Get the captured character
                # Validate if the extracted label is actually one of the expected ones
                if predicted_label in expected_chars_set:
                    print(f"Groq Parsed Prediction (CoT/ToT pattern): '{predicted_label}'")
                    return predicted_label, raw_response
                else:
                    print(f"Warning: Regex matched '{predicted_label}', but it's not in expected labels {expected_labels_str}.")
                    predicted_label = None # Reset if invalid match

            # 2. Try direct match (for zero-shot or if model behaves perfectly)
            if predicted_label is None and raw_response in expected_chars_set:
                predicted_label = raw_response
                print(f"Groq Parsed Prediction (direct match): '{predicted_label}'")
                return predicted_label, raw_response

            # 3. Try stripping common bracket/formatting issues and checking length 1
            if predicted_label is None:
                # More aggressive cleaning: remove markdown bold/italics, brackets, then strip
                cleaned_response = re.sub(r'[*_`]', '', raw_response) # Remove markdown chars
                cleaned_response = cleaned_response.replace('[', '').replace(']', '')
                strip_chars = '{}()\'".:- ' # Added dot, quote, double quote
                cleaned_response = cleaned_response.strip(strip_chars)

                if len(cleaned_response) == 1 and cleaned_response in expected_chars_set:
                    predicted_label = cleaned_response
                    print(f"Groq Parsed Prediction (single char after cleaning '{raw_response[:50]}...'): '{predicted_label}'")
                    return predicted_label, raw_response

            # 4. Try simple regex search for the character itself at the end as a last resort
            if predicted_label is None:
                # Look for the character potentially at the end of the string, possibly after spaces/newlines/dots/stars
                match = re.search(r"([" + expected_chars_pattern_str + r"])[.\s*]*$", raw_response)
                if match:
                    potential_label = match.group(1)
                    predicted_label = potential_label
                    print(f"Groq Parsed Prediction (regex fallback - last char match on '{raw_response[:50]}...'): '{predicted_label}'")
                    return predicted_label, raw_response

            # New fallback: match single label char at start before punctuation or dash
            if predicted_label is None:
                match = re.match(rf"^\s*([{expected_chars_pattern_str}])[\s\-\:\.\)\]\[]", raw_response)
                if match:
                    predicted_label = match.group(1)
                    print(f"Groq Parsed Prediction (prefix char match on '{raw_response[:50]}...'): '{predicted_label}'")
                    return predicted_label, raw_response

            # 5. If still not found after all attempts
            if predicted_label is None:
                print(f"Warning: Could not parse expected label from Groq response. Expected one of {expected_labels_str} or CoT/ToT pattern.")
                return None, raw_response

            break # Exit loop if successful

        except RateLimitError as e:
            attempt += 1
            print(f"Groq API rate limit exceeded (Attempt {attempt}/{max_retries}): {e}. Retrying in {backoff_time} seconds...")
            if attempt < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2 # Exponential backoff
            else:
                print(f"Groq API call failed after {max_retries} rate limit attempts.")
                return None, None # Failed after retries

        except APIError as e:
            attempt += 1
            print(f"Groq API error (Attempt {attempt}/{max_retries}): {e}. Retrying in {backoff_time} seconds...")
            if attempt < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2 # Exponential backoff
            else:
                print(f"Groq API call failed after {max_retries} API error attempts.")
                return None, None # Failed after retries

        except Exception as e:
            attempt += 1
            print(f"Groq API call failed (Attempt {attempt}/{max_retries}) with unexpected error: {e}")
            # Optionally log the full traceback here for debugging
            # import traceback
            # traceback.print_exc()
            if attempt < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                print(f"Groq API call failed after {max_retries} attempts due to unexpected errors.")
                return None, None # Failed after retries

    # If loop finishes without success (e.g., max retries on errors)
    print(f"Groq API call failed after {max_retries} attempts (loop ended).")
    return None, None # Indicate failure


# --- End of New Groq Prediction Function ---

# --- New OpenAI Prediction Function ---
def predict_classification_openai(client, model_name, input_text, labels, lang_alpa):
    """
    Predicts the classification using an OpenAI model with retry logic.

    Args:
        client (OpenAI): The initialized OpenAI client.
        model_name (str): The specific OpenAI model to use (e.g., 'gpt-3.5-turbo').
        input_text (str): The formatted input prompt containing the question and choices.
        labels (list): The list of possible label strings (used to determine number of choices).
        lang_alpa (str): 'ar' or 'en' to select the alphabet for choices.

    Returns:
        tuple: (prediction, raw_response) - prediction is the predicted letter or None on failure.
                                            raw_response is the full text output from the API.
    """
    if not client:
        print("Error: OpenAI client is not initialized.")
        return None, None
    if not labels:
        print(f"Warning: Received empty labels list for OpenAI input. Skipping prediction.")
        return None, None

    alpa = alpa_ar if lang_alpa == 'ar' else alpa_en
    expected_labels = list(alpa.values())[:len(labels)]
    expected_labels_str = ", ".join(f"'{label}'" for label in expected_labels)
    expected_chars_set = set(expected_labels)

    # System prompt can be helpful for OpenAI models
    system_prompt_content = f"You are an assistant helping with multiple-choice questions. Please answer the following question. Your response must be only the single character representing your choice from {expected_labels_str}. Do not include any other text, explanations, or introductory phrases."
    if "cot_" in input_text.lower() or "_tot" in input_text.lower() or "final answer:" in input_text.lower() or "الإجابة النهائية:" in input_text.lower() : # Heuristic to detect CoT/ToT
        # For CoT/ToT, the main instruction is in the user prompt, so a simpler system prompt might be better
        # or let the user prompt fully guide. For now, we'll keep the detailed one as the user prompt
        # itself contains the "Final Answer: ..." instruction.
        pass # Keep system_prompt_content as is, or make it more generic if needed for CoT/ToT

    messages = [
        {"role": "system", "content": system_prompt_content},
        {"role": "user", "content": input_text}
    ]

    max_retries = 5
    attempt = 0
    backoff_time = 1  # Initial backoff time in seconds

    while attempt < max_retries:
        try:
            print(f"Attempting OpenAI API call (Attempt {attempt + 1}/{max_retries}, Model: {model_name})...")
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,  # For deterministic output
                max_tokens=2048 if "cot_" in input_text.lower() or "_tot" in input_text.lower() else 500, # More tokens for CoT/ToT reasoning, less for direct answer
                # top_p=1,
                # stop=None,
            )

            raw_response = chat_completion.choices[0].message.content.strip()
            print(f"OpenAI Raw Response: '{raw_response}'")

            predicted_label = None

            # 1. Try regex for "Final Answer: ... is X" or "الإجابة النهائية: ... هي X"
            expected_chars_pattern_str = "".join(re.escape(l) for l in expected_labels)
            final_answer_pattern_en = rf"Final Answer:\s*The final answer is\s*(?:\[\[)?([{expected_chars_pattern_str}])(?:\]\])?"
            final_answer_pattern_ar = rf"الإجابة النهائية:\s*الإجابة النهائية هي\s*(?:\[\[)?([{expected_chars_pattern_str}])(?:\]\])?"

            match = re.search(final_answer_pattern_en, raw_response, re.IGNORECASE | re.DOTALL)
            if not match:
                match = re.search(final_answer_pattern_ar, raw_response, re.DOTALL)

            if match:
                predicted_label = match.group(1).strip()
                if predicted_label in expected_chars_set:
                    print(f"OpenAI Parsed Prediction (CoT/ToT pattern): '{predicted_label}'")
                    return predicted_label, raw_response
                else:
                    print(f"Warning: OpenAI CoT/ToT regex matched '{predicted_label}', but it's not in expected labels {expected_labels_str}.")
                    predicted_label = None

            # 2. Try direct match (for zero-shot or if model behaves perfectly)
            if predicted_label is None and raw_response in expected_chars_set:
                predicted_label = raw_response
                print(f"OpenAI Parsed Prediction (direct match): '{predicted_label}'")
                return predicted_label, raw_response

            # 3. Try stripping common formatting issues and checking length 1
            if predicted_label is None:
                cleaned_response = re.sub(r'[*_`]', '', raw_response)
                cleaned_response = cleaned_response.replace('[', '').replace(']', '')
                strip_chars = '{}()\'".:- '
                cleaned_response = cleaned_response.strip(strip_chars)

                if len(cleaned_response) == 1 and cleaned_response in expected_chars_set:
                    predicted_label = cleaned_response
                    print(f"OpenAI Parsed Prediction (single char after cleaning '{raw_response[:50]}...'): '{predicted_label}'")
                    return predicted_label, raw_response
            
            # 4. Try simple regex search for the character itself at the end as a last resort
            if predicted_label is None:
                match = re.search(r"([" + expected_chars_pattern_str + r"])[.\s*]*$", raw_response)
                if match:
                    potential_label = match.group(1)
                    if potential_label in expected_chars_set:
                         predicted_label = potential_label
                         print(f"OpenAI Parsed Prediction (regex fallback - last char match on '{raw_response[:50]}...'): '{predicted_label}'")
                         return predicted_label, raw_response

            if predicted_label is None:
                print(f"Warning: Could not parse expected label from OpenAI response. Expected one of {expected_labels_str} or CoT/ToT pattern. Raw: '{raw_response}'")
                # For OpenAI, if parsing fails, we might still want to return the raw response for manual inspection
                # but for automated metrics, this will count as incorrect.
                return None, raw_response # Return None for pred if parsing fails

            # Should be unreachable if parsing logic is correct and returns
            break 

        except OpenAI_RateLimitError as e:
            attempt += 1
            print(f"OpenAI API rate limit exceeded (Attempt {attempt}/{max_retries}): {e}. Retrying in {backoff_time} seconds...")
            if attempt < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                print(f"OpenAI API call failed after {max_retries} rate limit attempts.")
                return None, f"RateLimitError: {e}"

        except OpenAI_APIError as e: # Catches other API errors (e.g., server errors, bad requests if not caught by validation)
            attempt += 1
            print(f"OpenAI API error (Attempt {attempt}/{max_retries}): {e}. Retrying in {backoff_time} seconds...")
            if attempt < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                print(f"OpenAI API call failed after {max_retries} API error attempts.")
                return None, f"APIError: {e}"
        
        except Exception as e:
            attempt += 1
            print(f"OpenAI API call failed (Attempt {attempt}/{max_retries}) with unexpected error: {e}")
            if attempt < max_retries:
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                print(f"OpenAI API call failed after {max_retries} attempts due to unexpected errors.")
                return None, f"UnexpectedError: {e}"

    print(f"OpenAI API call failed after {max_retries} attempts (loop ended).")
    return None, "Max retries reached"
# --- End of New OpenAI Prediction Function ---