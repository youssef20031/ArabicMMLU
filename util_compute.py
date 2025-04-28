import torch
import numpy as np
import os
import time  # <--- Add this import
# import google.generativeai as genai 
from google.api_core import exceptions as google_exceptions # <-- Import google exceptions
import math # <-- Add math import for ceil



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
            genai.configure(api_key=_active_api_key)
            _gemini_configured = True
            print(f"Gemini API configured successfully with {_current_key_type.upper()} key.")
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
         configure_gemini()

    if model_name not in _gemini_model_cache:
        try:
            # Configuration should already be done by the check above
            print(f"Initializing Gemini model: {model_name} using {_current_key_type.upper()} key.")
            # Safety settings moved inside predict function, only need model name here
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
        model = get_gemini_model(model_name)

        alpa = alpa_ar if lang_alpa == 'ar' else alpa_en
        choices = list(alpa.values())[:len(labels)]

        # Construct the prompt for Gemini
        if lang_alpa == 'ar':
             prompt = f"{input_text}\n\nالرجاء الإجابة فقط بحرف الخيار الصحيح الموافق للإجابة الصحيحة من الخيارات {', '.join(choices)}."
        else: # lang_alpa == 'en'
             prompt = f"{input_text}\n\nPlease answer with only the single letter corresponding to the correct option from the choices {', '.join(choices)}."

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
                if hasattr(e, 'retry_delay') and e.retry_delay and hasattr(e.retry_delay, 'total_seconds'):
                     suggested_wait = e.retry_delay.total_seconds()
                     wait_time = max(1, math.ceil(suggested_wait)) + 1
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

    except Exception as e:
        # Catch errors during model initialization or other unexpected issues outside the loop
        print(f"Error during Gemini prediction setup/initialization: {e}")
        return None, None # Indicate failure

    # This part should ideally not be reached
    print("Warning: Reached end of Gemini prediction function unexpectedly.")
    return None, None

# --- End of New Gemini Functionality ---
