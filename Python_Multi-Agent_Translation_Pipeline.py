# Using json for structured data (like input/output) and potentially parsing feedback later if needed
import json
# To interact with the file system (reading data files) and environment variables (API keys)
import os
# Import the Google Generative AI library (Gemini)
import google.generativeai as genai
# Import specific types from the Gemini library for configuration and content handling
import google.generativeai.types as genai_types
# Import specific exceptions from google.api_core to catch API errors like quota limits
from google.api_core import exceptions as google_exceptions
# Import Groq library and specific exceptions
try:
    from groq import Groq, RateLimitError as GroqRateLimitError, APIError as GroqAPIError
    groq_available = True
except ImportError:
    print("Warning: Groq library not installed. Groq API provider will not be available.")
    print("Install using: pip install groq")
    groq_available = False
    # Define dummy exceptions if groq is not available to avoid NameErrors later
    class GroqRateLimitError(Exception): pass
    class GroqAPIError(Exception): pass

# To add delays between API calls, measure time for ETA
import time
# To add randomness (jitter) to backoff delays, preventing thundering herd issues
import random
# To help find option keys using regular expressions (though not used in the final version)
import re
# For calculating time differences easily
from datetime import timedelta

# --- Configuration ---
# Select API provider: 'gemini' or 'groq'
API_PROVIDER = 'groq' # <-- CHANGE THIS TO 'groq' TO USE GROQ

# --- Constants ---
MAX_GEMINI_API_KEYS = 6 # Maximum number of Gemini API keys to check for
MAX_RETRIES_PER_KEY = 5 # Max retries on a single key before switching (for Gemini) or retrying (for Groq rate limits)
INITIAL_BACKOFF_DELAY = 10 # Initial delay in seconds for exponential backoff
BACKOFF_FACTOR = 2 # Factor to multiply delay by on each retry
ETA_UPDATE_INTERVAL_LINES = 10 # Update ETA every N lines processed
MIN_LINES_FOR_ETA = 5 # Minimum lines to process before calculating first ETA

# --- Global Variables ---
# Gemini specific
gemini_api_keys = []
gemini_key_names = []
current_gemini_key_index = 0
gemini_model_name = 'gemini-1.5-flash'

# Groq specific
groq_api_key = None
groq_client = None
# Common Groq models: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it
groq_model_name = 'meta-llama/llama-4-maverick-17b-128e-instruct'

# --- API Key Loading & Client Initialization ---
print(f"--- Selected API Provider: {API_PROVIDER} ---")

if API_PROVIDER == 'gemini':
    # --- Gemini Configuration ---
    print("--- Loading Gemini API Keys from OS Environment Variables ---")
    keys_loaded_count = 0
    for i in range(1, MAX_GEMINI_API_KEYS + 1):
        key_env_var = f"GEMINI_API_KEY_{i}"
        key = os.environ.get(key_env_var)
        if key:
            gemini_api_keys.append(key)
            gemini_key_names.append(f"Gemini_Key_{i}")
            keys_loaded_count += 1
            print(f"Found {key_env_var}.")

    if keys_loaded_count == 0:
        print("\nCRITICAL WARNING: No GEMINI_API_KEY environment variables found (checked GEMINI_API_KEY_1 to GEMINI_API_KEY_6).")
        print("Gemini API calls WILL fail if selected.")
    else:
        print(f"\nLoaded {keys_loaded_count} Gemini API key(s).")
        try:
            current_gemini_key_index = 0
            print(f"Configuring Gemini API with initial key: {gemini_key_names[current_gemini_key_index]} using model {gemini_model_name}.")
            genai.configure(api_key=gemini_api_keys[current_gemini_key_index])
            print("Gemini API configured successfully.")
        except IndexError:
             print("Error: Attempted to configure Gemini, but the api_keys list appears empty.")
        except Exception as e:
            print(f"Error configuring Gemini API with initial key {gemini_key_names[current_gemini_key_index]}: {e}")
            print("Will attempt configuration again within agent functions.")

elif API_PROVIDER == 'groq':
    # --- Groq Configuration ---
    if not groq_available:
         print("\nCRITICAL ERROR: API_PROVIDER set to 'groq', but the groq library is not installed.")
         print("Please install it (`pip install groq`) or choose the 'gemini' provider.")
         # Exit or raise error might be appropriate here depending on use case
         groq_client = None # Ensure client is None
    else:
        print("--- Loading Groq API Key from OS Environment Variable ---")
        key_env_var = "GROQ_API_KEY"
        groq_api_key = os.environ.get(key_env_var)

        if not groq_api_key:
            print(f"\nCRITICAL WARNING: {key_env_var} environment variable not set.")
            print("Groq API calls WILL fail.")
            groq_client = None
        else:
            print(f"Found {key_env_var}.")
            try:
                print(f"Initializing Groq client using model {groq_model_name}.")
                groq_client = Groq(api_key=groq_api_key)
                # Optional: Test connection with a simple call? Might consume quota/tokens.
                print("Groq client initialized.")
            except Exception as e:
                print(f"Error initializing Groq client: {e}")
                groq_client = None
else:
    print(f"\nCRITICAL ERROR: Invalid API_PROVIDER '{API_PROVIDER}'. Choose 'gemini' or 'groq'.")


# --- LLM Agent Functions (Now handle both Gemini and Groq) ---

def switch_gemini_key(agent_name: str) -> bool:
    """
    Attempts to switch to the next available Gemini API key.
    (Only relevant when API_PROVIDER is 'gemini')
    """
    global current_gemini_key_index, gemini_api_keys, gemini_key_names

    if len(gemini_api_keys) <= 1:
        print(f"{agent_name}: [Gemini] No other keys available to switch to.")
        return False

    initial_index = current_gemini_key_index
    next_key_index = (current_gemini_key_index + 1) % len(gemini_api_keys)
    print(f"{agent_name}: [Gemini] Attempting to switch from {gemini_key_names[current_gemini_key_index]} to {gemini_key_names[next_key_index]}.")

    try:
        genai.configure(api_key=gemini_api_keys[next_key_index])
        current_gemini_key_index = next_key_index
        print(f"{agent_name}: [Gemini] Successfully switched to API key: {gemini_key_names[current_gemini_key_index]}.")
        return True
    except Exception as config_e:
        print(f"{agent_name} Error: [Gemini] Failed to configure/re-init with key {gemini_key_names[next_key_index]}: {config_e}.")
        print(f"{agent_name}: [Gemini] Continuing with the current key: {gemini_key_names[initial_index]}.")
        return False


def agent1_translate_gemini(english_text: str) -> str:
    """
    Uses Agent-1 (Selected API Provider) for translation with retry logic.
    """
    global current_gemini_key_index, gemini_api_keys, gemini_key_names, gemini_model_name
    global groq_client, groq_model_name
    agent_id = "Agent 1"

    if API_PROVIDER == 'gemini':
        # --- Gemini Logic ---
        if not gemini_api_keys:
            print(f"{agent_id} Error: [Gemini] No API keys loaded.")
            return f"###ERROR: {agent_id} [Gemini] - Translation failed (No API Keys)###"

        prompt = f"""You are an AI assistant whose job is to translate some given questions from English to Arabic. While translation, there will be tags starting with ### signs (like ###Question). Do not translate these tags. Make sure you translate all the tag contents. Please follow the guidelines:
1. Maintain Meaning: Ensure the translated question conveys the original intent.
2. Cultural Adaptation: Adjust cultural references to suit the target language. For example, adapt place names, idioms, festivals, toys, objects and cultural symbols as needed (e.g., ’Louvre Museum’ should be localized appropriately).
3. Context Sensitivity: Choose translations that match the context, avoiding direct word-for-word translations that may distort meaning.
4. Natural Expression: Ensure the translation flows naturally in Arabic, preserving the readability and coherence.
5. There will be options with A,B,C or D. Do not translate the options letters and keep them in the same order.
6. Make sure all the elements are present in your response, like ###STORY, ###QUESTION and ###OPTIONS.

Input Text (English):
---
{english_text}
---

Output Translation (Arabic):"""

        retries_on_current_key = 0
        total_attempts = 0
        max_total_attempts = MAX_RETRIES_PER_KEY * len(gemini_api_keys) + len(gemini_api_keys)
        should_retry = False

        while total_attempts < max_total_attempts:
            if current_gemini_key_index >= len(gemini_key_names):
                 print(f"{agent_id} Error: [Gemini] Invalid key index.")
                 return f"###ERROR: {agent_id} [Gemini] - Internal key index error.###"
            active_key_for_attempt = gemini_key_names[current_gemini_key_index]

            try:
                genai.configure(api_key=gemini_api_keys[current_gemini_key_index])
                model = genai.GenerativeModel(gemini_model_name)
                time.sleep(0.5 + random.uniform(0, 0.5))
                request_options = {"timeout": 120}
                response = model.generate_content(prompt, request_options=request_options)

                if response.candidates and response.candidates[0].content.parts:
                    return response.text # Success
                else:
                    print(f"{agent_id} Error: [Gemini] No content/blocked (Key: {active_key_for_attempt}).")
                    finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
                    return f"###ERROR: {agent_id} [Gemini] - Translation failed (No Content/Blocked - Reason: {finish_reason}, Key: {active_key_for_attempt})###"

            except (google_exceptions.ResourceExhausted, google_exceptions.InternalServerError) as e:
                print(f"{agent_id} Caught retryable error [Gemini] ({type(e).__name__}) (Key: {active_key_for_attempt}): {e}")
                should_retry = True
            except google_exceptions.PermissionDenied as e:
                print(f"{agent_id} Error: [Gemini] Permission Denied (Key: {active_key_for_attempt}).")
                if len(gemini_api_keys) > 1:
                    if switch_gemini_key(agent_id):
                        retries_on_current_key = 0
                        total_attempts += 1
                        continue
                    else:
                        # Switch failed
                        return f"###ERROR: {agent_id} [Gemini] - Translation failed (Permission Denied on {active_key_for_attempt}, switch failed)###"
                else:
                    # Only one key, and it failed
                    return f"###ERROR: {agent_id} [Gemini] - Translation failed (Permission Denied on {active_key_for_attempt})###"
            except google_exceptions.InvalidArgument as e:
                 print(f"{agent_id} Error: [Gemini] Invalid Argument (Key: {active_key_for_attempt}): {e}")
                 return f"###ERROR: {agent_id} [Gemini] - Translation failed (Invalid Argument on {active_key_for_attempt})###"
            except Exception as e:
                error_str = str(e).lower()
                if "api_key" in error_str or "configure" in error_str:
                     print(f"{agent_id} Error: [Gemini] Config error (Key: {active_key_for_attempt}): {e}")
                     if len(gemini_api_keys) > 1:
                         if switch_gemini_key(agent_id):
                             retries_on_current_key = 0
                             total_attempts += 1
                             continue
                         else:
                             # Switch failed
                              return f"###ERROR: {agent_id} [Gemini] - Translation failed (Config error on {active_key_for_attempt}, switch failed)###"
                     else:
                          # Only one key, and it failed config
                         return f"###ERROR: {agent_id} [Gemini] - Translation failed (Config error on {active_key_for_attempt})###"
                elif "429" in error_str and ("quota" in error_str or "resource has been exhausted" in error_str):
                    print(f"{agent_id} Caught generic Quota Error [Gemini] (Key: {active_key_for_attempt}): {e}")
                    should_retry = True
                else:
                    print(f"{agent_id} Error: [Gemini] Unexpected error (Key: {active_key_for_attempt}): {type(e).__name__} - {e}")
                    return f"###ERROR: {agent_id} [Gemini] - Translation failed ({type(e).__name__} on {active_key_for_attempt})###"


            if should_retry:
                retries_on_current_key += 1
                total_attempts += 1
                if retries_on_current_key > MAX_RETRIES_PER_KEY:
                    print(f"{agent_id}: [Gemini] Max retries reached for key {active_key_for_attempt}.")
                    if len(gemini_api_keys) > 1:
                        if switch_gemini_key(agent_id):
                            retries_on_current_key = 0
                        else:
                            # Switch failed
                            return f"###ERROR: {agent_id} [Gemini] - Translation failed (Quota Exceeded on {active_key_for_attempt}, switch failed)###"
                    else:
                        # Only one key, exhausted retries
                        return f"###ERROR: {agent_id} [Gemini] - Translation failed (Quota Exceeded on {active_key_for_attempt})###"
                else:
                    # Calculate delay and wait
                    delay = INITIAL_BACKOFF_DELAY * (BACKOFF_FACTOR ** (retries_on_current_key - 1)) + random.uniform(0, 1)
                    print(f"{agent_id} Warning: [Gemini] Retrying in {delay:.2f}s (Attempt {retries_on_current_key}/{MAX_RETRIES_PER_KEY} on key {active_key_for_attempt})")
                    time.sleep(delay)
            should_retry = False # Reset for next attempt or error
        # --- End Gemini While Loop ---
        last_key_name = gemini_key_names[current_gemini_key_index] if current_gemini_key_index < len(gemini_key_names) else "Invalid Index"
        return f"###ERROR: {agent_id} [Gemini] - Translation failed (Max total attempts reached, last key: {last_key_name})###"

    elif API_PROVIDER == 'groq':
        # --- Groq Logic ---
        if not groq_client:
            print(f"{agent_id} Error: [Groq] Client not initialized.")
            return f"###ERROR: {agent_id} [Groq] - Translation failed (Client not initialized)###"

        system_prompt_content = "You are an AI assistant whose job is to translate some given questions from English to Arabic. While translation, there will be tags starting with ### signs (like ###Question). Do not translate these tags. Make sure you translate all the tag contents. Please follow the guidelines:\n1. Maintain Meaning: Ensure the translated question conveys the original intent.\n2. Cultural Adaptation: Adjust cultural references to suit the target language. For example, adapt place names, idioms, festivals, toys, objects and cultural symbols as needed (e.g., ’Louvre Museum’ should be localized appropriately).\n3. Context Sensitivity: Choose translations that match the context, avoiding direct word-for-word translations that may distort meaning.\n4. Natural Expression: Ensure the translation flows naturally in Arabic, preserving the readability and coherence.\n5. There will be options with A,B,C or D. Do not translate the options letters and keep them in the same order.\n6. Make sure all the elements are present in your response, like ###STORY, ###QUESTION and ###OPTIONS."
        user_prompt_content = f"Input Text (English):\n---\n{english_text}\n---\n\nOutput Translation (Arabic):"
        messages = [{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_prompt_content}]
        retries = 0; should_retry = False

        while retries <= MAX_RETRIES_PER_KEY:
            try:
                time.sleep(0.5 + random.uniform(0, 0.5))
                chat_completion = groq_client.chat.completions.create(messages=messages, model=groq_model_name)
                return chat_completion.choices[0].message.content # Success
            except GroqRateLimitError as e: print(f"{agent_id} Caught GroqRateLimitError: {e}"); should_retry = True
            except GroqAPIError as e: print(f"{agent_id} Error: [Groq] API Error: {type(e).__name__} - Status Code: {e.status_code} - Message: {e.message}"); return f"###ERROR: {agent_id} [Groq] - Translation failed (API Error {e.status_code})###"
            except Exception as e: print(f"{agent_id} Error: [Groq] Unexpected error: {type(e).__name__} - {e}"); return f"###ERROR: {agent_id} [Groq] - Translation failed (Unexpected {type(e).__name__})###"

            if should_retry:
                retries += 1
                if retries > MAX_RETRIES_PER_KEY: print(f"{agent_id} Error: [Groq] Max retries reached after rate limit."); return f"###ERROR: {agent_id} [Groq] - Translation failed (Rate Limit Exceeded after retries)###"
                else: delay = INITIAL_BACKOFF_DELAY * (BACKOFF_FACTOR ** (retries - 1)) + random.uniform(0, 1); print(f"{agent_id} Warning: [Groq] Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {retries}/{MAX_RETRIES_PER_KEY})"); time.sleep(delay)
            should_retry = False
        # --- End Groq While Loop ---
        return f"###ERROR: {agent_id} [Groq] - Translation failed (Exited retry loop unexpectedly)###"
    else:
        return f"###ERROR: {agent_id} - Invalid API_PROVIDER configured.###"


def agent2_validate_gemini(original_english: str, initial_arabic: str) -> str:
    """
    Uses Agent-2 (Selected API Provider) for validation with retry logic.
    """
    global current_gemini_key_index, gemini_api_keys, gemini_key_names, gemini_model_name
    global groq_client, groq_model_name
    agent_id = "Agent 2"

    if API_PROVIDER == 'gemini':
        # --- Gemini Logic ---
        if not gemini_api_keys: return f"###ERROR: {agent_id} [Gemini] - Validation failed (No API Keys)###"
        prompt = f"""You are tasked with verifying a translation from English to Arabic. You will be given the original text and the translated text. Your job is to check the following:
1. Accuracy of Meaning: Ensure that the translated text preserves the same meaning as the original. Point out any inconsistencies or loss of information.
2. Cultural Adaptation: Verify if any cultural references or context-specific terms are correctly translated, maintaining cultural appropriateness for Arabic.
3. Contextual Relevance: Check if the translation uses contextually correct words or phrases, ensuring that any ambiguities or multiple meanings are handled properly.
4. Suggestions: Provide constructive feedback on how to improve the translation, if necessary, in English.

Your goal is to ensure that the translation is accurate, natural, and culturally appropriate. Your task is to return two tags in your output:
###Quality: respond with okay if the translation looks good.
###Feedback: Only respond with feedback if the translation is not good. Put this tag before the feedback.

Example 1 (Good Translation):
###Quality: okay

Example 2 (Needs Improvement):
###Quality: not okay
###Feedback: The translation is not good. The aspect of location of the translation is missing.

Do not comment on the tags(###) inside the original or translated text provided below.

Original Text (English):
---
{original_english}
---

Translated Text (Arabic):
---
{initial_arabic}
---

Verification Output:"""
        retries_on_current_key = 0; total_attempts = 0; max_total_attempts = MAX_RETRIES_PER_KEY * len(gemini_api_keys) + len(gemini_api_keys); should_retry = False
        while total_attempts < max_total_attempts:
            if current_gemini_key_index >= len(gemini_key_names): return f"###ERROR: {agent_id} [Gemini] - Internal key index error.###"
            active_key_for_attempt = gemini_key_names[current_gemini_key_index]
            try:
                genai.configure(api_key=gemini_api_keys[current_gemini_key_index])
                model = genai.GenerativeModel(gemini_model_name)
                time.sleep(0.5 + random.uniform(0, 0.5))
                request_options = {"timeout": 120}
                response = model.generate_content(prompt, request_options=request_options)
                if response.candidates and response.candidates[0].content.parts:
                    feedback_text = response.text
                    if feedback_text and feedback_text.strip():
                        if "###Quality:" not in feedback_text: print(f"{agent_id} Warning: [Gemini] Output missing '###Quality:' tag (Key: {active_key_for_attempt}).")
                        return feedback_text
                    else: print(f"{agent_id} Error: [Gemini] Empty feedback (Key: {active_key_for_attempt}). Retrying..."); should_retry = True
                else:
                    print(f"{agent_id} Error: [Gemini] No content/blocked (Key: {active_key_for_attempt}).")
                    finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'
                    return f"###ERROR: {agent_id} [Gemini] - Validation failed (No Content/Blocked - Reason: {finish_reason}, Key: {active_key_for_attempt})###"
            except (google_exceptions.ResourceExhausted, google_exceptions.InternalServerError) as e: print(f"{agent_id} Caught retryable error [Gemini] ({type(e).__name__}) (Key: {active_key_for_attempt}): {e}"); should_retry = True
            except google_exceptions.PermissionDenied as e:
                print(f"{agent_id} Error: [Gemini] Permission Denied (Key: {active_key_for_attempt}).")
                if len(gemini_api_keys) > 1:
                    if switch_gemini_key(agent_id):
                        retries_on_current_key = 0
                        total_attempts += 1
                        continue
                    else:
                        return f"###ERROR: {agent_id} [Gemini] - Validation failed (Permission Denied on {active_key_for_attempt}, switch failed)###"
                else:
                    return f"###ERROR: {agent_id} [Gemini] - Validation failed (Permission Denied on {active_key_for_attempt})###"
            except google_exceptions.InvalidArgument as e: print(f"{agent_id} Error: [Gemini] Invalid Argument (Key: {active_key_for_attempt}): {e}"); return f"###ERROR: {agent_id} [Gemini] - Validation failed (Invalid Argument on {active_key_for_attempt})###"
            except Exception as e:
                error_str = str(e).lower()
                if "api_key" in error_str or "configure" in error_str:
                    print(f"{agent_id} Error: [Gemini] Config error (Key: {active_key_for_attempt}): {e}")
                    if len(gemini_api_keys) > 1:
                        if switch_gemini_key(agent_id):
                            retries_on_current_key = 0
                            total_attempts += 1
                            continue
                        else:
                             return f"###ERROR: {agent_id} [Gemini] - Validation failed (Config error on {active_key_for_attempt}, switch failed)###"
                    else:
                         return f"###ERROR: {agent_id} [Gemini] - Validation failed (Config error on {active_key_for_attempt})###"
                elif "429" in error_str and ("quota" in error_str or "resource has been exhausted" in error_str):
                    print(f"{agent_id} Caught generic Quota Error [Gemini] (Key: {active_key_for_attempt}): {e}")
                    should_retry = True
                else:
                    print(f"{agent_id} Error: [Gemini] Unexpected error (Key: {active_key_for_attempt}): {type(e).__name__} - {e}")
                    return f"###ERROR: {agent_id} [Gemini] - Validation failed ({type(e).__name__} on {active_key_for_attempt})###"

            if should_retry:
                retries_on_current_key += 1
                total_attempts += 1
                if retries_on_current_key > MAX_RETRIES_PER_KEY:
                    print(f"{agent_id}: [Gemini] Max retries reached for key {active_key_for_attempt}.")
                    if len(gemini_api_keys) > 1:
                        if switch_gemini_key(agent_id):
                            retries_on_current_key = 0
                        else:
                            return f"###ERROR: {agent_id} [Gemini] - Validation failed (Quota Exceeded on {active_key_for_attempt}, switch failed)###"
                    else:
                        return f"###ERROR: {agent_id} [Gemini] - Validation failed (Quota Exceeded on {active_key_for_attempt})###"
                else:
                    delay = INITIAL_BACKOFF_DELAY * (BACKOFF_FACTOR ** (retries_on_current_key - 1)) + random.uniform(0, 1)
                    print(f"{agent_id} Warning: [Gemini] Retrying in {delay:.2f}s (Attempt {retries_on_current_key}/{MAX_RETRIES_PER_KEY} on key {active_key_for_attempt})")
                    time.sleep(delay)
            should_retry = False
        # --- End Gemini While Loop ---
        last_key_name = gemini_key_names[current_gemini_key_index] if current_gemini_key_index < len(gemini_key_names) else "Invalid Index"; return f"###ERROR: {agent_id} [Gemini] - Validation failed (Max total attempts reached, last key: {last_key_name})###"

    elif API_PROVIDER == 'groq':
        # --- Groq Logic ---
        if not groq_client: return f"###ERROR: {agent_id} [Groq] - Validation failed (Client not initialized)###"
        system_prompt_content = "You are tasked with verifying a translation from English to Arabic. You will be given the original text and the translated text. Your job is to check the following:\n1. Accuracy of Meaning: Ensure that the translated text preserves the same meaning as the original. Point out any inconsistencies or loss of information.\n2. Cultural Adaptation: Verify if any cultural references or context-specific terms are correctly translated, maintaining cultural appropriateness for Arabic.\n3. Contextual Relevance: Check if the translation uses contextually correct words or phrases, ensuring that any ambiguities or multiple meanings are handled properly.\n4. Suggestions: Provide constructive feedback on how to improve the translation, if necessary, in English.\n\nYour goal is to ensure that the translation is accurate, natural, and culturally appropriate. Your task is to return two tags in your output:\n###Quality: respond with okay if the translation looks good.\n###Feedback: Only respond with feedback if the translation is not good. Put this tag before the feedback.\n\nExample 1 (Good Translation):\n###Quality: okay\n\nExample 2 (Needs Improvement):\n###Quality: not okay\n###Feedback: The translation is not good. The aspect of location of the translation is missing.\n\nDo not comment on the tags(###) inside the original or translated text provided below."
        user_prompt_content = f"Original Text (English):\n---\n{original_english}\n---\n\nTranslated Text (Arabic):\n---\n{initial_arabic}\n---\n\nVerification Output:"
        messages = [{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_prompt_content}]
        retries = 0; should_retry = False
        while retries <= MAX_RETRIES_PER_KEY:
            try:
                time.sleep(0.5 + random.uniform(0, 0.5))
                chat_completion = groq_client.chat.completions.create(messages=messages, model=groq_model_name)
                feedback_text = chat_completion.choices[0].message.content
                if feedback_text and feedback_text.strip():
                    if "###Quality:" not in feedback_text: print(f"{agent_id} Warning: [Groq] Output missing '###Quality:' tag.")
                    return feedback_text
                else: print(f"{agent_id} Error: [Groq] Received empty feedback. Retrying..."); should_retry = True
            except GroqRateLimitError as e: print(f"{agent_id} Caught GroqRateLimitError: {e}"); should_retry = True
            except GroqAPIError as e: print(f"{agent_id} Error: [Groq] API Error: {type(e).__name__} - Status Code: {e.status_code} - Message: {e.message}"); return f"###ERROR: {agent_id} [Groq] - Validation failed (API Error {e.status_code})###"
            except Exception as e: print(f"{agent_id} Error: [Groq] Unexpected error: {type(e).__name__} - {e}"); return f"###ERROR: {agent_id} [Groq] - Validation failed (Unexpected {type(e).__name__})###"
            if should_retry:
                retries += 1
                if retries > MAX_RETRIES_PER_KEY: print(f"{agent_id} Error: [Groq] Max retries reached after rate limit."); return f"###ERROR: {agent_id} [Groq] - Validation failed (Rate Limit Exceeded after retries)###"
                else: delay = INITIAL_BACKOFF_DELAY * (BACKOFF_FACTOR ** (retries - 1)) + random.uniform(0, 1); print(f"{agent_id} Warning: [Groq] Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {retries}/{MAX_RETRIES_PER_KEY})"); time.sleep(delay)
            should_retry = False
        return f"###ERROR: {agent_id} [Groq] - Validation failed (Exited retry loop unexpectedly)###"

    else: return f"###ERROR: {agent_id} - Invalid API_PROVIDER configured.###"


def agent3_refine_gemini(original_english: str, initial_arabic: str, feedback: str) -> str:
    """
    Uses Agent-3 (Selected API Provider) for refinement with retry logic.
    """
    global current_gemini_key_index, gemini_api_keys, gemini_key_names, gemini_model_name
    global groq_client, groq_model_name
    agent_id = "Agent 3"

    # --- Feedback Processing ---
    if feedback.startswith("###ERROR:"): return initial_arabic + f"\n###ERROR: {agent_id} - Skipped due to Agent 2 error ({feedback})###"
    else: feedback_prompt_part = f"Feedback from Verification Agent:\n---\n{feedback}\n---"
    # --- End Feedback Processing ---

    if API_PROVIDER == 'gemini':
        # --- Gemini Logic ---
        if not gemini_api_keys: return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Refinement failed (No API Keys)###"
        prompt = f"""You are an AI assistant who is capable of translating from English to Arabic.
You will be given an English text and an initial translation into Arabic.
However, there may be problems in the initial translation, such as missed incidents, tweaked details, or other inaccuracies.
You will also be given feedback from a verification agent. Closely follow the feedback provided (especially the text after the '###Feedback:' tag, if present) to improve the translation or insert any missing elements in the story, question, or options section.
Ensure all structural tags (###STORY, ###QUESTION, ###OPTIONS, including option markers like (A), (B)) are present, correctly formatted, and match the structure of the original English text.
Output ONLY the complete, refined Arabic text, including all necessary tags. Do not add any explanations or introductory phrases.

Original English Text:
---
{original_english}
---

Initial Arabic Translation:
---
{initial_arabic}
---

{feedback_prompt_part}

Refined Arabic Translation (including all tags):"""
        retries_on_current_key = 0; total_attempts = 0; max_total_attempts = MAX_RETRIES_PER_KEY * len(gemini_api_keys) + len(gemini_api_keys); should_retry = False
        while total_attempts < max_total_attempts:
            if current_gemini_key_index >= len(gemini_key_names): return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Internal key index error.###"
            active_key_for_attempt = gemini_key_names[current_gemini_key_index]
            try:
                genai.configure(api_key=gemini_api_keys[current_gemini_key_index])
                model = genai.GenerativeModel(gemini_model_name)
                time.sleep(0.5 + random.uniform(0, 0.5))
                request_options = {"timeout": 120}
                response = model.generate_content(prompt, request_options=request_options)
                if response.candidates and response.candidates[0].content.parts:
                    refined_arabic = response.text
                    if refined_arabic and "###STORY" in refined_arabic and "###QUESTION" in refined_arabic and len(refined_arabic) > len(initial_arabic) * 0.5 : return refined_arabic
                    else: print(f"{agent_id} Warning: [Gemini] Refined text invalid (Key: {active_key_for_attempt}). Retrying..."); should_retry = True
                else: print(f"{agent_id} Error: [Gemini] No content/blocked (Key: {active_key_for_attempt})."); finish_reason = response.candidates[0].finish_reason if response.candidates else 'UNKNOWN'; return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Refinement failed (No Content/Blocked - Reason: {finish_reason}, Key: {active_key_for_attempt})###"
            except (google_exceptions.ResourceExhausted, google_exceptions.InternalServerError) as e: print(f"{agent_id} Caught retryable error [Gemini] ({type(e).__name__}) (Key: {active_key_for_attempt}): {e}"); should_retry = True
            except google_exceptions.PermissionDenied as e:
                print(f"{agent_id} Error: [Gemini] Permission Denied (Key: {active_key_for_attempt}).")
                if len(gemini_api_keys) > 1:
                    if switch_gemini_key(agent_id):
                        retries_on_current_key = 0
                        total_attempts += 1
                        continue
                    else:
                        return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Refinement failed (Permission Denied on {active_key_for_attempt}, switch failed)###"
                else:
                    return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Refinement failed (Permission Denied on {active_key_for_attempt})###"
            except google_exceptions.InvalidArgument as e: print(f"{agent_id} Error: [Gemini] Invalid Argument (Key: {active_key_for_attempt}): {e}"); return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Refinement failed (Invalid Argument on {active_key_for_attempt})###"
            except Exception as e:
                error_str = str(e).lower()
                if "api_key" in error_str or "configure" in error_str:
                    print(f"{agent_id} Error: [Gemini] Config error (Key: {active_key_for_attempt}): {e}")
                    if len(gemini_api_keys) > 1:
                        if switch_gemini_key(agent_id):
                            retries_on_current_key = 0
                            total_attempts += 1
                            continue
                        else:
                            return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Refinement failed (Config error on {active_key_for_attempt}, switch failed)###"
                    else:
                        return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Refinement failed (Config error on {active_key_for_attempt})###"
                elif "429" in error_str and ("quota" in error_str or "resource has been exhausted" in error_str):
                    print(f"{agent_id} Caught generic Quota Error [Gemini] (Key: {active_key_for_attempt}): {e}")
                    should_retry = True
                else:
                    print(f"{agent_id} Error: [Gemini] Unexpected error (Key: {active_key_for_attempt}): {type(e).__name__} - {e}")
                    return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Refinement failed ({type(e).__name__} on {active_key_for_attempt})###"

            if should_retry:
                retries_on_current_key += 1
                total_attempts += 1
                if retries_on_current_key > MAX_RETRIES_PER_KEY:
                    print(f"{agent_id}: [Gemini] Max retries reached for key {active_key_for_attempt}.")
                    if len(gemini_api_keys) > 1:
                        if switch_gemini_key(agent_id):
                            retries_on_current_key = 0
                        else:
                            return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Refinement failed (Quota Exceeded on {active_key_for_attempt}, switch failed)###"
                    else:
                        return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Refinement failed (Quota Exceeded on {active_key_for_attempt})###"
                else:
                    delay = INITIAL_BACKOFF_DELAY * (BACKOFF_FACTOR ** (retries_on_current_key - 1)) + random.uniform(0, 1)
                    print(f"{agent_id} Warning: [Gemini] Retrying in {delay:.2f}s (Attempt {retries_on_current_key}/{MAX_RETRIES_PER_KEY} on key {active_key_for_attempt})")
                    time.sleep(delay)
            should_retry = False
        # --- End Gemini While Loop ---
        last_key_name = gemini_key_names[current_gemini_key_index] if current_gemini_key_index < len(gemini_key_names) else "Invalid Index"; return initial_arabic + f"\n###ERROR: {agent_id} [Gemini] - Refinement failed (Max total attempts reached, last key: {last_key_name})###"

    elif API_PROVIDER == 'groq':
        # --- Groq Logic ---
        if not groq_client: return initial_arabic + f"\n###ERROR: {agent_id} [Groq] - Refinement failed (Client not initialized)###"
        system_prompt_content = "You are an AI assistant who is capable of translating from English to Arabic.\nYou will be given an English text and an initial translation into Arabic.\nHowever, there may be problems in the initial translation, such as missed incidents, tweaked details, or other inaccuracies.\nYou will also be given feedback from a verification agent. Closely follow the feedback provided (especially the text after the '###Feedback:' tag, if present) to improve the translation or insert any missing elements in the story, question, or options section.\nEnsure all structural tags (###STORY, ###QUESTION, ###OPTIONS, including option markers like (A), (B)) are present, correctly formatted, and match the structure of the original English text.\nOutput ONLY the complete, refined Arabic text, including all necessary tags. Do not add any explanations or introductory phrases."
        user_prompt_content = f"Original English Text:\n---\n{original_english}\n---\n\nInitial Arabic Translation:\n---\n{initial_arabic}\n---\n\n{feedback_prompt_part}\n\nRefined Arabic Translation (including all tags):"
        messages = [{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_prompt_content}]
        retries = 0; should_retry = False
        while retries <= MAX_RETRIES_PER_KEY:
            try:
                time.sleep(0.5 + random.uniform(0, 0.5))
                chat_completion = groq_client.chat.completions.create(messages=messages, model=groq_model_name)
                refined_arabic = chat_completion.choices[0].message.content
                # --- Sanity Check ---
                if refined_arabic and "###STORY" in refined_arabic and "###QUESTION" in refined_arabic and len(refined_arabic) > len(initial_arabic) * 0.5 :
                    return refined_arabic # Success
                else:
                    # FIX: Handle invalid output - Log, return warning, DO NOT retry automatically
                    print(f"{agent_id} Warning: [Groq] Refined text invalid (missing tags or too short).")
                    print(f"--- Invalid Groq Output Start ---\n{refined_arabic[:500]}...\n--- Invalid Groq Output End ---")
                    return initial_arabic + f"\n###WARNING: {agent_id} [Groq] - Refinement produced potentially invalid output ###"
            except GroqRateLimitError as e:
                # Only retry on actual rate limit errors
                print(f"{agent_id} Caught GroqRateLimitError: {e}")
                should_retry = True
            except GroqAPIError as e: print(f"{agent_id} Error: [Groq] API Error: {type(e).__name__} - Status Code: {e.status_code} - Message: {e.message}"); return initial_arabic + f"\n###ERROR: {agent_id} [Groq] - Refinement failed (API Error {e.status_code})###"
            except Exception as e: print(f"{agent_id} Error: [Groq] Unexpected error: {type(e).__name__} - {e}"); return initial_arabic + f"\n###ERROR: {agent_id} [Groq] - Refinement failed (Unexpected {type(e).__name__})###"

            if should_retry:
                retries += 1
                if retries > MAX_RETRIES_PER_KEY:
                    print(f"{agent_id} Error: [Groq] Max retries reached after rate limit.")
                    return initial_arabic + f"\n###ERROR: {agent_id} [Groq] - Refinement failed (Rate Limit Exceeded after retries)###"
                else:
                    delay = INITIAL_BACKOFF_DELAY * (BACKOFF_FACTOR ** (retries - 1)) + random.uniform(0, 1)
                    # Clarified retry reason
                    print(f"{agent_id} Warning: [Groq] Rate limit hit. Retrying in {delay:.2f} seconds... (Attempt {retries}/{MAX_RETRIES_PER_KEY})")
                    time.sleep(delay)
            should_retry = False # Reset for next attempt
        # --- End Groq While Loop ---
        return initial_arabic + f"\n###ERROR: {agent_id} [Groq] - Refinement failed (Exited retry loop unexpectedly)###"

    else: return initial_arabic + f"\n###ERROR: {agent_id} - Invalid API_PROVIDER configured.###"


# --- Main Pipeline Function ---
# Added per-agent timing
def arabic_translation_pipeline(english_sample: dict) -> dict:
    """
    Orchestrates the multi-agent translation pipeline for a single sample using
    the updated prompts and logic. Prints time taken for each agent step.

    Args:
        english_sample: A dictionary containing 'story', 'question', and 'options'.

    Returns:
        A dictionary containing results, including 'feedback' as a raw string.
    """
    results = {}
    pipeline_step_start_time = time.time() # Start timer for this specific pipeline run

    # Prepare Input
    full_english_text = f"###STORY\n{english_sample['story']}\n\n###QUESTION\n{english_sample['question']}\n\n###OPTIONS\n{english_sample['options']}"
    results['original_english'] = full_english_text

    # --- Step 1: Initial Translation ---
    agent1_start_time = time.time()
    initial_arabic_translation = agent1_translate_gemini(full_english_text)
    agent1_end_time = time.time()
    agent1_duration = agent1_end_time - agent1_start_time
    results['initial_arabic'] = initial_arabic_translation
    print(f"    Agent 1 (Translate) completed in: {agent1_duration:.2f} seconds")
    # --- End Step 1 ---

    # --- Step 2: Validation & Feedback ---
    agent2_start_time = time.time()
    if not initial_arabic_translation.startswith("###ERROR:"):
        validation_feedback = agent2_validate_gemini(full_english_text, initial_arabic_translation)
        results['feedback'] = validation_feedback
    else:
        results['feedback'] = f"###ERROR: Skipped Agent 2 due to Agent 1 error ({initial_arabic_translation})###"
        results['refined_arabic'] = initial_arabic_translation
        agent2_end_time = time.time() # Still record time even if skipped
        agent2_duration = agent2_end_time - agent2_start_time
        print(f"    Agent 2 (Validate) skipped due to Agent 1 error ({agent2_duration:.2f} seconds)")
        pipeline_end_time = time.time()
        total_pipeline_duration = pipeline_end_time - pipeline_step_start_time
        print(f"    -- Pipeline for this line took: {total_pipeline_duration:.2f} seconds --")
        return results # Exit early

    agent2_end_time = time.time()
    agent2_duration = agent2_end_time - agent2_start_time
    print(f"    Agent 2 (Validate) completed in: {agent2_duration:.2f} seconds")
    # --- End Step 2 ---

    # --- Step 3: Refinement ---
    agent3_start_time = time.time()
    refined_arabic_translation = agent3_refine_gemini(full_english_text, initial_arabic_translation, results['feedback'])
    agent3_end_time = time.time()
    agent3_duration = agent3_end_time - agent3_start_time
    results['refined_arabic'] = refined_arabic_translation
    print(f"    Agent 3 (Refine) completed in: {agent3_duration:.2f} seconds")
    # --- End Step 3 ---

    pipeline_end_time = time.time()
    total_pipeline_duration = pipeline_end_time - pipeline_step_start_time
    print(f"    -- Pipeline for this line took: {total_pipeline_duration:.2f} seconds --")

    return results


# --- Data Loading and Processing ---
# ADDED ETA Calculation
def load_and_process_data(data_dir: str):
    """
    Loads data from .jsonl files, processes each line through the pipeline,
    collects results, and provides ETA updates.

    Args:
        data_dir: The path to the directory containing the .jsonl data files.

    Returns:
        A list of dictionaries with results for each processed line.
    """
    print(f"\n--- Loading and Processing Data from Directory: {data_dir} ---")
    processed_results = []
    file_count = 0
    total_lines_attempted = 0
    lines_successfully_parsed = 0
    lines_skipped_or_failed_parsing = 0
    pipeline_error_count = 0
    total_lines_to_process = 0 # For ETA

    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' not found or is not a directory.")
        return processed_results

    # --- First Pass: Count total lines for ETA ---
    print("--- Counting total lines for ETA estimation ---")
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.endswith(".jsonl") and os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip(): # Count only non-empty lines
                            total_lines_to_process += 1
            except Exception as e:
                print(f"Warning: Could not read file {filename} during count: {e}")
    print(f"--- Found {total_lines_to_process} non-empty lines to process across all .jsonl files ---")
    # --- End First Pass ---

    if total_lines_to_process == 0:
        print("No lines found to process. Exiting.")
        return processed_results

    # --- Second Pass: Process data ---
    start_time = time.time() # Record start time for ETA
    processed_lines_count = 0 # Track processed lines for ETA

    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)

        if filename.endswith(".jsonl") and os.path.isfile(file_path):
            file_count += 1
            print(f"\n--- Processing file: {filename} ---")
            lines_in_file = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line_num = i + 1
                        total_lines_attempted += 1
                        lines_in_file += 1
                        print(f"  Processing line {line_num}/{lines_in_file} in {filename} (Overall: {processed_lines_count+1}/{total_lines_to_process})...") # More detailed progress

                        if not line.strip():
                            # print(f"    Skipping empty line {line_num}.") # Reduced verbosity
                            lines_skipped_or_failed_parsing += 1
                            continue

                        try:
                            data = json.loads(line)
                            lines_successfully_parsed += 1

                            # --- Extract and Validate Required Data ---
                            data_upper = {k.upper(): v for k, v in data.items()}
                            story_text = data_upper.get("STORY")
                            question_text = data_upper.get("QUESTION")
                            option_keys = sorted([k for k in data if k.upper().startswith("OPTION-")])

                            missing_keys = []
                            if not story_text: missing_keys.append("STORY")
                            if not question_text: missing_keys.append("QUESTION")
                            if not option_keys: missing_keys.append("OPTION-*")

                            if missing_keys:
                                print(f"    Warning: Skipping line {line_num} in {filename} due to missing keys: {', '.join(missing_keys)}.")
                                lines_skipped_or_failed_parsing += 1
                                continue
                            # --- End Extract and Validate ---

                            # --- Prepare Options String ---
                            options_string = ""
                            option_values = [data[k] for k in option_keys]
                            if option_values:
                                options_string = "\n".join(
                                    [f"({chr(65+j)}) {opt}" for j, opt in enumerate(option_values)]
                                )
                            # --- End Prepare Options String ---

                            # --- Prepare Input for Pipeline ---
                            english_sample = {
                                "story": story_text,
                                "question": question_text,
                                "options": options_string
                            }
                            # --- End Prepare Input ---

                            # --- Run the Updated Translation Pipeline ---
                            pipeline_output = arabic_translation_pipeline(english_sample)
                            # --- End Run Pipeline ---

                            # --- Store Results ---
                            result_entry = {
                                "source_file": filename,
                                "line_number": line_num,
                                # FIX: Use raw string r"..." for the key containing \N
                                "scenario_id": data_upper.get(r"序号\NINDEX", data_upper.get("INDEX", data_upper.get("ID", f"{filename}_line{line_num}"))),
                                **pipeline_output
                            }
                            processed_results.append(result_entry)
                            processed_lines_count += 1 # Increment successfully processed count for ETA
                            # --- End Store Results ---

                            # --- Check for Pipeline Errors ---
                            if "###ERROR:" in pipeline_output.get('refined_arabic', '') or \
                               "###WARNING:" in pipeline_output.get('refined_arabic', ''):
                                # print(f"    Pipeline completed with errors/warnings for line {line_num}.") # Reduced verbosity
                                pipeline_error_count += 1
                            # else:
                                # print(f"    Successfully processed line {line_num}.") # Reduced verbosity
                            # --- End Check for Pipeline Errors ---

                            # --- ETA Calculation and Update ---
                            if processed_lines_count >= MIN_LINES_FOR_ETA and processed_lines_count % ETA_UPDATE_INTERVAL_LINES == 0:
                                current_time = time.time()
                                elapsed_time = current_time - start_time
                                avg_time_per_line = elapsed_time / processed_lines_count
                                remaining_lines = total_lines_to_process - processed_lines_count
                                if remaining_lines > 0: # Avoid division by zero or negative time if already finished
                                     eta_seconds = avg_time_per_line * remaining_lines
                                     eta_formatted = str(timedelta(seconds=int(eta_seconds))) # Format as H:MM:SS
                                else:
                                     eta_formatted = "0:00:00"
                                progress_percent = (processed_lines_count / total_lines_to_process) * 100
                                print(f"  Progress: {processed_lines_count}/{total_lines_to_process} lines ({progress_percent:.1f}%) processed. Avg time/line: {avg_time_per_line:.2f}s. ETA: {eta_formatted}")
                            # --- End ETA ---


                        except json.JSONDecodeError:
                            print(f"    Error: Could not decode JSON from line {line_num} in {filename}. Skipping.")
                            lines_skipped_or_failed_parsing += 1
                        except Exception as e:
                            print(f"    Error processing line {line_num} in {filename}: {type(e).__name__} - {e}. Skipping.")
                            lines_skipped_or_failed_parsing += 1

                # print(f"--- Finished processing file: {filename} ({lines_in_file} lines attempted) ---") # Reduced verbosity

            except FileNotFoundError:
                print(f"Error: File not found: {file_path}")
            except Exception as e:
                print(f"Error reading file {filename}: {type(e).__name__} - {e}")

        elif os.path.isdir(file_path):
             print(f"Skipping directory: {filename}")
        elif not filename.endswith(".jsonl"):
             print(f"Skipping non-JSONL file: {filename}")
    # --- End Second Pass ---

    # --- Final Summary ---
    end_time = time.time()
    total_processing_time = end_time - start_time
    total_time_formatted = str(timedelta(seconds=int(total_processing_time)))

    print(f"\n--- Finished processing all files in directory: {data_dir} ---")
    print(f"Total processing time: {total_time_formatted}")
    print(f"Total .jsonl files found: {file_count}")
    print(f"Total non-empty lines found: {total_lines_to_process}")
    # print(f"Total lines attempted: {total_lines_attempted}") # Can be slightly different if files change between passes
    print(f"Lines successfully parsed as JSON: {lines_successfully_parsed}")
    print(f"Lines skipped or failed parsing/validation: {lines_skipped_or_failed_parsing}")
    print(f"Lines processed through pipeline (input valid): {len(processed_results)}") # This is processed_lines_count
    print(f"Lines completed with pipeline errors/warnings in final output: {pipeline_error_count}")
    # --- End Final Summary ---

    return processed_results


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Pre-run Check ---
    provider_ready = False
    if API_PROVIDER == 'gemini':
        if gemini_api_keys: provider_ready = True
        else: print("\nCRITICAL ERROR: API_PROVIDER is 'gemini' but no Gemini keys were loaded.")
    elif API_PROVIDER == 'groq':
        if groq_client: provider_ready = True
        else: print("\nCRITICAL ERROR: API_PROVIDER is 'groq' but the Groq client failed to initialize (check key and library install).")
    else:
        # This case should already be caught during config, but added for safety
        print(f"\nCRITICAL ERROR: Invalid API_PROVIDER '{API_PROVIDER}'.")

    if not provider_ready:
         print("Please check API key setup (OS environment variables) and selected API_PROVIDER.")
         print("Script cannot proceed.")
    else:
        # --- Define Data and Output Paths ---
        actual_data_dir = "./to_be_translated" # MODIFY THIS if your data is elsewhere
        # Default output path for local execution
        output_file_path = f"translation_results_pipeline_local2_{API_PROVIDER}.jsonl" # Include provider in filename
        print(f"Output path set to local directory: {output_file_path}")
        # --- End Define Paths ---

        # --- Run Data Processing ---
        all_final_results = load_and_process_data(actual_data_dir)
        # --- End Run Data Processing ---

        # --- Display Example Result ---
        if all_final_results:
            print("\n--- Example Result (from first processed line) ---")
            # Avoid error if list is somehow empty after processing
            try:
                first_result = all_final_results[0]
                print(f"Source File: {first_result.get('source_file', 'N/A')}")
                print(f"Line Number: {first_result.get('line_number', 'N/A')}")
                print(f"Scenario ID: {first_result.get('scenario_id', 'N/A')}")
                print(f"\nOriginal English:\n{first_result.get('original_english', 'N/A')}")
                print(f"\nInitial Arabic (Agent 1 - {API_PROVIDER}):\n{first_result.get('initial_arabic', 'N/A')}")

                # Display raw feedback string from Agent 2
                feedback_raw = first_result.get('feedback', 'N/A')
                print(f"\nValidation Feedback (Agent 2 - {API_PROVIDER} - Raw Text):\n{feedback_raw}")

                print(f"\nRefined Arabic (Agent 3 - {API_PROVIDER} - Ready for Human Review):\n{first_result.get('refined_arabic', 'N/A')}")
            except IndexError:
                print("Processing seemed complete, but no results found in the list to display.")
        else:
            print("\nProcessing completed, but no results were generated.")
            print("Check data directory, file format (.jsonl), content (required keys), and API key setup.")
        # --- End Display Example Result ---

        # --- Save Results ---
        if all_final_results:
            print(f"\nSaving all {len(all_final_results)} results to {output_file_path}...")
            try:
                output_dir = os.path.dirname(output_file_path)
                # Ensure output directory exists if it's not the current directory
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Created output directory: {output_dir}")

                with open(output_file_path, 'w', encoding='utf-8') as f:
                    for result in all_final_results:
                        json_line = json.dumps(result, ensure_ascii=False)
                        f.write(json_line + '\n')
                print("Results saved successfully.")
            except Exception as e:
                print(f"Error saving results to file {output_file_path}: {type(e).__name__} - {e}")
        else:
            print("\nNo results to save.")
        # --- End Save Results ---

        print("\n--- Script Execution Finished ---")
        # print("Next Step Suggestion: Human review of the 'refined_arabic' outputs, paying attention to Agent 2 feedback.") # Already in summary

