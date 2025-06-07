# trans_gen_ai.py
from google import genai
import time
from google.genai import types # For HarmCategory, HarmBlockThreshold, GenerationConfig, exceptions
import os
import json
import traceback
from typing import List, Dict, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

_GOOGLE_AUTH_AVAILABLE = False
try:
    from google.oauth2 import service_account
    import google.auth # To catch general auth errors
    _GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    pass # Error will be handled during initialization if SA method is chosen

from .base import BaseTranslator, register_translator
# Assuming self.logger is provided by BaseModule

class GenAIAPIError(Exception):
    """Custom exception for Google GenAI API errors."""
    pass

@register_translator('GoogleGenAI')
class GenAITranslator(BaseTranslator):
    concate_text = False # Translates text line by line or block by block
    cht_require_convert = True # Assumes conversion for Traditional Chinese might be needed post-translation

    params: Dict = {
        'auth_method': {
            'type': 'selector',
            'options': ['API Key', 'Service Account JSON'],
            'value': 'API Key',
            'description': 'Authentication method for Google GenAI. Select "API Key" to use an API key, or "Service Account JSON" to use a GCP service account file.',
        },
        'api_key': {
            'value': '',
            'description': 'Your Google AI API key for Gemini models. Used if "Auth Method" is "API Key". If left empty and "API Key" method is chosen, Application Default Credentials (ADC) will be attempted (requires GOOGLE_APPLICATION_CREDENTIALS environment variable to be set or running in a configured Google Cloud environment).',
        },
        'service_account_json_path': {
            'type': 'file_path_selector', # Assumes UI provides a file browser
            'value': '',
            'description': 'Path to your GCP Service Account JSON key file. Used if "Auth Method" is "Service Account JSON".',
        },
        'gcp_project_id_display': {
            'type': 'text_readonly', # UI should render this as read-only
            'value': 'N/A',
            'description': 'Detected GCP Project ID from the Service Account file (Read-only). Updates when a valid SA file is configured.',
        },
        'model_name': {
            'type': 'selector',
            'options': [
                'gemini-2.0-flash',
                'gemini-2.5-flash-preview-05-20',
                'gemini-2.5-pro-exp-03-25', # Included for broader compatibility
            ],
            'value': 'gemini-1.5-flash-latest',
            'description': 'Select the Gemini model to use for translation.',
        },
        'override_model_name': {
            'value': '',
            'description': 'Specify a custom model name to override the selection (e.g., fine-tuned models).',
        },
        'system_prompt_template': {
            'type': 'editor',
            'value': 'You are a highly skilled translator specializing in manga and colloquial expressions. Translate the given text from {from_lang} to {to_lang}. Preserve the original meaning, tone, and nuances. If the text is already in {to_lang} or appears to be gibberish, output it as is, without any additional commentary or explanation.',
        },
        'user_prompt_template': {
            'type': 'editor',
            'value': 'Translate the following text:\n\n"{text_to_translate}"',
        },
        'max_output_tokens': {
            'value': 2048,
            'description': 'Maximum number of tokens to generate in the translation response.',
        },
        'temperature': {
            'value': 0.5, # Adjusted for more faithful translation
            'description': 'Controls randomness. Lower values (e.g., 0.2-0.5) are often better for translation.',
        },
        'top_p': {
            'value': 1.0,
            'description': 'Nucleus sampling parameter. Not always used if temperature is low.',
        },
        'top_k': {
            'value': 40,
            'description': 'Top-k sampling parameter. Limits the selection of tokens.',
        },
        'retry_attempts': {
            'value': 3,
            'description': 'Number of retry attempts if an API call fails.',
        },
        'retry_timeout': {
            'value': 10, # Seconds
            'description': 'Time in seconds to wait between retry attempts.',
        },
        'delay': {
            'value': 1.0, # Seconds (Gemini free tier is often 60 RPM)
            'description': 'Minimum delay in seconds between consecutive API requests to manage rate limits.',
        },
        'max_requests_per_minute': {
            'value': 0, # 0 means disabled
            'description': 'Maximum requests per minute (RPM). 0 to disable. This works in conjunction with the "Delay" parameter; the more restrictive limit applies.',
        },
        'max_parallel_requests': {
            'value': 5, # Sensible default
            'description': 'Maximum number of parallel requests to the API. Higher values can speed up translation but may hit rate limits faster. Set to 1 for sequential processing.',
        },
        'proxy': { # Note: google-generativeai SDK primarily uses env vars for proxy
            'value': '',
            'description': 'Proxy address (e.g., http://user:pass@host:port). The SDK usually respects HTTP_PROXY/HTTPS_PROXY environment variables.',
        },
        'gcp_location': {
            'type': 'text',
            'value': '',
            'description': 'GCP Location (e.g., us-central1). Note: This is not directly used by the `genai.Client` for standard Gemini API calls but is provided for completeness or future Vertex AI integration.',
        },
        'safety_settings': {
            'type': 'editor',
            'value': """HARASSMENT:BLOCK_NONE
HATE_SPEECH:BLOCK_NONE
SEXUALLY_EXPLICIT:BLOCK_NONE
DANGEROUS_CONTENT:BLOCK_NONE""",
            'description': 'Configure safety settings for content generation. Format: CATEGORY:THRESHOLD (e.g., HARASSMENT:BLOCK_NONE). Valid thresholds: BLOCK_NONE, BLOCK_ONLY_HIGH, BLOCK_MEDIUM_AND_ABOVE, BLOCK_LOW_AND_ABOVE.'
        }
    }

    def _setup_translator(self):
        # BaseTranslator initializes self.lang_map from LANGMAP_GLOBAL
        # We update it with specific English names for prompts
        self.lang_map.update({
            '简体中文': 'Simplified Chinese',
            '繁體中文': 'Traditional Chinese',
            '日本語': 'Japanese',
            'English': 'English',
            '한국어': 'Korean',
            'Tiếng Việt': 'Vietnamese',
            'Français': 'French',
            'Deutsch': 'German',
            'Español': 'Spanish',
            'Italiano': 'Italian',
            'Português': 'Portuguese',
            'русский язык': 'Russian',
            'čeština': 'Czech',
            'magyar nyelv': 'Hungarian',
            'Polski': 'Polish',
            'limba română': 'Romanian',
            'Türk dili': 'Turkish',
            'украї́нська мо́ва': 'Ukrainian',
            'Thai': 'Thai',
            'Arabic': 'Arabic',
            'Malayalam': 'Malayalam',
            'Tamil': 'Tamil',
            'Hindi': 'Hindi',
        })
        self.client = None
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.executor = None # Will be initialized later
        self._initialize_client()

    def _initialize_client(self):
        auth_method = self._get_param_value('auth_method', 'API Key')
        api_key = self._get_param_value('api_key')
        sa_json_path = self._get_param_value('service_account_json_path')
        proxy_str = self._get_param_value('proxy')
        self.client = None # Ensure client is reset

        # Initialize or re-initialize ThreadPoolExecutor
        max_workers = int(self._get_param_value('max_parallel_requests', 5))
        if self.executor:
            self.executor.shutdown(wait=False) # Shutdown existing executor
        self.executor = ThreadPoolExecutor(max_workers=max_workers if max_workers > 0 else None)
        self.logger.info(f"ThreadPoolExecutor initialized with max_workers={max_workers if max_workers > 0 else 'default'}")
        # Reset any previous global genai configuration
        # This is important because genai.configure is global.
        try:
            genai.configure(credentials=None, api_key=None)
        except Exception as e:
            self.logger.warning(f"Could not reset genai global configuration: {e}")

        try:
            if proxy_str:
                self.logger.info(f"Proxy '{proxy_str}' is set. Ensure HTTP_PROXY/HTTPS_PROXY environment variables are configured if SDK does not pick this up directly.")

            if auth_method == 'Service Account JSON':
                if not _GOOGLE_AUTH_AVAILABLE:
                    self.logger.error("Python package 'google-auth' is not installed. Please install it to use Service Account authentication (e.g., pip install google-auth).")
                    self.params['gcp_project_id_display']['value'] = "google-auth missing"
                    return
                if sa_json_path and os.path.exists(sa_json_path):
                    project_id = self._parse_project_id_from_sa(sa_json_path)
                    if project_id:
                        self.params['gcp_project_id_display']['value'] = project_id
                        self.logger.info(f"Attempting to initialize with Service Account: {sa_json_path}, Project ID: {project_id}")
                    else:
                        self.params['gcp_project_id_display']['value'] = "Parse Error"
                        self.logger.warning(f"Could not parse project_id from SA file: {sa_json_path}. Proceeding with SA file for auth.")

                    credentials = service_account.Credentials.from_service_account_file(sa_json_path)
                    genai.configure(credentials=credentials)
                    self.client = genai.Client()
                    self.logger.info("Google GenAI client initialized using Service Account JSON.")
                elif sa_json_path: # Path provided but does not exist
                    self.logger.error(f"Service Account JSON file not found: {sa_json_path}")
                    self.params['gcp_project_id_display']['value'] = "File Not Found"
                else: # No path provided
                    self.logger.error("Auth method is 'Service Account JSON' but no path is provided.")
                    self.params['gcp_project_id_display']['value'] = "Path Not Set"

            elif auth_method == 'API Key':
                self.params['gcp_project_id_display']['value'] = 'N/A (API Key Auth)'
                if api_key:
                    self.logger.info("Initializing Google GenAI client with API key.")
                    genai.configure(api_key=api_key)
                    self.client = genai.Client()
                else:
                    self.logger.info("API key not provided for 'API Key' auth. Attempting Application Default Credentials (ADC).")
                    self.logger.info("Ensure GOOGLE_APPLICATION_CREDENTIALS env var is set or running in a configured Google Cloud environment.")
                    self.client = genai.Client() # ADC will be used
            else:
                self.logger.error(f"Unknown authentication method: {auth_method}")
                self.params['gcp_project_id_display']['value'] = "Invalid Auth Method"

            if self.client:
                self.parsed_safety_settings = self._parse_safety_settings()
                self.logger.info(f"Google GenAI client initialized successfully. Auth Method: {auth_method}. Safety Settings: {self.parsed_safety_settings or 'SDK defaults'}")
            else:
                # Ensure parsed_safety_settings is initialized even on client init failure,
                # as _parse_safety_settings doesn't depend on the client.
                self.parsed_safety_settings = self._parse_safety_settings()
                self.logger.error("Google GenAI client initialization failed.")

        except Exception as e:
            self.logger.error(f"Failed to initialize Google GenAI client: {e}")
            if auth_method == 'API Key' and not api_key: # ADC attempt failed
                if isinstance(e, google.auth.exceptions.DefaultCredentialsError) or \
                   "Could not find Application Default Credentials" in str(e) or \
                   "default credentials" in str(e).lower():
                    self.logger.error("ADC failure: Ensure GOOGLE_APPLICATION_CREDENTIALS is set correctly, or the environment (e.g., GCE, GKE, Cloud Functions) has a service account with necessary permissions.")
            elif auth_method == 'Service Account JSON' and 'gcp_project_id_display' in self.params:
                 self.params['gcp_project_id_display']['value'] = "Init Error"

            self.logger.debug(traceback.format_exc())
            self.client = None
            self.parsed_safety_settings = self._parse_safety_settings() # Still parse safety settings

    def _parse_project_id_from_sa(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sa_info = json.load(f)
            project_id = sa_info.get('project_id')
            if not project_id:
                self.logger.warning(f"'project_id' not found in service account file: {file_path}")
                return None
            return project_id
        except FileNotFoundError:
            self.logger.error(f"Service account file not found during parse: {file_path}")
            return None
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding JSON from service account file: {file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading/parsing service account file {file_path}: {e}")
            return None

    def _parse_safety_settings(self) -> Optional[Dict[types.HarmCategory, types.HarmBlockThreshold]]:
        settings_str = self._get_param_value('safety_settings', '').strip()
        if not settings_str:
            return None # Use SDK defaults
            
        parsed_settings = {}
        for line in settings_str.splitlines():
            line = line.strip()
            if not line or ':' not in line or line.startswith('#'): # Allow comments
                continue
            
            try:
                category_str, threshold_str = map(str.strip, line.split(':', 1))
                category_key = category_str.upper()
                threshold_key = threshold_str.upper()
                
                harm_category_member = getattr(types.HarmCategory, category_key, None)
                if harm_category_member is None: 
                    harm_category_member = getattr(types.HarmCategory, "HARM_CATEGORY_" + category_key, None)

                harm_block_threshold_member = getattr(types.HarmBlockThreshold, threshold_key, None)

                if harm_category_member and harm_block_threshold_member:
                    parsed_settings[harm_category_member] = harm_block_threshold_member
                else:
                    self.logger.warning(f"Invalid safety setting line: '{line}'. Category or Threshold not found. Skipping.")
            except Exception as e:
                self.logger.warning(f"Error parsing safety setting line '{line}': {e}. Skipping.")
        
        return parsed_settings if parsed_settings else None

    def _respect_delay(self):
        # RPM Limit Logic
        max_rpm = int(self._get_param_value('max_requests_per_minute', 0))
        if max_rpm > 0:
            current_time_for_rpm = time.time()
            # Check if a minute has passed since minute_start_time
            if current_time_for_rpm - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time_for_rpm # Reset to current time

            # Check if RPM limit is reached
            if self.request_count_minute >= max_rpm:
                # Calculate time until the current 60-second window ends
                wait_time = (self.minute_start_time + 60) - current_time_for_rpm
                if wait_time > 0:
                    self.logger.warning(
                        f"Global RPM limit ({max_rpm}) reached. Waiting {wait_time:.2f} seconds."
                    )
                    time.sleep(wait_time)
                # After waiting, the new minute effectively starts
                self.request_count_minute = 0
                self.minute_start_time = time.time() # Reset to current time after sleep

        # Fixed Delay Logic (runs after potential RPM sleep)
        current_time_for_fixed_delay = time.time()
        time_since_last_request = current_time_for_fixed_delay - self.last_request_time
        required_delay = float(self._get_param_value('delay', 1.0))
        if time_since_last_request < required_delay:
            sleep_time = required_delay - time_since_last_request
            if sleep_time > 0: # Ensure sleep_time is positive
                self.logger.debug(f"Waiting {sleep_time:.2f} seconds (fixed delay) before next GenAI request.")
                time.sleep(sleep_time)

        self.last_request_time = time.time()
        if max_rpm > 0:
            self.request_count_minute += 1

    def _make_api_call(self, text_to_translate: str, model_to_use: str, generation_config_dict: Dict, safety_settings: Optional[Dict]) -> Tuple[str, Optional[Any]]:
        """
        Synchronous part of the API call. Designed to be run in a thread.
        Returns a tuple: (translated_text_or_error_string, response_object_or_None)
        """
        if not self.client:
            return "[Client Error: Not initialized]", None

        from_lang_display = self.lang_source 
        to_lang_display = self.lang_target
        
        from_lang_for_prompt = self.lang_map.get(from_lang_display, from_lang_display)
        to_lang_for_prompt = self.lang_map.get(to_lang_display, to_lang_display)

        system_instruction = self._get_param_value('system_prompt_template').format(
            from_lang=from_lang_for_prompt,
            to_lang=to_lang_for_prompt
        )
        user_content_prompt = self._get_param_value('user_prompt_template').format(
            text_to_translate=text_to_translate
        )
        
        full_prompt = f"{system_instruction}\n\n{user_content_prompt}"
        
        self.logger.debug(self._format_prompt_log(text_to_translate, system_instruction, user_content_prompt))

        try:
            # The actual SDK call
            response = self.client.models.generate_content(
                model=f'models/{model_to_use}', # Model name needs to be prefixed with 'models/'
                contents=full_prompt,
                generation_config=generation_config_dict, # Parameter name is 'generation_config'
                safety_settings=safety_settings
            )
                        
            if not response.candidates:
                block_reason_msg = "Unknown reason"
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason_msg = response.prompt_feedback.block_reason.name
                
                safety_ratings_details = []
                if response.prompt_feedback and response.prompt_feedback.safety_ratings:
                    for rating in response.prompt_feedback.safety_ratings:
                        safety_ratings_details.append(f"{rating.category.name}: {rating.probability.name}")
                safety_info = ", ".join(safety_ratings_details) if safety_ratings_details else "No safety ratings available"

                self.logger.error(f"GenAI translation blocked. Reason: {block_reason_msg}. Safety Ratings: [{safety_info}]")
                return f"[Blocked by Safety Filter: {block_reason_msg}]", response

            if response.candidates[0].finish_reason.name != "STOP":
                finish_reason_name = response.candidates[0].finish_reason.name
                self.logger.warning(f"GenAI translation finished with reason: {finish_reason_name}. Partial translation might be returned.")

            translated_text = response.text.strip()
            # self.logger.debug(f"GenAI Raw Response Text: \"{translated_text[:100]}...\"") # Logged by caller
            return translated_text, response

        except Exception as e:
            self.logger.error(f"Google GenAI API request failed: {e}")
            self.logger.debug(traceback.format_exc())
            if isinstance(e, types.BlockedPromptError): # Updated exception type
                 return f"[Prompt blocked by API: {e}]", None
            if isinstance(e, types.StopCandidateError): # Updated exception type
                 return f"[Candidate generation stopped unexpectedly: {e}]", None
            return f"[API request failed: {e}]", None

    async def _request_translation_async_job(self, text_to_translate: str, index: int) -> Tuple[int, str]:
        """
        Asynchronous wrapper for a single translation job, including retries and delay.
        """
        if not self.client:
            self.logger.error("Google GenAI client not initialized. Cannot translate.")
            return index, "[Client Error: Not initialized]"

        translated_text = ""
        max_attempts = self._get_param_value('retry_attempts', 3)
        
        # Prepare these once per text_to_translate to pass to _make_api_call
        model_to_use = self._get_param_value('override_model_name') or self._get_param_value('model_name')
        generation_config_dict = {
            'max_output_tokens': int(self._get_param_value('max_output_tokens')),
            'temperature': float(self._get_param_value('temperature')),
            'top_p': float(self._get_param_value('top_p')),
            'top_k': int(self._get_param_value('top_k')),
        }
        safety_settings = self.parsed_safety_settings

        for attempt in range(max_attempts + 1):
            try:
                # Apply delay *before* the call
                # _respect_delay is synchronous, so it's fine to call directly here
                # as it will block this specific async task, not the whole event loop.
                self._respect_delay()

                self.logger.info(f"Translating text ({index + 1}): \"{text_to_translate[:50]}...\" (Attempt {attempt+1}/{max_attempts+1})")
                
                loop = asyncio.get_event_loop()
                # Run the synchronous _make_api_call in a thread
                result_text, response_obj = await loop.run_in_executor(
                    self.executor, 
                    self._make_api_call, 
                    text_to_translate, 
                    model_to_use, 
                    generation_config_dict, 
                    safety_settings
                )

                if response_obj: # Indicates a successful or partially successful call
                     self.logger.debug(f"GenAI Raw Response Text for item {index+1}: \"{result_text[:100]}...\"")
                
                # Check if the result_text indicates an error string from _make_api_call
                if result_text.startswith("[Blocked by Safety Filter:") or \
                   result_text.startswith("[Prompt blocked by API:") or \
                   result_text.startswith("[Candidate generation stopped unexpectedly:") or \
                   result_text.startswith("[API request failed:") or \
                   result_text.startswith("[Client Error:"):
                    raise GenAIAPIError(result_text) # Propagate as an exception to trigger retry

                translated_text = result_text
                break 
            except GenAIAPIError as e:
                self.logger.warning(f"Attempt {attempt + 1} for item {index+1} (\"{text_to_translate[:50]}...\") failed: {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(self._get_param_value('retry_timeout', 10))
                else:
                    self.logger.error(f"All {max_attempts + 1} attempts failed for item {index+1} (\"{text_to_translate[:50]}...\").")
                    translated_text = str(e) # Return the error message as translation
            except Exception as e: 
                self.logger.error(f"Unexpected error on attempt {attempt + 1} for item {index+1} (\"{text_to_translate[:50]}...\"): {e}")
                self.logger.debug(traceback.format_exc())
                if attempt < max_attempts:
                    await asyncio.sleep(self._get_param_value('retry_timeout', 10))
                else:
                    translated_text = f"[Unexpected Translation Error: {e}]"
        return index, translated_text

    def _format_prompt_log(self, original_text: str, system_part: str, user_part: str) -> str:
        log_message = [
            "Google GenAI Prompt Log:",
            f"  Original Text Snippet: \"{original_text[:70]}...\"",
            f"  System Instruction Part: \"{system_part[:100]}...\"",
            f"  User Content Part: \"{user_part[:100]}...\"",
            # Target language info is already part of system_part or known contextually
        ]
        return "\n".join(log_message)

    async def _translate_async(self, src_list: List[str]) -> List[str]:
        if not self.client:
            self.logger.error("Google GenAI client not initialized. Cannot translate.")
            return [""] * len(src_list)

        translations = []
        num_texts = len(src_list)
        for i, src_text in enumerate(src_list):
            if not src_text.strip(): 
                translations.append("")
                continue

            translated_text = ""
            max_attempts = self._get_param_value('retry_attempts', 3)
            for attempt in range(max_attempts + 1):
                try:
                    self.logger.info(f"Translating text ({i+1}/{num_texts}): \"{src_text[:50]}...\" (Attempt {attempt+1}/{max_attempts+1})")
                    translated_text = self._request_translation(src_text)
                    break 
                except GenAIAPIError as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed for \"{src_text[:50]}...\": {e}")
                    if attempt < max_attempts:
                        time.sleep(self._get_param_value('retry_timeout', 10))
                    else:
                        self.logger.error(f"All {max_attempts + 1} attempts failed for \"{src_text[:50]}...\".")
                        translated_text = f"[Translation Error: {e}]" 
                except Exception as e: 
                    self.logger.error(f"Unexpected error on attempt {attempt + 1} for \"{src_text[:50]}...\": {e}")
                    self.logger.debug(traceback.format_exc())
                    if attempt < max_attempts:
                        time.sleep(self._get_param_value('retry_timeout', 10))
                    else:
                        translated_text = f"[Unexpected Translation Error: {e}]"
            translations.append(translated_text)
        return translations

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        critical_params = ['auth_method', 'api_key', 'service_account_json_path', 'proxy', 'safety_settings']
        if param_key in critical_params:
            self.logger.info(f"Parameter '{param_key}' updated. Re-initializing Google GenAI client.")
            self._initialize_client()
        elif param_key == 'service_account_json_path' and self._get_param_value('auth_method') == 'Service Account JSON':
            # Special handling to update project_id display even if not re-initializing fully (e.g. path typed manually)
            # However, _initialize_client will be called anyway by the check above.
            # This block is mostly for immediate feedback if the UI allows it.
            pass

    def _get_param_value(self, key: str, default_val=None):
        """Helper to safely get parameter values."""
        if key in self.params and 'value' in self.params[key]:
            return self.params[key]['value']
        if key in self.params and not isinstance(self.params[key], dict): # Fallback for simple params
            return self.params[key]
        return default_val