import re
import time
import yaml
import traceback
from typing import List, Dict, Optional
import json
import os

import httpx
from google import genai # 공식 문서에 따른 import
from google.genai import types as genai_types




from .base import BaseTranslator, register_translator


class InvalidNumTranslations(Exception):
    pass


@register_translator("LLM_API_Translator")
class GeminiTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        "apikey": {
            "value": "",
            "description": "API key for Google Gemini.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "Multiple API keys separated by semicolons (;). One key per line for readability. Rotates keys to manage RPM limits.",
        },
        "model": {
            "type": "selector",
            "options": [
                "gemini-2.5-pro-preview-06-05",
                "gemini-2.5-flash-preview-05-20",
                "gemini-2.0-flash",
                "gemini-2.5-pro-exp-03-25",
            ],
            "value": "gemini-2.0-flash",
            "description": "Select the Gemini model.",
        },
        "override model": {
            "value": "",
            "description": "Specify a custom Gemini model name to override the selection (e.g., specific version).",
        },
        "prompt template": {
            "type": "editor",
            "value": "Please help me to translate the following text from a manga to {to_lang} (if it's already in {to_lang} or looks like gibberish you have to output it as it is instead):\n",
        },
        "chat system template": {
            "type": "editor",
            "value": "You are a professional translation engine, please translate the text into a colloquial, elegant and fluent content, without referencing machine translations. You must only translate the text content, never interpret it. If there's any issue in the text, output the text as is.\nTranslate to {to_lang}.",
        },
        "chat sample": {
            "type": "editor",
            "value": """日本語-简体中文:
    source:
        - 二人のちゅーを 目撃した ぼっちちゃん
        - ふたりさん
        - 大好きなお友達には あいさつ代わりに ちゅーするんだって
        - アイス あげた
        - 喜多ちゃんとは どどど どういった ご関係なのでしようか...
        - テレビで見た！
    target:
        - 小孤独目击了两人的接吻
        - 二里酱
        - 我听说人们会把亲吻作为与喜爱的朋友打招呼的方式
        - 我给了她冰激凌
        - 喜多酱和你是怎么样的关系啊...
        - 我在电视上看到的！""",
        },
        "invalid repeat count": {
            "value": 2,
            "description": "Number of invalid repeat counts before considering translation failed.",
        },
        "max requests per minute": {
            "value": 20, # Gemini API는 분당 요청 수 제한(RPM)이 있으므로 이 값을 사용합니다.
            "description": "Maximum requests per minute for EACH API key.",
        },
        "delay": {
            "value": 0.3,
            "description": "Global delay in seconds between requests.",
        },
        "max tokens": {
            "value": 4096, # Gemini는 max_output_tokens로 제어합니다.
            "description": "Maximum output tokens for the response.",
        },
        "temperature": {
            "value": 0.5,
            "description": "Controls randomness. Lower for more deterministic, higher for more creative.",
        },
        "top p": {
            "value": 1.,
            "description": "Top P for sampling. Consider values like 0.95.",
        },
        "retry attempts": {
            "value": 5,
            "description": "Number of retry attempts on failure.",
        },
        "retry timeout": {
            "value": 15,
            "description": "Timeout between retry attempts (seconds).",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port). Note: google-genai SDK uses HTTP_PROXY/HTTPS_PROXY environment variables.",
        },
        "low vram mode": {
            "value": False,
            "description": "Check if running locally and facing VRAM issues.",
            "type": "checkbox",
        },
        "use_vertex_ai": {
            "value": False,
            "description": "Use Vertex AI instead of Gemini Developer API",
            "type": "checkbox",
        },
        "vertex_service_account_file": {
            "value": "",
            "description": "Path to Vertex AI service account JSON file. If empty, uses environment variables.",
            "path_selector": True, # 찾아보기 버튼 활성화
        },
        "vertex_location": {
            "value": "us-central1",
            "description": "GCP location for Vertex AI (e.g., us-central1, europe-west1)",
        },
        "vertex_project_id": {
            "value": "",
            "description": "GCP Project ID for Vertex AI. If empty, auto-extracted from service account file.",
        },
    }

    def _setup_translator(self):
        self.lang_map = {
            "简体中文": "Simplified Chinese",
            "繁體中文": "Traditional Chinese",
            "日本語": "Japanese",
            "English": "English",
            "한국어": "Korean",
            "Tiếng Việt": "Vietnamese",
            "čeština": "Czech",
            "Français": "French",
            "Deutsch": "German",
            "magyar nyelv": "Hungarian",
            "Italiano": "Italian",
            "Polski": "Polish",
            "Português": "Portuguese",
            "limba română": "Romanian",
            "русский язык": "Russian",
            "Español": "Spanish",
            "Türk dili": "Turkish",
            "украї́нська мо́ва": "Ukrainian",
            "Thai": "Thai",
            "Arabic": "Arabic",
            "Malayalam": "Malayalam",
            "Tamil": "Tamil",
            "Hindi": "Hindi",
        }
        self.token_count = 0 # Gemini API는 현재 토큰 사용량을 직접 반환하지 않습니다.
        self.token_count_last = 0
        self.current_key_index = 0
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}  # {api_key: (count, minute_start_time)}
        self.client = None # Google Gen AI Client
        # Vertex AI 환경 설정 추가
        self._setup_vertex_environment()

    def _setup_vertex_environment(self):
        """Vertex AI 환경 변수 자동 설정"""
        if not self.get_param_value("use_vertex_ai"):
            return
        
        project_id = self._get_project_id()
        location = self.get_param_value("vertex_location") or "us-central1"
        
        # 환경 변수 설정
        if project_id:
            os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
            os.environ['GOOGLE_CLOUD_LOCATION'] = location
            # google-genai SDK는 vertexai=True 플래그로 Vertex AI 사용을 명시하므로,
            # GOOGLE_GENAI_USE_VERTEXAI는 필수는 아닐 수 있습니다.
            # os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'true' 
            
            self.logger.debug(f"Set Vertex AI environment variables: PROJECT={project_id}, LOCATION={location}")

    def _load_service_account_info(self) -> Optional[Dict]:
        """서비스 계정 JSON 파일에서 정보 로드"""
        service_account_file = self.get_param_value("vertex_service_account_file").strip()
        
        if not service_account_file:
            return None
        
        try:
            from pathlib import Path # pathlib 임포트
            file_path = Path(service_account_file).expanduser()
            if not file_path.exists():
                self.logger.error(f"Service account file not found: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                service_account_info = json.load(f)
            
            self.logger.debug(f"Loaded service account from: {file_path}")
            return service_account_info
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in service account file: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading service account file: {e}")
            return None

    def _ensure_client(self):
        """Gemini Developer API 또는 Vertex AI 클라이언트 초기화"""
        use_vertex = self.get_param_value("use_vertex_ai")
        
        if use_vertex:
            self._ensure_vertex_client()
        else:
            self._ensure_gemini_client()

    def _ensure_gemini_client(self):
        """기존 Gemini Developer API 클라이언트 초기화"""
        current_api_key = self._select_api_key()
        if not current_api_key:
            self.logger.error("No API key available for translation.")
            raise ValueError("No API key available.")
        
        # 프록시 설정은 genai.Client 생성 시 직접 지원하지 않음.
        # google-genai는 일반적으로 HTTP_PROXY/HTTPS_PROXY 환경 변수를 사용합니다.
        # httpx_client를 명시적으로 전달하여 프록시를 설정할 수 있으나,
        # 여기서는 환경 변수 사용을 권장하는 로그만 남깁니다.
        if self.proxy:
            self.logger.info(
                f"Proxy configured: {self.proxy}. Ensure your environment (HTTP_PROXY, HTTPS_PROXY) is set up for google-generativeai SDK."
            )

        if not self.client: # 또는 API 키가 변경된 경우 클라이언트 재생성 (여기서는 단순화)
            try:
                self.client = genai.Client(api_key=current_api_key)
                self.logger.debug(f"Initialized Google Gen AI client")
            except Exception as e:
                self.logger.error(f"Failed to initialize client: {e}")
                self.client = None
                raise

    def _get_project_id(self) -> Optional[str]:
        """프로젝트 ID 자동 추출"""
        # 수동 설정된 프로젝트 ID 우선 사용
        manual_project_id = self.get_param_value("vertex_project_id").strip()
        if manual_project_id:
            return manual_project_id
        
        # 서비스 계정 파일에서 추출
        service_account_info = self._load_service_account_info()
        if service_account_info and 'project_id' in service_account_info:
            project_id = service_account_info['project_id']
            self.logger.info(f"Auto-extracted project ID: {project_id}")
            return project_id
        
        # 환경 변수에서 확인 (GOOGLE_CLOUD_PROJECT는 _setup_vertex_environment에서 설정될 수 있음)
        env_project = os.environ.get('GOOGLE_CLOUD_PROJECT')
        if env_project:
            self.logger.info(f"Using project ID from environment: {env_project}")
            return env_project
        
        return None

    def _ensure_vertex_client(self):
        """Vertex AI 클라이언트 초기화"""
        project_id = self._get_project_id()
        if not project_id:
            self.logger.error("No project ID available for Vertex AI")
            raise ValueError("No project ID available for Vertex AI")
        
        location = self.get_param_value("vertex_location") or "us-central1"
        service_account_file = self.get_param_value("vertex_service_account_file").strip()
        
        try:
            # 서비스 계정 파일 설정
            if service_account_file:
                from pathlib import Path # pathlib 임포트
                file_path = Path(service_account_file).expanduser()
                if file_path.exists():
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(file_path)
                    self.logger.debug(f"Set GOOGLE_APPLICATION_CREDENTIALS: {file_path}")
                else:
                    self.logger.warning(f"Service account file not found: {file_path}, will try default credentials.")
            
            # Vertex AI 클라이언트 생성
            self.client = genai.Client(
                project=project_id, # project_id 대신 project 사용
                location=location,
                vertexai=True,
            )
            
            self.logger.info(f"Initialized Vertex AI client for project: {project_id}, location: {location}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Vertex AI client: {e}")
            self.client = None
            raise

    @property
    def use_vertex_ai(self) -> bool:
        return bool(self.get_param_value("use_vertex_ai"))

    @property
    def vertex_location(self) -> str:
        return self.get_param_value("vertex_location") or "us-central1"

    @property
    def vertex_project_id(self) -> Optional[str]:
        return self.get_param_value("vertex_project_id").strip() or None

    @property
    def vertex_service_account_file(self) -> Optional[str]:
        return self.get_param_value("vertex_service_account_file").strip() or None

    def get_param_value(self, key: str) -> any:
        """Helper to get param value, considering dict structure."""
        param = self.params.get(key)
        if isinstance(param, dict):
            return param.get("value")
        return param # Direct value if not a dict (e.g. for older param structures)

    def _select_api_key(self) -> str:
        api_keys = self.multiple_keys_list
        if not api_keys:
            return self.apikey # 단일 키 사용

        # 여러 키가 있는 경우 로테이션
        selected_key = None
        for _ in range(len(api_keys)): # 모든 키를 한 번씩 확인
            index = self.current_key_index % len(api_keys)
            key_to_check = api_keys[index]
            
            count, start_time = self.key_usage.get(key_to_check, (0, time.time()))
            now = time.time()
            rpm_limit = int(self.get_param_value("max requests per minute"))

            if now - start_time >= 60: # 1분이 지났으면 리셋
                self.key_usage[key_to_check] = (0, now)
                count = 0
            
            if rpm_limit <= 0 or count < rpm_limit: # RPM 제한이 없거나, 아직 여유가 있으면
                selected_key = key_to_check
                self.key_usage[selected_key] = (count + 1, start_time if now - start_time < 60 else now)
                self.current_key_index = (index + 1) % len(api_keys) # 다음 요청을 위해 인덱스 이동
                break
            
            self.current_key_index = (index + 1) % len(api_keys) # 다음 키로 넘어감

        if not selected_key: # 모든 키가 RPM 제한에 도달한 경우
            # 가장 오래전에 사용된 (또는 곧 리셋될) 키를 선택하고 대기
            # 간단하게 첫 번째 키를 선택하고 _respect_key_limit에서 대기하도록 함
            selected_key = api_keys[self.current_key_index % len(api_keys)]
            self._respect_key_limit(selected_key) # 여기서 대기 발생
            count, start_time = self.key_usage.get(selected_key, (0, time.time()))
            self.key_usage[selected_key] = (count + 1, start_time) # 사용량 업데이트
            self.current_key_index = (self.current_key_index + 1) % len(api_keys)

        return selected_key

    def _respect_key_limit(self, key: str):
        rpm = int(self.get_param_value("max requests per minute"))
        if rpm <= 0:
            return
        
        count, start_time = self.key_usage.get(key, (0, time.time()))
        now = time.time()

        if now - start_time >= 60:
            self.key_usage[key] = (0, now) # 분이 지났으면 카운트 리셋
            count = 0 # 아래 로직에서 사용하기 위해 업데이트
        
        if count >= rpm:
            wait_time = 60.1 - (now - start_time) # 약간의 버퍼 추가
            masked_key = key[:6] + "*" * (len(key) - 6)
            self.logger.warning(
                f"Key {masked_key} reached RPM limit ({rpm}). Waiting {wait_time:.2f} seconds."
            )
            if wait_time > 0:
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time()) # 대기 후 카운트 리셋

    def _respect_delay(self):
        current_time = time.time()
        # Global RPM limit (if any, though individual key limits are primary)
        # This part can be simplified if key-specific RPM is the main concern.
        # For now, keeping a global check as a fallback or for overall rate control.
        global_rpm_limit = int(self.get_param_value("max requests per minute")) # Assuming this is a global limit if no multiple keys

        if global_rpm_limit > 0 and not self.multiple_keys_list: # Only apply if single key or as a general cap
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time

            if self.request_count_minute >= global_rpm_limit:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Reached global RPM limit ({global_rpm_limit}). Waiting {wait_time:.2f} seconds."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        delay = float(self.get_param_value("delay"))
        if time_since_last_request < delay:
            sleep_time = delay - time_since_last_request
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.last_request_time = time.time()
        if not self.multiple_keys_list: # Only increment global counter if not using multiple keys (key-specific handles it)
            self.request_count_minute += 1

    def _request_translation(self, prompt: str, chat_sample: Optional[List[str]]) -> str:
        self._respect_delay() # Global delay

        # Select key and respect its limit BEFORE making the call
        if not self.use_vertex_ai: # Vertex AI uses service account, not API keys in this way
            api_key = self._select_api_key()
            if not api_key:
                return "Error: No API key available."
            self._respect_key_limit(api_key) # Respect key-specific RPM
        
        try:
            self._ensure_client()
        except ValueError as ve:
            return str(ve)
        except Exception as e:
            self.logger.error(f"Error during client initialization: {e}")
            return f"Error: Client initialization failed - {e}"

        result = self._request_translation_with_chat_sample_google(prompt, chat_sample)
        if not isinstance(result, str):
            result = str(result)
        return result

    def _translate(self, src_list: List[str]) -> List[str]:
        translations = []
        to_lang = self.lang_map[self.lang_target]
        queries = src_list
        chat_sample = self.chat_sample

        for prompt, num_src in self._assemble_prompts(queries, to_lang=to_lang):
            retry_attempt = 0
            while True:
                try:
                    response_text = self._request_translation(prompt, chat_sample)
                    if not isinstance(response_text, str):
                        response_text = str(response_text) # 응답이 문자열이 아닐 경우 변환
                    
                    new_translations = re.split(r"<\|\d+\|>", response_text)[-num_src:]
                    
                    if len(new_translations) != num_src:
                        # 번역 결과가 예상과 다를 경우, 줄바꿈 기준으로 재분할 시도
                        _tr2 = re.sub(r"<\|\d+\|>", "", response_text).split("\n")
                        if len(_tr2) == num_src:
                            new_translations = _tr2
                        else:
                            # 그래도 개수가 맞지 않으면 예외 발생
                            raise InvalidNumTranslations(f"Expected {num_src} translations, got {len(new_translations)}. Response: '{response_text[:200]}...'")
                    break 
                except InvalidNumTranslations as e:
                    retry_attempt += 1
                    message = f"Translation count mismatch: {e}\nprompt:\n{prompt}\ntranslations:\n{new_translations}\nresponse:\n{response_text}"
                    if retry_attempt >= self.retry_attempts:
                        self.logger.error(message)
                        new_translations = [""] * num_src # 실패 시 빈 문자열로 채움
                        break
                    self.logger.warning(
                        message + f"\nRetrying. Attempt: {retry_attempt}"
                    )
                except Exception as e:
                    retry_attempt += 1
                    if retry_attempt >= self.retry_attempts:
                        new_translations = [""] * num_src # 실패 시 빈 문자열로 채움
                        break
                    self.logger.warning(
                        f"Translation failed: {e}. Attempt: {retry_attempt}, sleep {self.retry_timeout}s..."
                    )
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    time.sleep(self.retry_timeout)
            translations.extend([t.strip() for t in new_translations])

        # Gemini API는 현재 토큰 사용량 정보를 응답에 포함하지 않음
        # self.logger.info(f"Token count information is not available for Gemini API.")

        return translations
    
    def _translate_single_block(self, text_block: str, to_lang: str, chat_sample: Optional[List[str]]) -> str:
        """단일 텍스트 블록을 번역합니다."""
        prompt_template = (
            self.params["prompt template"]["value"].format(to_lang=to_lang).rstrip()
        )

        # 단일 블록이므로 <|1|> 구분자는 제거하거나, _request_translation_with_chat_sample_google에서 처리하도록 단순화
        prompt = f"{prompt_template}\n{text_block}"

        # 단일 블록 번역 시에도 전체 번역과 동일한 안전 설정 및 기타 설정을 사용합니다.
        config = genai_types.GenerateContentConfig(
            system_instruction=self.chat_system_template,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            safety_settings=[
                genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
        )

        

        retry_attempt = 0
        while retry_attempt < self.retry_attempts:
            try:
                # _request_translation_with_chat_sample_google을 직접 호출하는 대신,
                # 여기서는 _request_translation_with_chat_sample_google을 직접 호출한다고 가정.
                # chat_sample은 contents 구성 시 함께 전달
                contents_for_single_block = []
                if chat_sample:
                    contents_for_single_block.extend([
                        {"role": "user", "parts": [{"text": chat_sample[0]}]},
                        {"role": "model", "parts": [{"text": chat_sample[1]}]}
                    ])
                # 시스템 프롬프트는 config에 있으므로, 여기서는 사용자 입력(번역할 텍스트 블록)만 전달
                contents_for_single_block.append({"role": "user", "parts": [{"text": text_block}]})


                response = self.client.models.generate_content(
                    model=(self.override_model or self.model),
                    contents=contents_for_single_block,
                    config=config
                )

                if self._handle_safety_response(response): # 안전 필터에 걸렸는지 확인
                    self.logger.warning(f"Single block '{text_block[:30]}...' blocked by safety filters (response check).")
                    return f"[안전 필터로 차단됨: {text_block[:30]}...]"
                
                return response.text.strip() if hasattr(response, 'text') and response.text else ""


            except Exception as e:
                if self._check_safety_block(e): # 예외 객체에서 안전 필터링 확인
                    self.logger.warning(f"Single block blocked by safety filters (exception): {text_block[:30]}... Error: {e}")
                    return f"[안전 필터로 차단됨: {text_block[:30]}...]"
        
                retry_attempt += 1
                if retry_attempt >= self.retry_attempts:
                    self.logger.error(f"Single block translation for '{text_block[:30]}...' failed after {self.retry_attempts} attempts: {e}")
                    return f"[번역 실패: {str(e)[:30]}]"
                
                self.logger.warning(f"Single block translation for '{text_block[:30]}...' failed. Attempt {retry_attempt}/{self.retry_attempts}. Error: {e}. Sleeping for {self.retry_timeout}s...")
                
                
                time.sleep(self.retry_timeout)
        return f"[번역 실패: 최대 재시도 도달]"


    def _check_safety_block(self, exception: Exception) -> bool:
        """안전 필터링으로 인한 차단인지 확인"""


        # 1. 특정 예외 타입 확인 (가장 확실한 방법)
        if isinstance(exception, genai_types.BlockedPromptException):
            self.logger.warning(f"Content blocked by API (BlockedPromptException). Reason: {exception}")
            return True


        if hasattr(exception, 'response'): # 일부 API 오류는 response 속성을 가질 수 있음
            response = exception.response
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    block_reason_value = response.prompt_feedback.block_reason
                    # BlockedReason enum 값과 비교 또는 문자열로 변환하여 비교
                    if isinstance(block_reason_value, genai_types.BlockedReason):
                        block_reason_str = block_reason_value.name # PROHIBITED_CONTENT, OTHER 등
                    else: # 이미 문자열일 경우
                        block_reason_str = str(block_reason_value)
                    
                    if block_reason_str in ['PROHIBITED_CONTENT', 'OTHER', 'SAFETY']: # SAFETY도 추가
                        self.logger.warning(f"Content blocked by API (exception.response.prompt_feedback). Reason: {block_reason_str}")
                        return True
            
            # 2. candidates 내 finishReason 확인
            if hasattr(response, 'candidates'):
                for candidate in response.candidates:
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason_value = candidate.finish_reason
                        if isinstance(finish_reason_value, genai_types.FinishReason):
                            finish_reason_str = finish_reason_value.name # SAFETY 등
                        else:
                            finish_reason_str = str(finish_reason_value)

                        if finish_reason_str == 'SAFETY':
                            self.logger.warning("Content blocked due to safety filters (exception.response.candidate.finish_reason).")
                            # 상세 안전 평가 로깅
                            if hasattr(candidate, 'safety_ratings'):
                                for rating in candidate.safety_ratings:
                                    if hasattr(rating, 'blocked') and rating.blocked:
                                        self.logger.warning(f"Blocked category: {rating.category}, probability: {rating.probability}")
                            return True


        # 예외 메시지 확인 (보조적 방법)
        error_msg = str(exception).upper()
        safety_keywords = ['PROHIBITED_CONTENT', 'SAFETY', 'BLOCKED', 'HARMFUL'] # PROHIBITED_CONTENT는 로그에서 확인된 키워드
        is_safety_issue = any(keyword in error_msg for keyword in safety_keywords)
        if is_safety_issue:
            self.logger.warning(f"Safety-related keyword found in error message: {str(exception)}")
        return is_safety_issue
    
    def _handle_safety_response(self, response) -> bool:
        """공식 문서 기준 응답 구조 확인하여 안전 필터링으로 차단되었는지 여부 반환"""
        if not response:
            return False
            
        # promptFeedback 확인
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                block_reason_value = response.prompt_feedback.block_reason
                block_reason_str = block_reason_value.name if isinstance(block_reason_value, genai_types.BlockedReason) else str(block_reason_value)
                if block_reason_str in ['PROHIBITED_CONTENT', 'OTHER', 'SAFETY']:
                    self.logger.warning(f"Response indicates content block (prompt_feedback). Reason: {block_reason_str}")
                    return True
        
        # candidates의 safetyRatings 확인
        if hasattr(response, 'candidates'):
            for candidate in response.candidates:
                if hasattr(candidate, 'finish_reason') and isinstance(candidate.finish_reason, genai_types.FinishReason) and candidate.finish_reason == genai_types.FinishReason.SAFETY:
                    if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                        for rating in candidate.safety_ratings:
                            if hasattr(rating, 'blocked') and rating.blocked:
                                self.logger.warning(f"Content blocked (candidate.finish_reason=SAFETY). Category: {rating.category}, Probability: {rating.probability}")
                    return True
        return False


    def _fallback_to_block_translation(self, src_list: List[str], to_lang: str, chat_sample: Optional[List[str]]) -> List[str]:
        """블록별 번역 폴백 메서드"""
        translations = []
        for i, text_block in enumerate(src_list):
            try:
                translation = self._translate_single_block(text_block, to_lang, chat_sample)
                translations.append(translation.strip())
                self.logger.debug(f"Successfully translated block {i+1}/{len(src_list)} via fallback.")
            except Exception as e_block:
                self.logger.error(f"Failed to translate block {i+1} during fallback: {e_block}")
                translations.append(f"[폴백 번역 실패: {text_block[:30]}...]")
        return translations

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        self.logger.debug(
            f"updateParam called for key: {param_key}, content: {param_content}"
        )
        if param_key in [
            "apikey",
            "multiple_keys", 
            "use_vertex_ai",
            "vertex_service_account_file",
            "vertex_location",
            "vertex_project_id",
            "model",
            "override_model",
            "proxy", # 프록시 변경 시 클라이언트 재설정 (환경 변수 외 명시적 설정 시)
        ]:
            self.client = None # 클라이언트 재초기화 플래그
            self.logger.debug(f"Client reset due to parameter change: {param_key}")
            if param_key in ["use_vertex_ai", "vertex_service_account_file", "vertex_location", "vertex_project_id"]:
                self._setup_vertex_environment() # Vertex AI 관련 환경변수 재설정
                raise

    @property
    def apikey(self) -> str:
        return self.get_param_value("apikey")

    @property
    def multiple_keys_list(self) -> List[str]:
        keys_str = self.get_param_value("multiple_keys").strip()
        return [key.strip() for key in keys_str.split(";") if key.strip()]

    @property
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override model") or None

    @property
    def temperature(self) -> float:
        return float(self.get_param_value("temperature"))

    @property
    def top_p(self) -> float:
        return float(self.get_param_value("top p"))

    @property
    def max_tokens(self) -> int: # Gemini에서는 max_output_tokens
        return int(self.get_param_value("max tokens"))

    @property
    def retry_attempts(self) -> int:
        return int(self.get_param_value("retry attempts"))

    @property
    def retry_timeout(self) -> int:
        return int(self.get_param_value("retry timeout"))

    @property
    def proxy(self) -> str:
        return self.get_param_value("proxy")

    @property
    def chat_system_template(self) -> str:
        to_lang = self.lang_map[self.lang_target]
        return self.params["chat system template"]["value"].format(to_lang=to_lang)

    @property
    def chat_sample(self):
        samples_str = self.get_param_value("chat sample")
        try:
            samples = yaml.load(samples_str, Loader=yaml.FullLoader)
        except Exception as e:
            self.logger.error(f"Failed to parse chat sample YAML: {samples_str} - {e}")
            return None
        
        src_tgt_key = f"{self.lang_source}-{self.lang_target}"
        if samples and src_tgt_key in samples:
            sample_data = samples[src_tgt_key]
            if "source" in sample_data and "target" in sample_data:
                src_queries = "\n".join(
                    [f"<|{i+1}|>{s}" for i, s in enumerate(sample_data["source"])]
                )
                tgt_queries = "\n".join(
                    [f"<|{i+1}|>{t}" for i, t in enumerate(sample_data["target"])]
                )
                return [src_queries, tgt_queries]
            else:
                self.logger.warning(f"'{src_tgt_key}' in chat sample is missing 'source' or 'target' keys.")
        return None

    def _assemble_prompts(
        self, queries: List[str], to_lang: str = None, max_tokens_override=None # max_tokens_override로 변경
    ):
        if to_lang is None:
            to_lang = self.lang_map[self.lang_target]
        prompt_template = (
            self.params["prompt template"]["value"].format(to_lang=to_lang).rstrip()
        )
        prompt = prompt_template
        num_src = 0
        i_offset = 0

        # Gemini는 입력 토큰 제한도 고려해야 하므로, max_tokens를 출력 토큰으로만 사용
        # 입력 길이 제어는 여기서 간단히 문자 수로 처리 (정확한 토큰화는 API 호출 전 수행)
        # 이 부분은 더 정교한 토큰 기반 분할 로직으로 개선될 수 있습니다.
        # 현재는 출력 max_tokens를 기준으로 분할합니다.
        effective_max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens


        for i, query in enumerate(queries):
            prompt += f"\n<|{i+1-i_offset}|>{query}"
            num_src += 1
            # Approximate check, real tokenization happens later
            if effective_max_tokens * 2 and len("".join(queries[i + 1 :])) * 1.5 > effective_max_tokens : # 1.5는 문자당 평균 토큰 추정치
                yield prompt.lstrip(), num_src
                prompt = prompt_template
                i_offset = i + 1
                num_src = 0
        yield prompt.lstrip(), num_src

    def _format_prompt_log(self, prompt: str) -> str:
        chat_sample = self.chat_sample
        if chat_sample: # Gemini는 항상 chat sample을 사용할 수 있음
            return "\n".join(
                [
                    "System:",
                    self.chat_system_template,
                    "User Sample:",
                    chat_sample[0],
                    "Assistant Sample:",
                    chat_sample[1],
                    "User Prompt:",
                    prompt,
                ]
            )
        return "\n".join(["System:", self.chat_system_template, "User Prompt:", prompt])

    def _respect_delay(self):
        current_time = time.time()
        rpm_limit = int(self.get_param_value("max requests per minute"))

        if rpm_limit > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time

            if self.request_count_minute >= rpm_limit:
                wait_time = 60.1 - (current_time - self.minute_start_time) # 약간의 버퍼 추가
                if wait_time > 0:
                    self.logger.warning(
                        f"Reached global RPM limit ({rpm_limit}). Waiting {wait_time:.2f} seconds."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        
        delay = float(self.get_param_value("delay"))
        if time_since_last_request < delay:
            sleep_time = delay - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    def _respect_key_limit(self, key: str):
        rpm = int(self.get_param_value("max requests per minute"))
        if rpm <= 0:
            return
        
        count, start_time = self.key_usage.get(key, (0, time.time()))
        now = time.time()

        if now - start_time >= 60:
            self.key_usage[key] = (0, now) # 분이 지났으면 카운트 리셋
            count = 0 # 아래 로직에서 사용하기 위해 업데이트
        
        if count >= rpm:
            wait_time = 60.1 - (now - start_time) # 약간의 버퍼 추가
            masked_key = key[:6] + "*" * (len(key) - 6)
            self.logger.warning(
                f"Key {masked_key} reached RPM limit ({rpm}). Waiting {wait_time:.2f} seconds."
            )
            time.sleep(wait_time)
            self.key_usage[key] = (0, time.time()) # 대기 후 카운트 리셋

    def _select_api_key(self) -> str:
        api_keys = self.multiple_keys_list
        if not api_keys:
            return self.apikey # 단일 키 사용

        # 여러 키가 있는 경우 로테이션
        selected_key = None
        for _ in range(len(api_keys)): # 모든 키를 한 번씩 확인
            index = self.current_key_index % len(api_keys)
            key_to_check = api_keys[index]
            
            count, start_time = self.key_usage.get(key_to_check, (0, time.time()))
            now = time.time()
            rpm_limit = int(self.get_param_value("max requests per minute"))

            if now - start_time >= 60: # 1분이 지났으면 리셋
                self.key_usage[key_to_check] = (0, now)
                count = 0
            
            if rpm_limit <= 0 or count < rpm_limit: # RPM 제한이 없거나, 아직 여유가 있으면
                selected_key = key_to_check
                self.key_usage[selected_key] = (count + 1, start_time if now - start_time < 60 else now)
                self.current_key_index = (index + 1) % len(api_keys) # 다음 요청을 위해 인덱스 이동
                break
            
            self.current_key_index = (index + 1) % len(api_keys) # 다음 키로 넘어감

        if not selected_key: # 모든 키가 RPM 제한에 도달한 경우
            # 가장 오래전에 사용된 (또는 곧 리셋될) 키를 선택하고 대기
            # 간단하게 첫 번째 키를 선택하고 _respect_key_limit에서 대기하도록 함
            selected_key = api_keys[self.current_key_index % len(api_keys)]
            self._respect_key_limit(selected_key) # 여기서 대기 발생
            count, start_time = self.key_usage.get(selected_key, (0, time.time()))
            self.key_usage[selected_key] = (count + 1, start_time) # 사용량 업데이트
            self.current_key_index = (self.current_key_index + 1) % len(api_keys)

        return selected_key


    def _request_translation(self, prompt: str, chat_sample: Optional[List[str]]) -> str:
        self._respect_delay()

        try:
            self._ensure_client()  # configure 대신 client 초기화
        except ValueError as ve: # No API key available
            return str(ve)
        except Exception as e:
            self.logger.error(f"Error during client initialization: {e}")
            return f"Error: Client initialization failed - {e}"

        result = self._request_translation_with_chat_sample_google(prompt, chat_sample)
        if not isinstance(result, str):
            result = str(result)
        return result

    def _request_translation_with_chat_sample_google(
        self, prompt: str, chat_sample: Optional[List[str]]
    ) -> str:
        if not self.client:
            self.logger.error("Client is not initialized.")
            return "Error: Client not initialized."

        model_name = self.override_model or self.model
        

        # Contents는 사용자 메시지만 포함

        contents = []
        
        if chat_sample:
            # 샘플 대화 추가
            contents.extend([
                {"role": "user", "parts": [{"text": chat_sample[0]}]},
                {"role": "model", "parts": [{"text": chat_sample[1]}]}
            ])
        
        # 실제 요청 프롬프트 추가 (시스템 프롬프트와 결합)
        contents.append({
            "role": "user", 
            "parts": [{"text": prompt}] # 시스템 프롬프트는 system_instruction으로 분리
        })

        # GenerateContentConfig 사용
        config = genai_types.GenerateContentConfig(
            system_instruction=self.chat_system_template,  # 문자열로 직접 전달
            max_output_tokens=self.max_tokens, # Gemini는 max_output_tokens 사용
            temperature=self.temperature, 
            top_p=self.top_p,
            safety_settings=[
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", 
                    threshold="BLOCK_NONE"
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                ),
            ]
        )

        try:
            # 올바른 API 호출 방법
            response = self.client.models.generate_content(
                model=model_name,  # "models/" 접두사 불필요
                contents=contents,
                config=config  # generation_config 대신 config 사용
            )
        except Exception as e:
            self.logger.error(f"Gemini API request failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: API request failed - {e}"


        # 응답 검증
        if not response or not hasattr(response, 'text'):
            self.logger.warning(f"Empty or invalid response received. Prompt: {prompt[:100]}... Response: {response}")
            return "Error: Invalid response format"
        
        if response.text:

            return response.text
        
        self.logger.warning(f"No content found in Gemini response. Prompt: {prompt[:100]}... Response: {response}")
        return ""


    def _translate(self, src_list: List[str]) -> List[str]:
        translations = []
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        queries = src_list
        chat_sample = self.chat_sample

        chunk_start_index = 0
        for assembled_prompt_str, num_src_in_chunk in self._assemble_prompts(queries, to_lang=to_lang):
            retry_attempt = 0
            while retry_attempt < self.retry_attempts: # 청크 단위 재시도 루프     
                try:
                    # _request_translation은 내부적으로 _request_translation_with_chat_sample_google을 호출하고,
                    # 이는 self.client.models.generate_content를 사용함.
                    # 여기서 반환되는 response_text는 실제 번역 결과 문자열이거나, 오류 발생 시 "Error: ..." 형태의 문자열.
                    # _request_translation_with_chat_sample_google에서 _handle_safety_response를 호출하여
                    # 안전 필터링된 경우 특정 문자열을 반환하도록 수정하거나, 예외를 발생시켜야 함.
                    # 현재 로직에서는 _request_translation_with_chat_sample_google이 예외를 발생시키거나,
                    # 응답 객체 자체를 반환하여 _handle_safety_response로 검사하는 것이 더 명확함.
                    # 우선은 _request_translation이 문자열을 반환한다고 가정하고 진행.
                    response_text = self._request_translation(assembled_prompt_str, chat_sample)

                    # _request_translation이 반환한 문자열이 실제 오류 메시지인지 확인
                    if response_text.startswith("Error: API request failed -") and "PROHIBITED_CONTENT" in response_text.upper():
                         # 이 경우는 _request_translation_with_chat_sample_google 내부에서 예외를 잡고 문자열로 반환한 경우
                        raise genai_types.BlockedPromptException(response_text) # 직접 예외 발생시켜 아래에서 처리
                    elif response_text.startswith("Error:"):
                        # 기타 API 또는 클라이언트 오류
                        raise Exception(response_text)


                    if not isinstance(response_text, str):
                        response_text = str(response_text) # 응답이 문자열이 아닐 경우 변환
                    
                    new_translations = re.split(r"<\|\d+\|>", response_text)[-num_src_in_chunk:]
                    
                    if len(new_translations) != num_src_in_chunk:
                        # 번역 결과가 예상과 다를 경우, 줄바꿈 기준으로 재분할 시도
                        _tr2 = re.sub(r"<\|\d+\|>", "", response_text).split("\n")
                        if len(_tr2) == num_src_in_chunk:
                            new_translations = _tr2
                        else:
                            # 그래도 개수가 맞지 않으면 예외 발생
                            raise InvalidNumTranslations(f"Expected {num_src_in_chunk} translations, got {len(new_translations)}. Response: '{response_text[:200]}...'")
                    translations.extend([t.strip() for t in new_translations]) # 성공 시 결과 추가
                    break 
                except InvalidNumTranslations as e:
                    retry_attempt += 1
                    message = f"Translation count mismatch for chunk: {e}\nprompt:\n{assembled_prompt_str[:100]}...\ntranslations:\n{new_translations}\nresponse:\n{response_text[:100]}..."
                    if retry_attempt >= self.retry_attempts: # 재시도 횟수 초과
                        self.logger.error(message)
                        # 해당 청크에 대해 빈 번역 또는 오류 메시지 추가
                        translations.extend(["[번역 오류: 개수 불일치]"] * num_src_in_chunk)
                        break
                    self.logger.warning(
                        message + f"\nRetrying. Attempt: {retry_attempt}"
                    )
                except Exception as e:
                    if self._check_safety_block(e):
                        self.logger.warning(f"Chunk translation blocked by safety filters. Error: {e}. Prompt: '{assembled_prompt_str[:100]}...'. Falling back to block-by-block for this chunk.")
                        # 현재 청크의 원본 텍스트 추출                    
                        current_chunk_texts = queries[chunk_start_index : chunk_start_index + num_src_in_chunk]
                        
                        chunk_translations = self._fallback_to_block_translation(current_chunk_texts, to_lang, chat_sample)
                        translations.extend(chunk_translations)
                        break # 현재 청크에 대한 재시도 루프 종료
                    else: # 안전 관련 예외가 아닌 다른 예외
                        retry_attempt += 1
                        if retry_attempt >= self.retry_attempts:
                            self.logger.error(f"Chunk translation failed after {self.retry_attempts} attempts for prompt: '{assembled_prompt_str[:100]}...'. Error: {e}")
                            translations.extend([f"[번역 오류: {str(e)[:30]}]"] * num_src_in_chunk)
                            
                            break 
                        self.logger.warning(f"Chunk translation attempt {retry_attempt}/{self.retry_attempts} failed: {e}. Prompt: '{assembled_prompt_str[:100]}...'. Sleeping for {self.retry_timeout}s...")
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                        time.sleep(self.retry_timeout)
            
            chunk_start_index += num_src_in_chunk # 다음 청크의 시작 인덱스 업데이트


        return translations

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        self.logger.debug(
            f"updateParam called for key: {param_key}, content: {param_content}"
        )
        if param_key in [
            "apikey", # API 키 변경 시 클라이언트 재설정
            "multiple_keys", # 위와 동일
            "proxy", # 프록시 변경 시 클라이언트 재설정 (환경 변수 외 명시적 설정 시)
            "model", 
            "override_model", 
        ]:
            self.client = None # 클라이언트 재초기화 플래그
