import re
import time
import yaml
import traceback
from typing import List, Dict, Optional
import json
import concurrent.futures
import threading
import os

import httpx
from google import genai # 공식 문서에 따른 import
from google.genai import types as genai_types # 공식 google-genai SDK에 따른 types import (이전 SDK에서는 from google.generativeai.types)



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
        self.rpm_lock = threading.Lock() # RPM 관리를 위한 Lock 추가
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
            # Gemini API 클라이언트는 _request_translation_thread_safe 내에서 API 키와 함께 처리
            pass

    def _ensure_gemini_client(self, api_key: str):
        """특정 API 키로 Gemini Developer API 클라이언트 초기화 또는 업데이트"""
        if not api_key:
            self.logger.error("No API key provided for Gemini client.")
            raise ValueError("No API key available.")

        if self.proxy:
            self.logger.info(
                f"Proxy configured: {self.proxy}. Ensure your environment (HTTP_PROXY, HTTPS_PROXY) is set up for google-generativeai SDK."
            )

        # 클라이언트가 없거나, 현재 클라이언트의 API 키가 다르다면 (이 부분은 단순화된 예시) 재생성
        # 실제로는 self.client.api_key 와 같은 속성이 없으므로, API 키 변경 시 항상 재생성
        if not self.client or getattr(self.client, '_api_key', None) != api_key:
            try:
                self.client = genai.Client(api_key=api_key)
                setattr(self.client, '_api_key', api_key) # API 키 추적을 위한 임시 속성
                self.logger.debug(f"Initialized Google Gen AI client with key: {api_key[:6]}...")
            except Exception as e:
                self.logger.error(f"Failed to initialize client with key {api_key[:6]}...: {e}")
                self.client = None
                raise

    def _get_project_id(self) -> Optional[str]:
        """프로젝트 ID 자동 추출"""
        manual_project_id = self.get_param_value("vertex_project_id").strip()
        if manual_project_id:
            return manual_project_id
        
        service_account_info = self._load_service_account_info()
        if service_account_info and 'project_id' in service_account_info:
            project_id = service_account_info['project_id']
            self.logger.info(f"Auto-extracted project ID: {project_id}")
            return project_id
        
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
            if service_account_file:
                from pathlib import Path
                file_path = Path(service_account_file).expanduser()
                if file_path.exists():
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(file_path)
                    self.logger.debug(f"Set GOOGLE_APPLICATION_CREDENTIALS: {file_path}")
                else:
                    self.logger.warning(f"Service account file not found: {file_path}, will try default credentials.")
            
            self.client = genai.Client(
                project=project_id,
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
    def chat_system_template(self) -> Optional[str]:
        template_value = self.get_param_value("chat system template")
        if template_value:
            to_lang = self.lang_map.get(self.lang_target, self.lang_target) # 목표 언어가 lang_map에 없을 경우를 대비
            return template_value.format(to_lang=to_lang)
        return None
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

    

    def _select_api_key_and_wait_if_needed(self) -> Optional[str]:
        """Selects an API key and waits if RPM limit is reached. Returns the key or None if no keys."""
        api_keys = self.multiple_keys_list
        if not api_keys:
            # 단일 키 사용 시 RPM 관리
            single_key = self.apikey
            if not single_key:
                self.logger.error("No API key configured.")
                return None
            self._respect_key_limit(single_key)
            return single_key

        # 여러 키가 있는 경우 로테이션 및 RPM 관리

        selected_key = None
        min_wait_time = float('inf')
        key_to_wait_for = None

        for i in range(len(api_keys)):
            index = self.current_key_index % len(api_keys)
            key_to_check = api_keys[index]
            
            count, start_time = self.key_usage.get(key_to_check, (0, time.time()))
            now = time.time()
            rpm_limit = int(self.get_param_value("max requests per minute"))

            if now - start_time >= 60:
                self.key_usage[key_to_check] = (0, now)
                count = 0
            
            if rpm_limit <= 0 or count < rpm_limit:
                selected_key = key_to_check
                self.key_usage[selected_key] = (count + 1, start_time if now - start_time < 60 else now)
                self.current_key_index = (index + 1) % len(api_keys)
                break
            
            else:
                # 이 키는 현재 사용할 수 없음, 대기 시간 계산
                wait_time_for_this_key = 60.1 - (now - start_time)
                if wait_time_for_this_key < min_wait_time:
                    min_wait_time = wait_time_for_this_key
                    key_to_wait_for = key_to_check
            self.current_key_index = (index + 1) % len(api_keys)

        if not selected_key and key_to_wait_for:
            # 모든 키가 RPM 제한에 도달, 가장 빨리 사용 가능해질 키를 위해 대기
            if min_wait_time > 0:
                self.logger.warning(
                    f"All keys reached RPM limit. Waiting for key {key_to_wait_for[:6]}... for {min_wait_time:.2f} seconds."
                )
                time.sleep(min_wait_time)
            # 대기 후 해당 키 사용량 초기화 및 선택
            self.key_usage[key_to_wait_for] = (0, time.time())
            selected_key = key_to_wait_for

            
            count, start_time = self.key_usage.get(selected_key, (0, time.time()))
            self.key_usage[selected_key] = (count + 1, start_time)

        return selected_key


    def _request_translation_thread_safe(self, prompt: str, chat_sample: Optional[List[str]]) -> str:
        with self.rpm_lock:
            self._respect_delay() # 전역 최소 요청 간격 준수

            if self.use_vertex_ai:
                # Vertex AI는 서비스 계정 사용, 키별 RPM 관리 불필요
                # _ensure_client에서 Vertex AI 클라이언트 처리
                pass
            else:
                # Gemini API는 키별 RPM 관리 필요
                api_key = self._select_api_key_and_wait_if_needed()
                
                if not api_key:
                    return "Error: No API key available."
                self._ensure_gemini_client(api_key) # 선택된 키로 클라이언트 설정
        try:
            self._ensure_client() # Vertex AI의 경우 여기서 클라이언트 초기화
        except ValueError as ve: # No API key available
            return str(ve)
        except Exception as e:
            self.logger.error(f"Error during client initialization: {e}")
            return f"Error: Client initialization failed - {e}"

        result = self._request_translation_with_chat_sample_google(prompt, chat_sample) # This now uses self.client
        if not isinstance(result, str):
            result = str(result)
        return result
    
    def _respect_key_limit(self, key: str):
        """특정 API 키의 RPM 제한을 확인하고 필요한 경우 대기합니다."""
        rpm_limit = int(self.get_param_value("max requests per minute"))
        if rpm_limit <= 0: # RPM 제한이 설정되지 않았으면 통과
            return

        count, start_time = self.key_usage.get(key, (0, time.time()))
        now = time.time()

        if now - start_time >= 60: # 1분이 지났으면 사용량 리셋
            self.key_usage[key] = (0, now)
            count = 0
        
        if count >= rpm_limit: # RPM 제한에 도달했다면
            wait_time = 60.1 - (now - start_time) # 0.1초 버퍼 추가
            masked_key = key[:6] + "*" * (len(key) - 6) if len(key) > 6 else key
            self.logger.warning(f"Key {masked_key} reached RPM limit ({rpm_limit}). Waiting {wait_time:.2f} seconds.")
            if wait_time > 0:
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time()) # 대기 후 사용량 리셋



    def _request_translation_with_chat_sample_google(
        self, prompt: str, chat_sample: Optional[List[str]]
    ) -> str:
        if not self.client:
            self.logger.error("Client is not initialized.")
            return "Error: Client not initialized."
        
        # 모델 이름 설정
        if self.use_vertex_ai:
            model_name = self.override_model or self.model # Vertex AI는 "models/" 접두사 없이 모델 ID 사용
        else:
            model_name = f"models/{self.override_model or self.model}" # Gemini API는 "models/" 접두사 필요
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

        # 시스템 프롬프트 설정
        system_instruction = self.chat_system_template

        # GenerateContentConfig 사용
        config = genai_types.GenerateContentConfig(
            system_instruction = system_instruction,
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

        # Vertex AI 사용 시에는 system_instruction을 GenerateContentConfig에 직접 전달하지 않고,
        # client.generate_content의 system_instruction 파라미터로 전달해야 할 수 있음.
        # google-genai SDK의 최신 버전에 따라 이 부분이 다를 수 있으므로 확인 필요.
        # 현재 google.generativeai.GenerativeModel.generate_content 에는 system_instruction 파라미터가 있음.
        # client.models.generate_content 에는 직접적인 system_instruction 파라미터가 없을 수 있음.
        # 이 경우, contents에 system role을 추가하거나, 모델 자체에 system prompt를 설정해야 함.
        # 여기서는 config에 포함시키는 것으로 가정. (만약 오류 발생 시, contents에 추가하는 방식 고려)

        try:
            # 올바른 API 호출 방법
            response = self.client.models.generate_content(
                model=model_name, # Vertex AI 사용 시 "models/" 접두사 없음
                contents=contents,
                config=config 
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
        num_queries = len(src_list)

        if not src_list:
            return []
        
        translations = [""] * num_queries # Initialize with correct size

        to_lang = self.lang_map.get(self.lang_target, self.lang_target) 
        chat_sample = self.chat_sample
        prompt_template_base = self.params["prompt template"]["value"].format(to_lang=to_lang).rstrip()

        def translate_single_query_thread_safe(query_idx_pair):
            idx, query_text = query_idx_pair
            if not query_text.strip(): # 빈 문자열은 번역하지 않음
                if 0 <= idx < num_queries:
                    translations[idx] = ""
                else:
                    self.logger.error(f"Critical Error: Index {idx} out of bounds for translations list of size {num_queries} (empty query).")
                return

            # 각 쿼리에 대한 프롬프트 생성
            prompt = f"{prompt_template_base}\n<|1|>{query_text}" # 단일 쿼리

            retry_attempt = 0
            translated_text = ""
            while retry_attempt < self.retry_attempts:
                try:
                    # _request_translation_thread_safe는 내부적으로 RPM 및 딜레이 관리
                    response_text = self._request_translation_thread_safe(prompt, chat_sample)
                    if not isinstance(response_text, str):
                        response_text = str(response_text)

                    # 단일 쿼리 응답 파싱
                    parsed_list = re.split(r"<\|\d+\|>", response_text)
                    if len(parsed_list) > 1 and parsed_list[-1].strip(): # 태그가 있고, 번역 내용이 있으면
                        translated_text = parsed_list[-1].strip()
                    elif not parsed_list[-1].strip() and len(parsed_list) > 1 and parsed_list[0].strip() and not parsed_list[0].startswith(prompt_template_base):
                        # <|1|> 태그 없이 내용만 반환된 경우 (예: "번역된 텍스트")
                        # 또는 태그는 있었으나 분리 후 마지막 요소가 비어있고, 첫 요소가 프롬프트가 아닌 번역문인 경우
                        translated_text = parsed_list[0].strip()
                    elif parsed_list[-1].strip(): # 태그 없이 내용만 반환된 경우
                         translated_text = parsed_list[-1].strip()
                    else: # 예상치 못한 형식 또는 빈 응답
                        # 응답이 프롬프트 자체를 포함하고 있다면, 실제 번역은 없다고 간주
                        if prompt_template_base in response_text and query_text in response_text:
                             self.logger.warning(f"Response for '{query_text[:30]}...' seems to be the prompt itself. Treating as empty translation.")
                             translated_text = "" # 또는 query_text로 설정하거나 오류 발생
                        else: # 그 외의 경우, 응답 전체를 사용하거나 오류 처리
                            translated_text = response_text.strip()

                    if not translated_text and query_text: # 번역 결과가 비었으면 오류로 간주 (필요시 원본 사용)
                        raise InvalidNumTranslations(f"Empty translation for query: {query_text}")

                    if 0 <= idx < num_queries:
                        translations[idx] = translated_text
                    else:
                        self.logger.error(f"Critical Error: Index {idx} out of bounds for translations list of size {num_queries} (success case).")
                    return
                except InvalidNumTranslations as e:
                    retry_attempt += 1
                    self.logger.warning(f"Invalid translation for query '{query_text[:30]}...': {e}. Attempt {retry_attempt}/{self.retry_attempts}")
                    if retry_attempt >= self.retry_attempts:
                        if 0 <= idx < num_queries:
                            translations[idx] = f"[번역 오류: 내용 없음]"
                        else:
                            self.logger.error(f"Critical Error: Index {idx} out of bounds for translations list of size {num_queries} (InvalidNumTranslations fallback).")
                        return
                except Exception as e:
                    retry_attempt += 1
                    self.logger.warning(f"Error translating query '{query_text[:30]}...': {e}. Attempt {retry_attempt}/{self.retry_attempts}")
                    if retry_attempt >= self.retry_attempts:
                        if 0 <= idx < num_queries: # Check bounds before assignment
                            translations[idx] = f"[번역 오류: {str(e)[:30]}]"
                        else:
                            # This case should ideally not be reached if idx is always correct.
                            # Logging it helps diagnose if idx is somehow corrupted.
                            self.logger.error(f"Critical Error: Index {idx} out of bounds for translations list of size {num_queries} (Exception fallback).")
                        return
                    time.sleep(self.retry_timeout)
            
            if 0 <= idx < num_queries: # Default fallback after retries
                translations[idx] = "[번역 실패]"
            else:
                self.logger.error(f"Critical Error: Index {idx} out of bounds for translations list of size {num_queries} (final fallback).")

        # ThreadPoolExecutor 설정
        num_keys = len(self.multiple_keys_list) if self.multiple_keys_list else 1
        rpm_per_key = int(self.get_param_value("max requests per minute"))
        
        # 워커 수 결정 로직: RPM이 매우 낮으면 병렬성 줄임, 아니면 키 개수만큼 (최대치 제한)
        # Ensure max_workers is at least 1
        if rpm_per_key <= 0: # 제한 없음
            max_workers = min(num_keys * 2, 10) # 키당 2개, 최대 10개 (임의의 값)
        elif rpm_per_key < 15: # 낮은 RPM
            max_workers = min(num_keys, 2) # 키 개수만큼 하되 최대 2개
        elif rpm_per_key < 60 : # 중간 RPM
            max_workers = min(num_keys, 5) # 키 개수만큼 하되 최대 5개
        else: # 높은 RPM
            max_workers = min(num_keys, 10) # 키 개수만큼 하되 최대 10개

        if self.use_vertex_ai and rpm_per_key > 0: # Vertex AI는 일반적으로 더 높은 처리량을 가짐
            max_workers = max(1, int(rpm_per_key / 2)) # Ensure at least 1 worker
        
        max_workers = max(1, max_workers) # Globally ensure at least 1 worker

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use num_queries (captured length) for range
            pairs = list(zip(range(num_queries), src_list))
            future_to_idx = {executor.submit(translate_single_query_thread_safe, pair): pair[0] 
                             for pair in pairs}
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    future.result() # 예외가 발생했다면 여기서 다시 발생 (이미 내부에서 로깅 및 처리)
                except Exception as exc:
                    self.logger.error(f'Query (idx {idx}) translation generated an exception: {exc}')
                    if 0 <= idx < num_queries and translations[idx] == "": # 아직 오류 메시지가 설정되지 않았다면
                        translations[idx] = "[번역 중 예외 발생]"
                    elif not (0 <= idx < num_queries):
                        self.logger.error(f"Critical Error: Index {idx} out of bounds for translations list of size {num_queries} (as_completed fallback).")

        # Gemini API는 현재 토큰 사용량 정보를 응답에 포함하지 않음
        # self.logger.info(f"Token count information is not available for Gemini API.")

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
