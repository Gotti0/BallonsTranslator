import re
import time
import yaml
import traceback
from typing import List, Dict, Optional

import httpx
import google.genai as genai # 수정된 임포트 구문

from .base import BaseTranslator, register_translator


class InvalidNumTranslations(Exception):
    pass


@register_translator("Gemini")
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
                "gemini-1.5-pro-latest",
                "gemini-1.5-flash-latest",
                "gemini-1.0-pro",
            ],
            "value": "gemini-1.5-flash-latest",
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
        self.model_client = None # Gemini 모델 클라이언트
        self._initialize_client()

    def _initialize_client(self):
        if self.proxy:
            self.logger.info(
                f"Proxy configured: {self.proxy}. Ensure your environment (HTTP_PROXY, HTTPS_PROXY) is set up for google-generativeai SDK."
            )

        api_keys = self.multiple_keys_list
        api_key_to_use = api_keys[0] if api_keys else self.apikey

        if not api_key_to_use:
            self.logger.warning(
                "No API key provided. Please set either 'apikey' or 'multiple_keys'."
            )
            self.model_client = None
            return

        masked_key = api_key_to_use[:6] + "*" * (len(api_key_to_use) - 6)
        self.logger.debug(f"Configuring Google GenAI with initial API key: {masked_key}")
        
        try:
            genai.configure(api_key=api_key_to_use) # 실제 요청 시점에 키 변경 가능
            model_name = self.override_model or self.model
            self.model_client = genai.GenerativeModel(model_name)
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client or model: {e}")
            self.model_client = None


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

        current_api_key = self._select_api_key()
        if not current_api_key:
            self.logger.error("No API key available for translation.")
            return "Error: No API key provided."

        # API 키 변경 시 genai 재설정
        # genai.configure는 전역 설정이므로, 현재 설정된 키와 다를 경우에만 호출
        # (주의: genai 라이브러리에 현재 설정된 키를 직접 가져오는 API가 없을 수 있음.
        #  이 경우, 마지막으로 설정한 키를 self에 저장하여 비교하거나, 매번 호출)
        if not hasattr(self, '_current_genai_api_key') or self._current_genai_api_key != current_api_key:
            try:
                genai.configure(api_key=current_api_key)
                self._current_genai_api_key = current_api_key
                masked_key = current_api_key[:6] + "*" * (len(current_api_key) - 6)
                self.logger.debug(f"Configured GenAI with API key: {masked_key}")
            except Exception as e:
                self.logger.error(f"Failed to configure GenAI with API key: {e}")
                return f"Error: Failed to configure API key."

        model_name_from_param = self.override_model or self.model
        
        # 모델 클라이언트 재설정 (모델 이름이 변경되었을 수 있으므로)
        if not self.model_client or self.model_client.model_name != model_name_from_param:
            try:
                self.model_client = genai.GenerativeModel(model_name_from_param)
                self.logger.debug(f"Initialized Gemini model: {model_name_from_param}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini model {model_name_from_param}: {e}")
                return f"Error: Failed to initialize model {model_name_from_param}."
        
        result = self._request_translation_with_chat_sample_google(
            prompt, chat_sample
        )

        if not isinstance(result, str):
            result = str(result)
        return result

    def _request_translation_with_chat_sample_google(
        self, prompt: str, chat_sample: Optional[List[str]]
    ) -> str:
        if not self.model_client:
            self.logger.error("Gemini model client is not initialized.")
            return "Error: Model client not initialized."

        # Gemini API는 messages 대신 contents를 사용하고, role도 user/model로 다름
        gemini_contents = []
        
        # 시스템 프롬프트 추가 (첫 번째 user 메시지에 결합)
        current_user_prompt_parts = [self.chat_system_template, prompt]

        if chat_sample:
            # 샘플 유저 메시지
            gemini_contents.append({"role": "user", "parts": [{"text": chat_sample[0]}]})
            # 샘플 모델 응답
            gemini_contents.append({"role": "model", "parts": [{"text": chat_sample[1]}]})
        
        # 실제 번역 요청 프롬프트 (시스템 프롬프트와 결합됨)
        gemini_contents.append({"role": "user", "parts": [{"text": "\n".join(current_user_prompt_parts)}]})


        generation_config = genai.types.GenerationConfig(
            max_output_tokens=self.max_tokens, # Gemini는 max_output_tokens 사용
            temperature=self.temperature, 
            top_p=self.top_p 
        )
        
        safety_settings = [ # 모든 안전 설정을 비활성화 (번역 작업에 불필요)
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]

        try:
            response = self.model_client.generate_content(
                contents=gemini_contents,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        except Exception as e:
            self.logger.error(f"Gemini API request failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: API request failed - {e}"


        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        
        self.logger.warning(f"No content found in Gemini response. Prompt: {prompt[:100]}... Response: {response}")
        return ""


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
            "proxy",
            "apikey",
            "model",
            "override_model",
            "multiple_keys",
        ]:
            self._initialize_client()

