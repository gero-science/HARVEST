import time
import asyncio
import aiohttp
import random
import json
import os
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List


# Global list for collecting failed LLM requests in debug mode
GLOBAL_FAILED_REQUESTS_LOG = []


# Constants for improved error handling
class ErrorCategory:
    """Error categories for different retry handling"""
    FATAL = "fatal"  # Do not retry
    RETRYABLE = "retryable"  # Can retry
    RATE_LIMITED = "rate_limited"  # Rate limit exceeded
    SERVER_ERROR = "server_error"  # Server errors


def classify_error(status_code: int) -> str:
    """Classify an error to determine retry strategy"""
    if status_code in [400, 401, 402, 403, 422]:  # Added 402 Payment Required
        return ErrorCategory.FATAL
    elif status_code == 429:
        return ErrorCategory.RATE_LIMITED
    elif status_code >= 500:
        return ErrorCategory.SERVER_ERROR
    elif status_code == 408:
        return ErrorCategory.RETRYABLE
    else:
        return ErrorCategory.RETRYABLE


def calculate_delay(attempt: int, base_delay: float, error_category: str) -> float:
    """Compute delay based on error type and attempt number"""
    if error_category == ErrorCategory.RATE_LIMITED:
        # Exponential backoff for rate limiting with jitter
        exponential_delay = base_delay * (2 ** attempt)
        jitter = random.uniform(0.1, 0.3) * exponential_delay
        return min(exponential_delay + jitter, 60.0)  # Max 60 seconds
    elif error_category == ErrorCategory.SERVER_ERROR:
        # Increased delay for server errors
        return min(base_delay * 3 * (1.5 ** attempt), 30.0)  # Max 30 seconds
    else:
        # Standard delay
        return base_delay * (1 + attempt * 0.5)


def parse_error_metadata(response_data: Dict[str, Any]) -> Tuple[str, Optional[Dict]]:
    """Extract detailed error information from the response"""
    if not response_data or "error" not in response_data:
        return "", None
    
    error_details = response_data["error"]
    message = error_details.get("message", "Unknown error")
    metadata = error_details.get("metadata")
    
    return message, metadata


def log_error_details(logger, status_code: int, message: str, metadata: Optional[Dict], attempt: int):
    """Log detailed error information"""
    # Add human-readable status code description
    status_descriptions = {
        400: "Bad Request", 401: "Unauthorized", 402: "Payment Required", 
        403: "Forbidden", 404: "Not Found", 408: "Request Timeout",
        422: "Unprocessable Entity", 429: "Too Many Requests",
        500: "Internal Server Error", 502: "Bad Gateway", 
        503: "Service Unavailable", 504: "Gateway Timeout"
    }
    
    status_desc = status_descriptions.get(status_code, "Unknown Error")
    logger.error(f"API Error (attempt {attempt}): HTTP {status_code} ({status_desc}) - {message}")
    
    if metadata:
        if "reasons" in metadata and "flagged_input" in metadata:
            # Moderation error
            reasons = ", ".join(metadata["reasons"])
            flagged_text = metadata["flagged_input"]
            provider = metadata.get("provider_name", "unknown")
            logger.error(f"Moderation Error - Provider: {provider}, Reasons: {reasons}")
            logger.error(f"Flagged input: {flagged_text}")
        elif "provider_name" in metadata and "raw" in metadata:
            # Provider error
            provider = metadata["provider_name"]
            raw_error = metadata["raw"]
            logger.error(f"Provider Error - {provider}: {raw_error}")
        else:
            # Other metadata
            logger.error(f"Error metadata: {metadata}")


def is_successful_response_with_error(response_data: Dict[str, Any]) -> bool:
    """Check whether a successful (200 OK) response contains an error"""
    return response_data is not None and "error" in response_data


class LLM:
    def __init__(
        self,
        api_retry_attempts,
        api_key,
        api_base_url,
        temperature,
        max_tokens_response,
        model_name,
        api_retry_delay,
        logger,
        provider_sort="throughput",
        provider_quantizations="",
        provider_ignore=None,
        debug_mode=False,
        debug_output_dir=None,
        provider=None
    ):
        self.api_retry_attempts = api_retry_attempts
        self.api_retry_delay = api_retry_delay
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.temperature = temperature
        self.max_tokens_response = max_tokens_response
        self.model_name = model_name
        self.logger = logger
        self.provider_sort = provider_sort
        self.provider_quantizations = provider_quantizations
        self.provider_ignore = provider_ignore if provider_ignore else []
        self.debug_mode = debug_mode
        self.debug_output_dir = debug_output_dir
        self.provider = provider
        self.failed_requests_count = 0  # Failed request counter (not kept in memory)
        
        # Session management for connection reuse
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

    async def get_session(self) -> aiohttp.ClientSession:
        """
        Lazy initialization of a reusable session.
        Creates one session with connection pooling for all requests.
        """
        if self._session is None or self._session.closed:
            self._connector = aiohttp.TCPConnector(
                limit=100,           # Max total connections
                limit_per_host=100,  # Max connections per host (OpenRouter)
                ttl_dns_cache=300,   # DNS cache 5 min
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            timeout = aiohttp.ClientTimeout(total=900)  # 15 minute timeout
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout
            )
        return self._session

    async def close(self):
        """Close the session on shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._connector and not self._connector.closed:
            await self._connector.close()
        self._session = None
        self._connector = None

    def log_failed_request(self, user_prompt: str, system_prompt: str, error_type: str, error_message: str, status_code: int = None, attempt: int = None, request_data: dict = None, patent_id: str = None):
        """
        Log a failed request directly to file (do not accumulate in memory).
        
        Args:
            user_prompt: User prompt
            system_prompt: System prompt
            error_type: Error type (timeout, http_error, json_error, network_error, etc.)
            error_message: Error message
            status_code: HTTP status code (if applicable)
            attempt: Attempt number
            request_data: Request payload
            patent_id: Patent ID for tracing
        """
        if not self.debug_mode or not self.debug_output_dir:
            return
            
        failed_request = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'error_type': error_type,
            'error_message': error_message,
            'status_code': status_code,
            'attempt': attempt,
            'patent_id': patent_id,  # Add patent_id for tracing
            'user_prompt': user_prompt if user_prompt else None,  # Save full prompts for debug
            'system_prompt': system_prompt if system_prompt else None,
            'user_prompt_length': len(user_prompt) if user_prompt else 0,
            'system_prompt_length': len(system_prompt) if system_prompt else 0,
            'request_data': {
                'temperature': request_data.get('temperature') if request_data else self.temperature,
                'max_tokens': request_data.get('max_tokens') if request_data else self.max_tokens_response,
                'model': request_data.get('model') if request_data else self.model_name,
            },
            'traceback': traceback.format_exc() if error_type in ['network_error', 'unexpected_error'] else None
        }
        
        # Write to file immediately (JSONL format, do not buffer in memory)
        try:
            failed_requests_file = os.path.join(self.debug_output_dir, "llm_failed_requests.jsonl")
            with open(failed_requests_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(failed_request, ensure_ascii=False, default=str) + '\n')
            self.failed_requests_count += 1
        except Exception as e:
            self.logger.warning(f"Failed to write failed request to file: {e}")

    def save_failed_requests_log(self):
        """Report the number of failed requests (records are already in the file)."""
        if not self.debug_mode or not self.debug_output_dir:
            return
        
        if self.failed_requests_count > 0:
            failed_requests_file = os.path.join(self.debug_output_dir, "llm_failed_requests.jsonl")
            self.logger.info(f"Wrote {self.failed_requests_count} failed LLM requests to {failed_requests_file}")

    @classmethod
    def from_config(cls, config, logger, debug_mode=False, debug_output_dir=None):
        """
        Factory method to create an LLM from a configuration object.
        Automatically extracts all required parameters from the config.
        
        Args:
            config: Configuration object with API_KEY, MODEL_NAME, etc.
            logger: Logger object
            debug_mode: Enable debug mode for logging failed requests
            debug_output_dir: Directory for debug output files
            
        Returns:
            LLM: Configured LLM instance
        """
        # Credentials are checked here rather than when llm.config is imported:
        # that module is reachable from every pipeline entry point, including the
        # LLM-free stages, but this factory is only reached when a client is
        # actually about to be used. Stages that call an API therefore still fail
        # immediately, before any patent is processed.
        missing = [
            name
            for name, value in (
                ("LLM_KEY", config.API_KEY),
                ("LLM_URL", config.API_BASE_URL),
                ("LLM_MODEL", config.MODEL_NAME),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing LLM configuration: {', '.join(missing)}. Set them in .env "
                "(see .env.example). Stages that do not call an LLM, such as "
                "--stages export,postprocess, do not need these."
            )

        # Parse provider_ignore from string to list
        provider_ignore_str = getattr(config, 'PROVIDER_IGNORE', '')
        provider_ignore = [p.strip() for p in provider_ignore_str.split(',') if p.strip()] if provider_ignore_str else []
        
        return cls(
            api_retry_attempts=getattr(config, 'API_RETRY_ATTEMPTS', 3),
            api_key=config.API_KEY,
            api_base_url=config.API_BASE_URL,
            temperature=config.TEMPERATURE,
            max_tokens_response=config.MAX_TOKENS_RESPONSE,
            model_name=config.MODEL_NAME,
            api_retry_delay=getattr(config, 'API_RETRY_DELAY', 1),
            logger=logger,
            provider_sort=getattr(config, 'PROVIDER_SORT', 'throughput'),
            provider_quantizations=getattr(config, 'PROVIDER_QUANTIZATIONS', 'bf16'),
            provider_ignore=provider_ignore,
            debug_mode=debug_mode,
            debug_output_dir=debug_output_dir,
            provider=config.PROVIDER,
        )

    @classmethod
    def create_default(cls, logger, debug_mode=False, debug_output_dir=None):
        """
        Create an LLM with configuration loaded automatically from environment variables.
        Convenient for quick setup without passing config explicitly.
        
        Args:
            logger: Logger object
            debug_mode: Enable debug mode for logging failed requests
            debug_output_dir: Directory for debug output files
            
        Returns:
            LLM: Configured LLM instance with default configuration
        """
        # Imported here, not at module scope: llm.config reads os.environ
        # eagerly, and importing this client must not require a populated .env.
        from .config import ConfigLLM

        config = ConfigLLM()
        return cls.from_config(config, logger, debug_mode, debug_output_dir)

    async def async_call_llm(
        self, user_prompt: str = None, system_prompt: str | None = None, patent_id: str = None, response_format: dict | None = None, messages: list[dict] | None = None, max_tokens: int | None = None
    ) -> tuple[str | None, dict | None]:
        """
        Improved async LLM call with advanced error handling.

        Features:
        - Exponential backoff for rate limiting (429)
        - Increased delay for server errors (5xx)
        - Detailed logging of error metadata
        - Error handling inside successful responses (200 OK)
        - Improved timeout handling
        - Structured outputs support (JSON Schema)
        - Full messages array support for chat format (prompt caching)

        Args:
            user_prompt: User prompt (used when messages=None)
            system_prompt: System prompt (used when messages=None)
            patent_id: Patent ID for error tracing (optional)
            response_format: Response format for structured outputs (optional)
            messages: Full message array for chat format (optional)
            max_tokens: Maximum tokens in the response (optional; overrides self.max_tokens_response)

        Returns:
            tuple: (response_content, usage_stats) where usage_stats contains token information
        """
        # Two modes: messages array or user_prompt/system_prompt
        if messages:
            # Use provided messages array (for cached chat)
            pass
        else:
            # Build messages from user_prompt/system_prompt (backward compatibility)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
            
            # Ensure at least one prompt or messages is provided
            if not messages:
                error_msg = "No prompts provided: both user_prompt and system_prompt are None/empty, and messages is None/empty"
                self.logger.error(error_msg)
                self.log_failed_request(user_prompt, system_prompt, "validation_error", error_msg, None, 0, None, patent_id)
                return None, None
            
        # Prepare quantizations list
        quantizations = []
        if self.provider_quantizations:
            quantizations = [q.strip() for q in self.provider_quantizations.split(",") if q.strip()]
        
        # Prepare provider config
        if self.provider:
            provider_config = {"order": [self.provider]}
        else:
            provider_config = {"sort": self.provider_sort} if self.provider_sort else {}
        if quantizations:
            provider_config["quantizations"] = quantizations
        if self.provider_ignore:
            provider_config["ignore"] = self.provider_ignore
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        request_data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature
        }
        
        # Add usage parameter only for OpenRouter
        if "openrouter" in self.api_base_url:
            request_data["usage"] = {"include": True}
        
        # Add max_tokens only if it's set (not None)
        # Priority: method parameter > self.max_tokens_response
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens_response
        if effective_max_tokens is not None:
            request_data["max_tokens"] = effective_max_tokens
        
        # Add response_format for structured outputs if provided
        if response_format:
            request_data["response_format"] = response_format

        # Add provider config only for OpenRouter
        if "openrouter" in self.api_base_url:
            request_data["provider"] = provider_config
        
        url = f"{self.api_base_url.rstrip('/')}/chat/completions"
        
        for attempt in range(self.api_retry_attempts):
            try:
                session = await self.get_session()
                async with session.post(url, headers=headers, json=request_data) as response:
                    
                    # Handle successful responses
                    if response.status == 200:
                        try:
                            response_data = await response.json()
                        except Exception as json_error:
                            response_text = await response.text()
                            self.logger.error(f"Failed to parse JSON response (HTTP 200) on attempt {attempt + 1}: {json_error}")
                            self.logger.error(f"Raw response text: {response_text[:500]}{'...' if len(response_text) > 500 else ''}")
                            
                            # Debug logging
                            self.log_failed_request(user_prompt, system_prompt, "json_parse_error", 
                                f"Failed to parse JSON response: {json_error}. Raw response: {response_text[:200]}", 
                                200, attempt + 1, request_data, patent_id)
                            
                            if attempt < self.api_retry_attempts - 1:
                                await asyncio.sleep(calculate_delay(attempt, self.api_retry_delay, ErrorCategory.RETRYABLE))
                                continue
                            else:
                                return None, None
                        
                        # Check for an error inside a successful response (200 OK)
                        if is_successful_response_with_error(response_data):
                            error_message, error_metadata = parse_error_metadata(response_data)
                            embedded_error_code = response_data["error"].get("code", 500)
                            
                            log_error_details(self.logger, embedded_error_code, error_message, error_metadata, attempt + 1)
                            
                            error_category = classify_error(embedded_error_code)
                            
                            # Debug logging
                            self.log_failed_request(user_prompt, system_prompt, "embedded_error_in_200", 
                                error_message, embedded_error_code, attempt + 1, request_data, patent_id)
                            
                            if error_category == ErrorCategory.FATAL:
                                self.logger.error(f"Fatal error in 200 OK response, not retrying: {error_message}")
                                return None, None
                            
                            if attempt < self.api_retry_attempts - 1:
                                delay = calculate_delay(attempt, self.api_retry_delay, error_category)
                                self.logger.info(f"Retrying after {delay:.2f}s due to embedded error...")
                                await asyncio.sleep(delay)
                                continue
                            else:
                                return None, None
                        
                        # Validate response
                        if response_data is None:
                            self.logger.error(f"API returned None as JSON response (HTTP 200) on attempt {attempt + 1}")
                            self.logger.error(f"This may indicate an empty response or server issue")
                            
                            # Debug logging
                            self.log_failed_request(user_prompt, system_prompt, "null_response", 
                                "API returned None as JSON response", 200, attempt + 1, request_data, patent_id)
                            
                            if attempt < self.api_retry_attempts - 1:
                                await asyncio.sleep(calculate_delay(attempt, self.api_retry_delay, ErrorCategory.RETRYABLE))
                                continue
                            else:
                                return None, None
                        
                        content = None
                        usage = None
                        finish_reason = None
                        
                        # Safely extract content and finish_reason
                        try:
                            if response_data.get("choices") and response_data["choices"][0].get("message"):
                                content = response_data["choices"][0]["message"]["content"]
                                # Extract finish_reason from choices[0]
                                finish_reason = response_data["choices"][0].get("finish_reason", "stop")
                            else:
                                self.logger.warning(f"No content found in API response (HTTP 200) on attempt {attempt + 1}")
                                self.logger.debug(f"Response structure: {response_data}")
                        except (KeyError, IndexError, TypeError) as e:
                            self.logger.error(f"Failed to extract content from response (HTTP 200) on attempt {attempt + 1}: {e}")
                            self.logger.debug(f"Response data: {response_data}")
                        
                        # Safely extract usage
                        try:
                            if response_data.get("usage"):
                                usage = response_data["usage"]
                                # Add finish_reason to usage for convenience
                                if finish_reason:
                                    usage["finish_reason"] = finish_reason
                                # DEBUG: Log full usage info to verify caching
                                self.logger.debug(f"Full usage response: {json.dumps(usage, indent=2)}")
                            else:
                                self.logger.warning(f"API did not return usage information (HTTP 200) on attempt {attempt + 1}")
                                self.logger.debug(f"Response keys: {list(response_data.keys()) if response_data else 'None'}")
                                # If usage is missing, create a minimal dict with finish_reason
                                if finish_reason:
                                    usage = {"finish_reason": finish_reason}
                        except (KeyError, TypeError) as e:
                            self.logger.error(f"Failed to extract usage from response (HTTP 200) on attempt {attempt + 1}: {e}")
                            self.logger.debug(f"Response data: {response_data}")

                        return content, usage
                    
                    # Handle error responses
                    else:
                        try:
                            error_data = await response.json()
                            error_message, error_metadata = parse_error_metadata(error_data)
                        except:
                            # If JSON parsing fails, use response text
                            error_text = await response.text()
                            error_message = f"Non-JSON error response: {error_text[:500]}"
                            error_metadata = None
                        
                        # Detailed error logging
                        log_error_details(self.logger, response.status, error_message, error_metadata, attempt + 1)
                        self.logger.debug(f"Request URL: {url}")
                        
                        # Classify error
                        error_category = classify_error(response.status)
                        
                        # Debug logging
                        self.log_failed_request(user_prompt, system_prompt, "http_error", 
                            error_message, response.status, attempt + 1, request_data, patent_id)
                        
                        # Do not retry fatal errors
                        if error_category == ErrorCategory.FATAL:
                            self.logger.error(f"Fatal error {response.status}, not retrying: {error_message}")
                            return None, None
                        
                        # For other errors use the appropriate delay
                        if attempt < self.api_retry_attempts - 1:
                            delay = calculate_delay(attempt, self.api_retry_delay, error_category)
                            self.logger.info(f"Retrying after {delay:.2f}s (error category: {error_category})...")
                            await asyncio.sleep(delay)
                            
            except asyncio.CancelledError:
                self.logger.warning(f"API call was cancelled on attempt {attempt + 1}")
                raise  # Re-raise CancelledError
            except asyncio.TimeoutError:
                self.logger.error(f"Request timeout (300s) on attempt {attempt + 1}")
                self.logger.error(f"URL: {url}")
                
                # Debug logging
                self.log_failed_request(user_prompt, system_prompt, "timeout_error", 
                    "Request timeout (300s)", None, attempt + 1, request_data, patent_id)
                
                if attempt < self.api_retry_attempts - 1:
                    delay = calculate_delay(attempt, self.api_retry_delay, ErrorCategory.RETRYABLE)
                    self.logger.info(f"Retrying after timeout, delay: {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.api_retry_attempts} timeout retry attempts failed.")
                    return None, None
            except aiohttp.ClientError as e:
                self.logger.error(f"Network error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                self.logger.error(f"URL: {url}")
                
                # Debug logging
                self.log_failed_request(user_prompt, system_prompt, "network_error", 
                    f"{type(e).__name__}: {e}", None, attempt + 1, request_data, patent_id)
                
                if attempt < self.api_retry_attempts - 1:
                    delay = calculate_delay(attempt, self.api_retry_delay, ErrorCategory.RETRYABLE)
                    self.logger.info(f"Retrying after network error, delay: {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.api_retry_attempts} network retry attempts failed.")
                    return None, None
            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                self.logger.error(f"Request URL: {url}")
                self.logger.debug(f"Exception details:", exc_info=True)
                
                # Debug logging
                self.log_failed_request(user_prompt, system_prompt, "unexpected_error", 
                    f"{type(e).__name__}: {e}", None, attempt + 1, request_data, patent_id)
                
                if attempt < self.api_retry_attempts - 1:
                    delay = calculate_delay(attempt, self.api_retry_delay, ErrorCategory.RETRYABLE)
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"All {self.api_retry_attempts} retry attempts failed due to unexpected errors.")
                    return None, None
        
        # If we reach here, all attempts were exhausted
        self.logger.error(f"Failed to get response from API after {self.api_retry_attempts} attempts")
        
        # Debug logging for final failure
        self.log_failed_request(user_prompt, system_prompt, "all_attempts_failed", 
            f"Failed to get response from API after {self.api_retry_attempts} attempts", 
            None, self.api_retry_attempts, request_data, patent_id)
        
        return None, None


def save_global_failed_requests_log(output_dir: str) -> None:
    """
    Deprecated function - records are now written directly to file.
    Kept for backward compatibility.
    """
    # Records are now written incrementally to llm_failed_requests.jsonl
    pass


def clear_global_failed_requests_log() -> None:
    """Deprecated function - global list is no longer used."""
    pass
