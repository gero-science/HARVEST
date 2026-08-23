"""Module for error handling and classification in the pipeline."""


def classify_error(error_message: str) -> str:
    """
    Classify an error by type for statistics.
    
    Args:
        error_message: String representation of the error
        
    Returns:
        str: Error type (rate_limit_errors, timeout_errors, api_errors, 
             parse_errors, llm_failure_errors, other_errors)
    """
    error_str = error_message.lower()
    
    if "llm failure" in error_str or "no response received after all retry attempts" in error_str:
        return "llm_failure_errors"
    elif "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
        return "rate_limit_errors"
    elif "timeout" in error_str or "timed out" in error_str or "408" in error_str:
        return "timeout_errors"
    elif any(code in error_str for code in ["400", "401", "403", "500", "502", "503", "504"]) or "api" in error_str:
        return "api_errors"
    elif "json" in error_str or "parse" in error_str or "decode" in error_str:
        return "parse_errors"
    else:
        return "other_errors"


def create_empty_error_stats() -> dict:
    """Create an empty error statistics structure."""
    return {
        "api_errors": 0,
        "timeout_errors": 0,
        "rate_limit_errors": 0,
        "parse_errors": 0,
        "llm_failure_errors": 0,  # New type for critical LLM failures
        "other_errors": 0
    }


