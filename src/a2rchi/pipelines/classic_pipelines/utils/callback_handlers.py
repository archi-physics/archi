import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import BaseCallbackHandler

from src.utils.runtime_config_loader import load_runtime_config

data_path = load_runtime_config()["global"]["DATA_PATH"]

class PromptLogger(BaseCallbackHandler):
    """Lightweight callback handler to log prompts and responses to file"""
    
    def __init__(
            self,
            logfile: str
        ):
        self.logfile = logfile
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Log the prompt when LLM starts"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.logfile, 'a', encoding='utf-8') as f:
            f.write("-" * 41)
            f.write(f"\n[{timestamp}] Prompt sent to LLM:\n")
            f.write("-" * 41 + "\n\n")
            for prompt in prompts:
                f.write(f"{prompt}\n\n\n")
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Log the response when LLM ends"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.logfile, 'a', encoding='utf-8') as f:
            f.write("-" * 35)
            f.write(f"\n[{timestamp}] LLM Response:\n")
            f.write("-" * 35 + "\n\n")
            
            # handle different response formats
            if hasattr(response, 'generations'):
                for generation_list in response.generations:
                    for generation in generation_list:
                        if hasattr(generation, 'text'):
                            f.write(f"{generation.text}\n\n\n")
                        elif hasattr(generation, 'message'):
                            f.write(f"{generation.message.content}\n\n\n")
            else:
                f.write(f"{response}\n\n\n")
            
            f.write("=" * 96 + "\n\n\n")

    
