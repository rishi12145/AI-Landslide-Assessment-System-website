import logging
import threading
from typing import Generator, Dict, Any
from llm.base import LLMProvider

logger = logging.getLogger(__name__)

class TransformersProvider(LLMProvider):
    """
    Local HuggingFace Transformers LLM provider.
    Loads models (Qwen, Llama, Gemma, Phi) locally onto GPU or CPU.
    """
    
    def __init__(self, model_name: str, device: str = "cpu", torch_dtype: str = "float16"):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.tokenizer = None
        self.model = None
        
    def _lazy_init(self) -> None:
        """Initializes the model and tokenizer only when inference is first requested."""
        if self.model is not None:
            return
            
        logger.info(f"Loading local HuggingFace model: {self.model_name} on {self.device}")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Parse torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        selected_dtype = dtype_map.get(self.torch_dtype, torch.float32)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=selected_dtype,
            trust_remote_code=True
        )
        logger.info("Local HuggingFace model loaded successfully.")

    def generate(self, prompt: str, params: Dict[str, Any]) -> str:
        self._lazy_init()
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        gen_config = {
            "max_new_tokens": params.get("max_tokens", 2048),
            "temperature": params.get("temperature", 0.2),
            "top_p": params.get("top_p", 0.9),
            "top_k": params.get("top_k", 50),
            "do_sample": params.get("temperature", 0.2) > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_config)
            
        # Slice inputs out of generated tokens
        input_len = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_len:]
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
    def generate_stream(self, prompt: str, params: Dict[str, Any]) -> Generator[str, None, None]:
        self._lazy_init()
        from transformers import TextIteratorStreamer
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        gen_config = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": params.get("max_tokens", 2048),
            "temperature": params.get("temperature", 0.2),
            "top_p": params.get("top_p", 0.9),
            "top_k": params.get("top_k", 50),
            "do_sample": params.get("temperature", 0.2) > 0.0,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Start generation in a separate thread so streamer is non-blocking
        generation_thread = threading.Thread(target=self.model.generate, kwargs=gen_config)
        generation_thread.start()
        
        for new_text in streamer:
            yield new_text
            
        generation_thread.join()
