# ── modules/llm_wrapper.py ──────────────────────────────

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline  # ✅ 변경된 임포트
import torch

class LLMWrapper:
    def __init__(self, model_id: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"):
        self.device = 0 if torch.cuda.is_available() else -1

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True
        ).to("cuda" if self.device == 0 else "cpu")

        gen_pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1
        )

        self.pipeline = HuggingFacePipeline(pipeline=gen_pipeline)
