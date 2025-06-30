# llm_caller.py (클로바 버전)

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
import torch

# ✅ 환경 변수 로딩
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

# ✅ 모델 로딩 (예외 처리 포함)
generator = None
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN)

    # 자동 감지: GPU 있으면 0, 없으면 -1
    try:
        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            print("[info] GPU(CUDA) 사용: device=0")
        else:
            print("[info] CPU 사용: device=-1")
    except Exception as e:
        print(f"[warn] device 자동 감지 실패, CPU로 강제 설정: {e}")
        device = -1
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    print("[info] 모델 로딩 완료")
except Exception as e:
    print(f"[error] 모델 로딩 실패: {e}")

# ✅ LLM 호출 함수
def call_llm(prompt: str) -> str:
    if generator is None:
        raise RuntimeError("모델이 초기화되지 않았습니다.")

    try:
        result = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        else:
            raise ValueError("예상치 못한 응답 형식입니다.")
    except Exception as e:
        print(f"[error] LLM 호출 실패: {e}")
        raise

