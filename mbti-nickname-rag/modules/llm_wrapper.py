# modules/llm_wrapper.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from langchain_community.llms import HuggingFacePipeline
import torch
from typing import Dict, Any, List

class LLMWrapper:
    def __init__(self, model_id: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"):
        self.device = 0 if torch.cuda.is_available() else -1

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True
        ).to("cuda" if self.device == 0 else "cpu")

        gen_pipeline = hf_pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        self.pipeline = HuggingFacePipeline(pipeline=gen_pipeline)


class HyperclovaxChat:
    """
    HyperCLOVA-X 인스트럭션 모델을 위한 채팅 래퍼.
    LangChain의 create_stuff_documents_chain 등이
    내부적으로 .invoke(inputs) 인터페이스를 기대하므로
    invoke()를 구현해줍니다.
    """
    def __init__(self, model_id: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model     = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

        self.device = 0 if torch.cuda.is_available() else -1
        if self.device == 0:
            self.model.to("cuda")

        gen_pipeline = hf_pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            top_p=0.95
        )
        self.pipeline = HuggingFacePipeline(pipeline=gen_pipeline)

    def __call__(
        self,
        user_question: str,
        tool_list: List[str] = None,
        system_msg: str = None
    ) -> str:
        # 모든 메시지 content를 str()로 감싸서 PromptValue 혼합 방지
        chat = []
        tool_content = "\n".join(str(x) for x in (tool_list or []))
        system_content = (
            str(system_msg)
            if system_msg is not None
            else (
                "- AI 언어모델의 이름은 \"CLOVA X\" 이며 네이버에서 만들었다.\n"
                "- 오늘은 2025년 04월 24일(목)이다."
            )
        )
        chat.append({"role": "tool_list", "content": tool_content})
        chat.append({"role": "system",    "content": system_content})
        chat.append({"role": "user",      "content": str(user_question)})

        # 1) 토크나이저 내장 채팅 템플릿 적용
        inputs = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # 2) apply_chat_template 결과가 단일 Tensor인 경우 꼭 매핑 형태로 변환
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        elif not isinstance(inputs, dict):
            try:
                inputs = dict(inputs)
            except Exception:
                inputs = {"input_ids": inputs}

        # 3) GPU 사용 시 장치에 맞게 이동
        if self.device == 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # 4) 텍스트 생성
        out_ids = self.model.generate(**inputs, eos_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        return text

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """
        LangChain 체인이 invoke()로 호출할 때 사용하는 메서드.
        inputs = {"context": List[str], "question": str}
        """
        context = inputs["context"]
        question = inputs["question"]

        # 리스트면 한 덩어리로
        context_str = "\n".join(context) if isinstance(context, list) else str(context)

        # 내부 __call__에 전달할 단일 프롬프트 구성
        user_prompt = (
            "당신은 익명 별명 추천 전문가입니다.\n"
            "아래 내용을 참고해서 질문에 답해주세요:\n\n"
            f"=== 참고 정보 ===\n{context_str}\n\n"
            f"질문: {question}\n\n"
            "위 정보를 바탕으로, 사용자에게 어울리는 별명 5개를\n"
            "번호 매기기 형식으로 추천하세요.\n"
            "별명은 한국어로 짧고 기억하기 쉽게 만들어주세요."
        )

        output = self.__call__(user_question=user_prompt)
        return {"text": output}
