from flask import Flask, request, jsonify
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline # LLM 생성 파이프라인
import os

app = Flask(__name__)

# --- 전역 변수 및 설정 ---
# 학사 정보 청크 데이터를 여기에 삽입
DOCUMENT_CHUNKS = [
    "우석대학교의 2025학년도 1학기 등록금 납부 기간은 2월 20일부터 2월 24일까지입니다.",
    "등록금을 미납할 경우 학점 취소 및 제적 처리될 수 있습니다.",
    "2024년 11월에는 수강신청 안내 및 졸업 사정회가 예정되어 있습니다.",
    "셔틀버스는 삼례캠퍼스와 전주역, 익산역을 운행하며 자세한 시간표는 학교 포털 공지사항을 확인하세요."
]
MODEL = None       # Sentence Transformer (검색용)
FAISS_INDEX = None # Faiss 인덱스
GENERATOR = None   # LLM 생성 파이프라인 (생성용)
VECTOR_DIMENSION = 0

def retrieve_relevant_context(query, index, model, chunks, k=2):
    """Faiss 인덱스에서 가장 유사한 문맥을 검색합니다."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding, k) 
    retrieved_chunks = [chunks[i] for i in I[0]]
    return retrieved_chunks

def initialize_search_engine():
    """서버 시작 시 모든 모델과 인덱스를 초기화합니다."""
    global MODEL, FAISS_INDEX, VECTOR_DIMENSION, GENERATOR
    
    try:
        # 1. 검색 모델 (Sentence Transformer) 로드 및 Faiss 인덱스 구축
        print("1. 검색 엔진 초기화 중...")
        model_name = 'sentence-transformers/all-MiniLM-L6-v2' 
        MODEL = SentenceTransformer(model_name)
        document_embeddings = MODEL.encode(DOCUMENT_CHUNKS, convert_to_numpy=True)
        VECTOR_DIMENSION = document_embeddings.shape[1]
        FAISS_INDEX = faiss.IndexFlatIP(VECTOR_DIMENSION)
        FAISS_INDEX.add(document_embeddings)
        
        # 2. LLM 생성 모델 로드 (GPU가 없다면 CPU에서 실행되므로 매우 느릴 수 있습니다.)
        print("2. LLM 생성 모델 로드 중 (자원 소모 큼)...")
        # ⚠️ 성능 테스트용 예시 모델입니다. 실제 사용 시 한국어 특화 모델로 변경하세요.
        # CPU 사용 시 device=-1
        GENERATOR = pipeline(
            "text-generation", 
            model="skt/kogpt2-base-v2", 
            device=-1 
        )
        print("✅ 모든 엔진 초기화 완료.")
        return True
    except Exception as e:
        print(f"❌ 엔진 초기화 실패: {e}")
        return False

# --- 최종 챗봇 API 엔드포인트 ---
@app.route('/chat', methods=['POST'])
def get_chatbot_response():
    """사용자 질문을 받아 검색 -> 생성까지 모두 수행하고 최종 답변을 반환합니다."""
    if not GENERATOR:
        return jsonify({"error": "LLM 생성 모델이 아직 로드되지 않았습니다."}), 503
        
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "검색어(query)를 제공해야 합니다."}), 400
        
    try:
        # 1. (검색): Faiss를 이용해 관련 문맥(context) 검색
        retrieved_chunks = retrieve_relevant_context(query, FAISS_INDEX, MODEL, DOCUMENT_CHUNKS, k=3)
        context = "\n".join(retrieved_chunks)

        # 2. (프롬프트 구성)
        system_prompt = "당신은 친절하고 전문적인 우석대학교 학사 도우미입니다. 아래 참고 문맥을 기반으로 질문에 명확하게 답변하세요."
        rag_prompt = f"{system_prompt}\n\n--- 참고 문맥 ---\n{context}\n\n사용자 질문: {query}\n\n답변:"

        # 3. (생성): 로컬 LLM 호출
        response = GENERATOR(
            rag_prompt, 
            max_length=256, 
            num_return_sequences=1, 
            do_sample=True, 
            top_p=0.9,
            pad_token_id=GENERATOR.tokenizer.eos_token_id # 텍스트 생성 종료 처리
        )
        
        # 생성된 텍스트에서 실제 답변 부분만 추출 (모델에 따라 파싱이 달라질 수 있음)
        full_text = response[0]['generated_text']
        final_answer = full_text.split("답변:")[-1].strip()
        
        return jsonify({"response": final_answer})
        
    except Exception as e:
        print(f"LLM 답변 생성 중 오류 발생: {e}")
        return jsonify({"error": "파이썬 서버에서 답변 생성 오류가 발생했습니다."}), 500


if __name__ == '__main__':
    # 서버 시작 전에 엔진을 초기화
    if initialize_search_engine():
        print("--- 파이썬 LLM 서버 시작 ---")
        # 리눅스에서 외부 접근을 위해 host='0.0.0.0', 포트는 8000번
        app.run(host='0.0.0.0', port=8000, debug=False)