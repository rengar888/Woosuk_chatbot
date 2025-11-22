from flask import Flask, request, jsonify
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# --- 전역 변수 및 설정 ---
# ⚠️ 실제 사용하는 청크 데이터를 여기에 넣거나 파일에서 로드해야 합니다.
DOCUMENT_CHUNKS = [
    "우석대학교의 2025학년도 1학기 등록금 납부 기간은 2월 20일부터 2월 24일까지입니다.",
    "등록금을 미납할 경우 학점 취소 및 제적 처리될 수 있습니다. 반드시 기한 내 납부해야 합니다.",
    "2024년 11월에는 수강신청 안내 및 졸업 사정회가 예정되어 있습니다.",
    "셔틀버스는 삼례캠퍼스와 전주역, 익산역을 운행하며 자세한 시간표는 학교 포털 공지사항을 확인하세요."
]
MODEL = None
FAISS_INDEX = None
VECTOR_DIMENSION = 0

def initialize_search_engine():
    """서버 시작 시 임베딩 모델과 Faiss 인덱스를 초기화"""
    global MODEL, FAISS_INDEX, VECTOR_DIMENSION
    
    try:
        # 1. 모델 로드 (Hugging Face에서 다운로드)
        model_name = 'sentence-transformers/all-MiniLM-L6-v2' 
        MODEL = SentenceTransformer(model_name)
        
        # 2. 문서 임베딩 생성
        document_embeddings = MODEL.encode(DOCUMENT_CHUNKS, convert_to_numpy=True)
        VECTOR_DIMENSION = document_embeddings.shape[1]
        
        # 3. Faiss 인덱스 구축 (내적(IP) 기반)
        FAISS_INDEX = faiss.IndexFlatIP(VECTOR_DIMENSION)
        FAISS_INDEX.add(document_embeddings)
        print("✅ 파이썬 검색 엔진 초기화 완료 (Faiss 인덱스 구축됨).")
        return True
    except Exception as e:
        print(f"❌ 검색 엔진 초기화 실패: {e}")
        return False

# --- 검색 API 엔드포인트 ---
@app.route('/search', methods=['POST'])
def search_context():
    """사용자 쿼리를 받아 Faiss 인덱스에서 관련 문맥을 검색"""
    if not MODEL or not FAISS_INDEX:
        return jsonify({"error": "검색 엔진이 아직 로드되지 않았습니다."}), 503
        
    data = request.get_json()
    query = data.get('query')
    k = data.get('k', 2)
    
    if not query:
        return jsonify({"error": "검색어(query)를 제공해야 합니다."}), 400
        
    try:
        # 1. 질문 임베딩 (파이썬 로컬 모델 사용)
        query_embedding = MODEL.encode([query], convert_to_numpy=True)
        
        # 2. Faiss 검색
        D, I = FAISS_INDEX.search(query_embedding, k) 
        
        # 3. 결과 포맷팅
        retrieved_context = [
            {"text": DOCUMENT_CHUNKS[i], "similarity": float(D[0][idx])} 
            for idx, i in enumerate(I[0])
        ]
        
        return jsonify({"context": retrieved_context})
        
    except Exception as e:
        print(f"검색 중 오류 발생: {e}")
        return jsonify({"error": "검색 과정 중 서버 오류가 발생했습니다."}), 500


if __name__ == '__main__':
    # 서버 시작 전에 엔진을 초기화
    if initialize_search_engine():
        # Node.js 서버와 다른 포트 사용 (예: 8000번)
        # 리눅스 서버에서 외부 접근을 위해 host='0.0.0.0' 사용
        app.run(host='0.0.0.0', port=8000, debug=False)