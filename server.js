require('dotenv').config(); 
const express = require('express');
const { GoogleGenAI } = require('@google/genai');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 5000;

/*const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,  // .env 파일에서 API 키 로드
});*/

/* let academicEmbeddings = [];

function dotProduct(vecA, vecB) {
    return vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
}

async function initializeAcademicData() {
    try {
        console.log("학사 정보 임베딩을 시작합니다...");

        const dataPath = path.join(__dirname, 'data', 'academics.txt');
        const fileContent = fs.readFileSync(dataPath, 'utf-8');

        const chunks = fileContent.split('---')
        .map(c => c.trim())
        .filter(c => c.length > 0 && !c.startsWith('문서'));

        console.log(`총 ${chunks.length}개의 청크를 임베딩합니다...`);

        if (chunks.length === 0) {
            throw new Error("임베딩할 청크가 없습니다.");
            return;
        }

        const contents = chunks.map(text => ({ 
            role: "user", 
            parts: [{ text: text }] 
        }));


        const embeddingResponse = await ai.models.embedContent({
            model: "embedding-001",
            contents: contents,
        });

        academicEmbeddings = embeddingResponse.embeddings.map((embeddingItem, index) => ({
            text: chunks[index],
            embedding: embeddingItem.values
        }));

        console.log("학사 정보 임베딩이 완료되었습니다.");
    } catch (error) {
        console.error("학사 정보 임베딩 중 오류 발생:", error);
    }
}
*/


// 미들웨어 설정
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PYTHON_SEARCH_API_URL = 'http://localhost:8000/search';

app.post('/api/chat', async (req, res) => {
    // history를 받음 (프론트엔드에서 보낸 이전 대화 기록)
    const { message, history } = req.body; 
    
    if (!message || academicEmbeddings.length === 0) {
        return res.status(400).json({ error: "메시지 또는 임베딩 데이터가 없습니다." });
    }

    let context = "";

    try {
        // 1. 🐍 파이썬 검색 서버에 질문 전송
        console.log("파이썬 검색 서버로 요청 중...");
        const searchResponse = await fetch(PYTHON_SEARCH_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: message, k: 2 }), 
        });

        if (!searchResponse.ok) {
            throw new Error(`파이썬 검색 서버 오류! 상태 코드: ${searchResponse.status}`);
        }

        const searchData = await searchResponse.json();
        
        // 2. 검색된 청크를 RAG Context로 구성
        const relevantDocs = searchData.context;
        
        // 파이썬에서 받은 텍스트만 추출하여 Context 구성
        context = relevantDocs.map(doc => `[문서] ${doc.text}`).join('\n---\n');
        
        // 3. (4. 이후 단계) Gemini 모델 호출 (기존 RAG 로직 유지)
        
        const systemInstruction = `당신은 친절하고 전문적인 **[우석대학교] 학사 정보 도우미**입니다. 사용자의 질문에 답변할 때, 반드시 제공된 '참고 문서'를 기반으로 답변해야 합니다. 만약 참고 문서에 답변이 없다면, "관련 정보를 찾을 수 없습니다."라고 응답하세요.`;
        
        const ragUserPrompt = `사용자 질문: "${message}"\n\n--- 참고 문서 ---\n${context}`;
        
        const contents = [
            ...(history || []), 
            { 
                role: "user", 
                parts: [{ text: ragUserPrompt }] 
            }
        ];

        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            config: {
                systemInstruction: systemInstruction,
            },
            contents: contents
        });

        const botResponse = response.text;
        res.json({ response: botResponse });

    } catch (error) {
        console.error("❌ 처리 중 오류 발생:", error);
        res.status(500).json({ error: "서버에서 오류가 발생했습니다. (검색 또는 생성 단계)" });
    }
});

// 서버 실행
app.listen(port, () => {
    console.log(`Windows 개발 서버가 http://localhost:${port} 에서 실행 중입니다.`);
});