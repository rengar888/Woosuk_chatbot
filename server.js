require('dotenv').config(); 
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const app = express();
const port = process.env.PORT || 5000;

// 파이썬 LLM 서버 URL 설정
const PYTHON_LLM_API_URL = 'http://localhost:8000/chat';

// 미들웨어 설정
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});


app.post('/api/chat', async (req, res) => {
    // history를 받음 (프론트엔드에서 보낸 이전 대화 기록)

    const { message } = req.body; 
    
    if (!message) {
        return res.status(400).json({ error: "메시지가 없습니다" });
    }

    try {
        // 1.  파이썬 LLM 서버에 질문 전송
        console.log(`파이썬 LLM 서버로 요청 중: ${message}`);
        const llmResponse = await fetch(PYTHON_LLM_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: message }), 
        });

        if (!llmResponse.ok) {
            // 파이썬 서버가 500 또는 404 등을 반환했을 경우
            throw new Error(`파이썬 LLM 서버 오류! 상태 코드: ${llmResponse.status}`);
        }

        const data = await llmResponse.json();
        
        // 2. 파이썬 서버로부터 최종 답변 수신
        const botResponse = data.response; 
        res.json({ response: botResponse });

    } catch (error) {
        console.error("❌ 파이썬 LLM 통신 중 오류 발생:", error);
        res.status(500).json({ error: "서버에서 오류가 발생했습니다. (LLM 통신 실패)" });
    }
});

// 서버 실행
app.listen(port, () => {
    console.log(`챗봇 백엔드 서버가 http://localhost:${port} 에서 실행 중입니다.`);
    // ❌ initializeAcademicData() 호출 제거됨.
});