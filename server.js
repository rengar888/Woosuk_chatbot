// 모듈 불러오기
const express = require('express');
const path = require('path');
const { getBotResponse } = require('./utils/responseGenerator');

// 서버 및 포트 설정
const app = express();
const PORT = process.env.PORT || 5000; // 환경 변수가 없으면 5000번 포트 사용

// 미들웨어 설정
app.use(express.json()); 
app.use(express.static(path.join(__dirname, 'public')));


// 챗봇 API 엔드포인트 설정: /api/chat
app.post('/api/chat', (req, res) => {
    // 프런트엔드에서 보낸 메시지를 추출
    const { message, history } = req.body; // history는 사용하지 않지만, 요청 구조는 유지

    if (!message) {
        return res.status(400).json({ error: '메시지가 누락되었습니다.' });
    }

    try {
        // 규칙 기반 함수를 호출하여 응답 생성 
        const botResponse = getRuleBasedResponse(message);
        
        // 응답 전송
        res.json({ response: botResponse });

    } catch (error) {
        console.error('챗봇 응답 처리 중 오류 발생:', error);
        res.status(500).json({ error: '서버 내부에서 오류가 발생했습니다.' });
    }
});


// 서버 시작
app.listen(PORT, () => {
    console.log(`✅ 규칙 기반 Node.js 서버가 http://localhost:${PORT} 에서 실행 중입니다.`);
});