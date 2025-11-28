// 1. 필요한 모듈 불러오기
const express = require('express');
const path = require('path');

// 2. 서버 및 포트 설정
const app = express();
const PORT = process.env.PORT || 5000; // 환경 변수가 없으면 5000번 포트 사용

// 3. 규칙 기반 챗봇 응답 함수 
function getRuleBasedResponse(message) {
    const msg = message.toLowerCase().trim();

    if (msg.includes("학사일정")) {
        return "2024년 2학기 주요 학사일정은 학교 홈페이지 공지사항을 확인해 주세요. 9월에는 수강신청 정정 기간이 있습니다.";
    }
    if (msg.includes("학식메뉴") || msg.includes("학식")) {
        return "오늘의 학식 메뉴는 학생식당 앞에 게시되어 있거나, 우석대학교 포털 앱에서 확인 가능합니다.";
    }
    if (msg.includes("셔틀버스") || msg.includes("셔틀")) {
        return "셔틀버스 시간표는 학교 홈페이지 또는 포털에 자세히 안내되어 있습니다. **운행 노선**과 **시간**을 확인하세요.";
    }
    if (msg.includes("졸업")) {
        return "졸업 요건은 학과별로 다를 수 있으니, 소속 학과 사무실이나 교무처에 문의해 주시면 가장 정확한 정보를 얻을 수 있습니다.";
    }
    if (msg.includes("등록금")) {
        return "등록금 납부 기간은 매 학기 시작 전이며, 포털 시스템에서 고지서를 확인할 수 있습니다.";
    }
    if (msg.includes("포털")) {
        return "우석대학교 포털 시스템은 [https://portal.woosuk.ac.kr](https://portal.woosuk.ac.kr) 입니다. 아이디와 비밀번호를 사용하여 로그인하세요.";
    }
    if (msg.includes("안녕") || msg.includes("안녕하세요")) {
        return "안녕하세요! 저는 우석대 도우미 우디봇입니다. 무엇을 도와드릴까요? 학사일정, 학식메뉴 등을 말씀해 주세요.";
    }
    
    // 일치하는 키워드가 없는 경우 기본 응답
    return `죄송합니다. 현재는 '학사일정', '학식메뉴', '셔틀버스' 등 **핵심 키워드**에 대해서만 답변할 수 있습니다. 입력하신 "${message}"에 대해서는 아직 학습되지 않았습니다.`;
}


// 4. 미들웨어 설정
app.use(express.json()); // JSON 요청 본문 파싱
// 정적 파일 서빙 설정: 'public' 폴더의 파일들을 루트 경로(/)에서 제공
app.use(express.static(path.join(__dirname, 'public')));


// 5. 챗봇 API 엔드포인트 설정: /api/chat
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


// 6. 서버 시작
app.listen(PORT, () => {
    console.log(`✅ 규칙 기반 Node.js 서버가 http://localhost:${PORT} 에서 실행 중입니다.`);
});