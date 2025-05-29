from fastapi import FastAPI
from routers import restaurants,reviews,qa

app=FastAPI(title="식당 추천 & QA API")

#각 라우터 등록
app.include_router(restaurants.router,prefix="/restaurants")
app.include_router(reviews.router,prefix="/reviews")
app.include_router(qa.router,prefix="/qa")

#서버 시작 시 LangChain QA 시스템 초기화 
@app.on_event("startup") # startup 이벤트는 서버가 실행되자마자 딱 한번 실행됨
async def startup_event(): 
    from services.qa_service import build_knowledge_base # 이 함수는 LangChain 기반 질의응답 시스템을 초기화하는 역할을 한다.
    build_knowledge_base()