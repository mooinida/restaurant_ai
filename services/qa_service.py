from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA  # ✅ 최신 구조에 맞게 수정
from langchain.llms.openai import OpenAI
from langchain.docstore.document import Document
from openai import embeddings
from routers import restaurants
from services.spring_client import fetch_restaurants, fetch_reviews

#전역 QA 체인
qa_chain = None

#LangChain 기반 질의응답 시스템 초기화 함수
def build_knowledge_base():
    global qa_chain

    #Spring에서 식당정보 수집
    restaurants=fetch_restaurants()
    docs=[]

    for r in restaurants:
        reviews=fetch_reviews(r["placed"]) # 해당 식당 리뷰

        #LangChain 문서 형식으로 구성
        text=f"식당 이름: {r['name']}\n주소: {r['address']}\n평점: {r['rating']}\n"
        text +=f"영업시간: {'; '.join(r.get('openingHours',[]))}\n"
        text +="\n".join([f"[{v['author']}] {v['text']}" for v in reviews])
        docs.append(Document(page_content=text))
    
    #벡터 임베딩 및 벡터스토어 구축
    embeddings = OpenAIEmbeddings()# 문서를 수치화된 벡터로 바꿔주는 임베딩 모델을 설정하는 부분
                                    #저 함수는 오픈에이의 텍스트 임베딩 모델을 사용해서 문장을 고차원 벡터로 변환
                                    #이 벡터는 나중에 질문과 문서를 유사도로 비교할 떄 사용됨
                                    #예시로 강남역 맛집 알려줘라는 질문이 벡터로 바뀌고, 식당 설명 벡터와 비슷한 정도를 계산가능
    store=FAISS.from_documents(docs,embeddings)# 문서리스트 docs를 벡터저장소 faiss 에 저장하는 과정 
                                                #FAiss는 빠른 벡터 검색 라이브러리로, 유사한 내용을 빠르게 찾을 수 있음.

    #QA 체인 생성 (retriever: FAISS) 벡터 검색 결과를 기반으로 질문에 답변하는 체인을 만드는 부분
    qa_chain=RetrievalQA.from_chain_type(#RetrievalQA는 LangChain에서 제공하는 질의응답 템플릿 클래스.
        llm=OpenAI(),#OpenAi 기반 언어모델 사용. 질문과 관련 문서가 정리되었을 때 답변을 생성할 언어 모델을 OpenAi로 설정
        retriever=store.as_retriever()# FAISS 벡터스토어에엇 관련 문서 찾는 검색기를 연결하는 부분
    )
#질문에 대해 답변 생성
def ask(question:str)->str:
    if qa_chain is None:
        return "QA 시스템이 초기화되지않음"
    return qa_chain.run(question)



#흐름 문서 텍스트->임베딩->벡터화 --FAISS에 저장--> 질문 입력 -- 벡터화 -> 유사 문서 검색 --> Openai로 답변 생성