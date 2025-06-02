from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.docstore.document import Document

# 전역 QA 체인
qa_chain = None

# LangChain 기반 질의응답 시스템 초기화 함수
def build_knowledge_base():
    global qa_chain

    # 예시 문서 - 실제 Spring 없이 작동하게 설정
    example_texts = [
        Document(page_content="""
        식당 이름: 김밥천국
        주소: 서울특별시 강남구 역삼동
        평점: 4.3
        영업시간: 09:00~22:00
        [홍길동] 맛있어요!
        [김영희] 김밥이 신선해요.
        """),
        Document(page_content="""
        식당 이름: 삼겹살나라
        주소: 서울특별시 마포구 상수동
        평점: 4.7
        영업시간: 11:00~23:00
        [이철수] 고기가 진짜 부드러워요.
        [박미애] 된장찌개도 맛있습니다.
        """),
    ]

    # 벡터 임베딩 및 벡터스토어 구축
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(example_texts, embeddings)

    # QA 체인 생성 (FAISS 기반)
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=store.as_retriever()
    )

# 질문에 대해 답변 생성
def ask(question: str) -> str:
    if qa_chain is None:
        return "❌ QA 시스템이 초기화되지 않았습니다. 먼저 build_knowledge_base()를 호출하세요."
    return qa_chain.run(question)
