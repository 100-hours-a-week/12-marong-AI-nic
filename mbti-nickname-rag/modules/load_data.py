from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mysql.connector

def load_from_mysql(host, user, password, database, mbti_query, hobby_query):
    # 1. DB 연결
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    cursor = conn.cursor(dictionary=True)

    # 2. MBTI 문서 로드
    cursor.execute(mbti_query)
    mbti_rows = cursor.fetchall()
    mbti_docs = [
        Document(
            page_content=row["trait"],
            metadata={
                "type": "mbti",
                "MBTI": row["mbti"],
                "trait": row["trait"],
                "weight": float(row["weight"])
            }
        ) for row in mbti_rows
    ]

    # 3. Hobby 문서 로드
    cursor.execute(hobby_query)
    hobby_rows = cursor.fetchall()
    hobby_docs = [
        Document(
            page_content=row["subtrait"],
            metadata={
                "type": "hobby",
                "Hobby": row["hobby"],
                "subtrait": row["subtrait"],
                "weight": float(row["weight"])
            }
        ) for row in hobby_rows
    ]

    cursor.close()
    conn.close()

    # 4. 텍스트 정리 및 청킹
    for doc in mbti_docs + hobby_docs:
        doc.page_content = " ".join(doc.page_content.split())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    return splitter.split_documents(mbti_docs), splitter.split_documents(hobby_docs)
