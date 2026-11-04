from sentence_transformers import SentenceTransformer, CrossEncoder
from usearch.index import Index
import numpy as np

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

test_cases = [
    {
        "query": "DB 사용 중 오류가 발생해서 개발이 막힌 글",
        "relevant_ids": [2, 1, 4, 5, 6]
    },
    {
        "query": "자바 서버에 데이터베이스 연결하기",
        "relevant_ids": [0]
    },
    {
        "query": "데이터베이스 조회 속도를 개선한 사례",
        "relevant_ids": [7, 0]
    },
    {
        "query": "Spring Security로 로그인 기능 구현",
        "relevant_ids": [4, 5]
    },
    {
        "query": "자바에서 실행 중 예외가 발생하는 문제",
        "relevant_ids": [5, 1, 6, 0]
    },
    {
        "query": "Maven 빌드에서 의존성 오류가 발생함",
        "relevant_ids": [6, 0]
    },
    {
        "query": "서비스 배포 후 서버에서 오류가 발생하는 글",
        "relevant_ids": [5, 0]
    },
    {
        "query": "버튼을 한 번 눌렀는데 API 요청이 두 번 나감",
        "relevant_ids": []
    },
    {
        "query": "React 상태 관리 방법을 고민하는 글",
        "relevant_ids": []
    },
    {
        "query": "TCP와 UDP의 차이를 공부하는 내용",
        "relevant_ids": [10]
    },
    {
        "query": "학교 시험이나 중간고사와 관련된 글",
        "relevant_ids": [11, 10, 5]
    },
    {
        "query": "일본어 공부가 너무 어렵고 잘 안 되는 이야기",
        "relevant_ids": [10, 12]
    },
    {
        "query": "일본어 능력 시험 독해 문제 연습",
        "relevant_ids": [10, 11]
    },
    {
        "query": "밥 뭐 먹을지 고민하고 있는 글",
        "relevant_ids": []
    },
    {
        "query": "감기나 목 통증 때문에 몸이 안 좋은 이야기",
        "relevant_ids": [15]
    },
    {
        "query": "운동한 뒤 몸이 아프거나 통증이 생긴 글",
        "relevant_ids": [16]
    },
    {
        "query": "근육을 키우기 위한 근력 운동 이야기",
        "relevant_ids": [15]
    },
    {
        "query": "악기를 연습하면서 겪은 일",
        "relevant_ids": []
    },
    {
        "query": "영화를 보고 느낀 점이나 영상미 이야기",
        "relevant_ids": []
    },
    {
        "query": "원격으로 서버에 접속하거나 외부에 공개하기",
        "relevant_ids": [6, 7]
    }
]

posts = [
    "passage: " + case["query"]
    for case in test_cases
]


post_embeddings = model.encode(
    posts,
    normalize_embeddings=True
).astype(np.float32)

index = Index(
    ndim=post_embeddings.shape[1],
    metric="cos"
)

index.add(
    np.arange(len(posts)),
    post_embeddings
)


def evaluate_recall(case_id, k=5):
    case = test_cases[case_id]
    query = case["query"]

    relevant_ids = set(case["relevant_ids"]) - {case_id}
    if not relevant_ids:
        return None

    query_embedding = model.encode(
        "query: " + query, normalize_embeddings=True).astype(np.float32)

    ann_results = index.search(query_embedding, count=k + 1)

    ann_ids = [
        match.key for match in ann_results
        if match.key != case_id][:k]

    retrieved_ids = {id for id in ann_ids}
    relevant_found = len(relevant_ids.intersection(retrieved_ids))
    print(f"Query: {query}")
    print(
        f"Relevant IDs: {relevant_ids}")
    for i in relevant_ids:
        print(f"Relevant {i}: {test_cases[i]['query']}")
    print()
    print(
        f"Retrieved IDs: {retrieved_ids}")

    for i in retrieved_ids:
        print(f"Relevant found query: {i}: {test_cases[i]["query"]}")
    print(f"Relevant found: {relevant_found}")

    recall = relevant_found / len(relevant_ids) if relevant_ids else 0
    return str(recall) + "\n\n"


print(evaluate_recall(0))

print(evaluate_recall(1))
print(evaluate_recall(2))
print(evaluate_recall(3))
print(evaluate_recall(4))
