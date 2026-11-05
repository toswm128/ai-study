from sentence_transformers import SentenceTransformer, CrossEncoder
from usearch.index import Index
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi
import numpy as np

kiwi = Kiwi()

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


def embeddQuery(query):
    return model.encode(
        "query: " + query, normalize_embeddings=True).astype(np.float32)


def annEmbeddedQuery(EQ, index, count):
    return index.search(EQ, count)


def tokenize(text):
    return [
        token.form.lower()
        for token in kiwi.tokenize(text)
    ]


k = 10

documents = [case["query"] for case in test_cases]


tokenized_docs = [tokenize(doc) for doc in documents]


bm25 = BM25Okapi(tokenized_docs)

query = "자바 서버 오류"


query_tokens = tokenize(query)


scores = bm25.get_scores(query_tokens)


ranked_ids = np.argsort(scores)[::-1]

for doc_id in ranked_ids[:k]:
    print(doc_id, documents[doc_id], scores[doc_id], "\n")


ann_list = annEmbeddedQuery(embeddQuery(query), index, k)

ann_ids = [int(match.key) for match in ann_list]


for match in ann_list:
    print(f"{match.key}: {test_cases[match.key]["query"]}, {match.distance}")

print("\n\n\n")


def rrf_merge(ann_ids, bm25_ids, rrf_k=60):
    scores = {}
    for rank, id in enumerate(ann_ids, start=1):
        scores[id] = 1/(rank+rrf_k)
    for rank, id in enumerate(bm25_ids, start=1):
        scores[id] = scores.get(id, 0) + 1 / (rank + rrf_k)

    scores = list(zip([id for id in scores], [scores.get(id)
                  for id in scores]))

    scores.sort(key=lambda x: x[1], reverse=True)

    for id, score in scores:
        print(f"{id}: {test_cases[id]["query"]} {score}")


rrf_merge(ann_ids, ranked_ids)
