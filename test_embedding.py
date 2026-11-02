from sentence_transformers import SentenceTransformer, CrossEncoder
from usearch.index import Index
import numpy as np

reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)

model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

posts = [
    "passage: Spring Boot에서 JPA 오류가 발생했다",
    "passage: 자바 서버에 MySQL을 연결했다",
    "passage: Hibernate에서 트랜잭션 오류가 발생했다",
    "passage: REST API 응답 구조를 어떻게 설계해야 할지 고민이다",
    "passage: Spring Security 세션 로그인을 구현했다",
    "passage: 자바에서 NullPointerException이 계속 발생한다",
    "passage: Maven 빌드 중 dependency 오류가 발생했다",
    "passage: 데이터베이스 인덱스를 추가했더니 조회 속도가 빨라졌다",
    "passage: MySQL JOIN 쿼리를 작성하다가 막혔다",
    "passage: 서버 배포 후 500 에러가 발생한다",
    "passage: React에서 상태 관리 구조를 어떻게 잡을지 고민이다",
    "passage: Vite 프로젝트에서 API 프록시를 설정했다",
    "passage: TypeScript 타입 오류를 해결했다",
    "passage: 프론트엔드에서 페이지네이션을 구현했다",
    "passage: 버튼 클릭 시 API 요청이 두 번 발생한다",
    "passage: 오늘 학교에서 네트워크 시험을 봤다",
    "passage: TCP와 UDP 차이가 아직 헷갈린다",
    "passage: 서브넷 마스크 계산 문제를 풀었다",
    "passage: 와이어샤크로 패킷을 분석했다",
    "passage: 라우터와 게이트웨이의 차이를 공부했다",
    "passage: 오늘 자바 반복문 시험을 봤다",
    "passage: 파이썬 리스트 슬라이싱이 헷갈린다",
    "passage: SQL 중간고사를 준비하고 있다",
    "passage: 운영체제 수업에서 프로세스와 스레드를 배웠다",
    "passage: 알고리즘 문제를 DFS로 풀었다",
    "passage: 일본어 단어가 잘 안 외워진다",
    "passage: JLPT N2 독해 문제를 풀었다",
    "passage: 일본어 듣기가 너무 빠르게 느껴진다",
    "passage: 한자를 외울 때 부수를 같이 공부하고 있다",
    "passage: 일본어 문법은 이해되는데 회화가 어렵다",
    "passage: 오늘 뭐 먹을지 고민이다",
    "passage: 점심으로 라면을 먹었다",
    "passage: 배가 너무 고프다",
    "passage: 카레를 만들어 먹었다",
    "passage: 저녁에 치킨을 주문할까 고민 중이다",
    "passage: 오늘 아침부터 목이 아프다",
    "passage: 감기에 걸려서 하루 종일 쉬었다",
    "passage: 잠을 너무 적게 자서 피곤하다",
    "passage: 운동하고 나서 다리가 뻐근하다",
    "passage: 물을 많이 마셨더니 좀 괜찮아졌다",
    "passage: 헬스장에서 스쿼트를 했다",
    "passage: 풀업 개수를 늘리고 싶다",
    "passage: 체지방을 줄이면서 근육을 늘리고 싶다",
    "passage: 오늘은 운동을 쉬기로 했다",
    "passage: 러닝을 했는데 무릎이 조금 아프다",
    "passage: 기타 연습을 시작했다",
    "passage: 드럼 연습하다가 박자를 자꾸 놓친다",
    "passage: 새로 들을 음악을 찾고 있다",
    "passage: 영화 색감이 너무 마음에 들었다",
    "passage: 오래된 일본 영화를 다시 봤다",
    "passage: 게임에서 새로운 캐릭터를 뽑았다",
    "passage: 오늘 게임 업데이트가 있었다",
    "passage: 새로 나온 게임의 그래픽이 좋다",
    "passage: 컴퓨터 성능이 부족해서 프레임이 떨어진다",
    "passage: 키보드를 새로 살지 고민 중이다",
    "passage: 집에 있는 오래된 노트북을 서버로 쓰고 있다",
    "passage: SSH로 원격 서버에 접속했다",
    "passage: Cloudflare Tunnel을 이용해 서버를 외부에 공개했다",
    "passage: 리눅스에서 포트가 열려 있는지 확인했다",
    "passage: 서버 메모리 사용량을 확인하고 있다"
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


def init():
    while True:
        query = input("query: ")
        if (query == "exit"):
            break
        query = "query: " + query

        query_embedding = model.encode(
            query,
            normalize_embeddings=True
        )

        matches = index.search(
            query_embedding,
            count=10
        )

        candidate_ids = [match.key for match in matches]

        pairs = []

        for post_id in candidate_ids:
            post = posts[post_id].removeprefix("passage: ")
            pairs.append([query, post])

        rerank_scores = reranker.predict(pairs)
        results = []
        for i, post_id in enumerate(candidate_ids):
            results.append([
                posts[post_id],
                rerank_scores[i]
            ])
        results.sort(
            key=lambda x: float(x[1]),
            reverse=True
        )
        for post, score in results[:5]:
            print(post, score)
        oldVer(query_embedding)

        print("\n\n----------------------------")


def oldVer(query_embedding):

    print("=== ANN ===")
    matches = index.search(query_embedding, count=3)

    for match in matches:
        print(posts[match.key], match.distance)


init()
