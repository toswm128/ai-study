from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)

posts = [
    "passage: Spring Boot에서 JPA 오류가 발생했다",
    "passage: 자바 서버에 MySQL을 연결했다",
    "passage: 오늘 학교에서 네트워크 시험을 봤다",
    "passage: 오늘 뭐먹을까",
    "passage: 배고프다",
    "passage: 일본어 공부 개힘드네",
    "passage: 감기걸렸음"
]

post_embeddings = model.encode(posts)


def init():
    while True:
        query = input("query: ")
        if (query == "exit"):
            break
        query = "query: " + query
        query_embedding = model.encode(query)
        scores = model.similarity(query_embedding, post_embeddings)
        posts_with_scores = [[posts[i], scores[0][i]]
                             for i in range(len(posts))]
        posts_with_scores.sort(key=lambda x: x[1].item(), reverse=True)
        for ps in posts_with_scores:
            print(ps[0], ps[1])
        print("\n\n----------------------------")


init()
