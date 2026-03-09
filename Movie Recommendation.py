import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data = {
    "title": [
        "Avengers",
        "Iron Man",
        "Captain America",
        "Batman",
        "Superman"
    ],
    "genre": [
        "action superhero",
        "action technology superhero",
        "action war superhero",
        "action dark superhero",
        "action alien superhero"
    ]
}

df = pd.DataFrame(data)
tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(df["genre"])
similarity = cosine_similarity(tfidf_matrix)
def recommend(movie):

    index = df[df["title"] == movie].index[0]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:4]

    for i in movie_list:
        print(df.iloc[i[0]].title)
recommend("Avengers")
