from models.semantic_embedding import SemanticEmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
model = SemanticEmbeddingModel()

def semantic_search(supabase, query_text):

    # fetch latest descriptions records from database of latest observations 
    rows = supabase.table("latest_observations").select("*").execute()
    # extracting the description field from each record
    descriptions = [r["description"] for r in rows.data]
    # if no data available then return empty result
    if len(descriptions) == 0:
        return []

    # generate embeddings for database descriptions
    db_embeddings = model.encode(descriptions)
    # generate embeddings for input query
    query_embedding = model.encode([query_text])

    # compute cosine similarities between query and all descriptions
    similarities = cosine_similarity(query_embedding, db_embeddings)[0]

    THRESHOLD = 0.5

    results = []
    # looping through similarity scores and keeping only the results that is above threshold
    for i, score in enumerate(similarities):
        if score >= THRESHOLD:
            results.append({
                "description": descriptions[i],
                "score": float(score)
            })



    return results



# getting the most commonm matching description
def get_most_common(supabase, query_text):

    # fetch latest descriptions from database
    rows = supabase.table("observations").select("description").execute()
    # extract descriptions into list
    descriptions = [r["description"] for r in rows.data]
    # return empty result if no data avaialble
    if len(descriptions) == 0:
        return []

    # generate embedding for descriptions and query
    db_embeddings = model.encode(descriptions)
    query_embedding = model.encode([query_text])

    # compute cosine similarity
    similarities = cosine_similarity(query_embedding, db_embeddings)[0]

    THRESHOLD = 0.5

    # filter the descriptions above threshold
    filtered = []
    for i, score in enumerate(similarities):
        if score >= THRESHOLD:
            filtered.append(descriptions[i])
    # if no matches found, return empty
    if len(filtered) == 0:
        return []

    # counting frequency of filtered descriptions
    counts = Counter(filtered)

    # get most common descritptions(sorted)
    most_common = counts.most_common()

    # select the most frequent description
    most_common_desc = most_common[0][0]

    # extract last word (for the location)
    last_word = most_common_desc.split()[-1]

    return {
        "location": last_word
    }
