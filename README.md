# Video Game Recommendation System — metadata-based (content filtering)

## Summary

A content-based recommendation system for video games using metadata (title, platform, year, genre, publisher, rating). TF-IDF on a combined text field + cosine similarity is used to find similar games. The notebook includes data cleaning, EDA, TF-IDF vectorization, and a simple `get_recommendations` function.
**Practical note:** for better UX, restrict recommendations to the same platform (or boost same-platform matches).

## Dataset & preprocessing

* Numerical columns include sales, scores, counts, year of release.
* Categorical columns include `Name, Platform, Genre, Publisher, User_Score, Developer, Rating`.
* After cleaning, create a combined metadata column:

  ```python
  df['game'] = (
      df['Name'] + ', ' +
      df['Platform'] + ', ' +
      df['Year_of_Release'].astype(str) + ', ' +
      df['Genre'] + ', ' +
      df['Publisher'] + ', ' +
      df['Rating']
  )
  df.dropna(subset=['game'], inplace=True)
  df['game'] = df['game'].str.lower()
  ```
* For quick experiments you sampled 10k rows and deduplicated; for production use the full, deduplicated dataset.

## EDA highlights

* Top publishers and platforms by number of games (PS2, DS, PS3, Wii, X360, etc.).
* Genre and rating distributions examined to understand popularity and sparsity.

## Method (implementation)

* Vectorize `game` column with `TfidfVectorizer(stop_words='english')`.
* Compute cosine similarities: `cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)`.
* `get_recommendations(title, num_recommend=5)` looks up matching entries (handles multiple rows with same name), computes similarity scores per matching row, merges results, and returns top unique recommendations sorted by similarity.

## Example (usage idea)

```python
# compute tfidf + cosine_sim on samp (or full dataframe)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(samp['game']).toarray()
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# call the function
print(get_recommendations("Fighting Force 2"))
```

## Observed behavior & tip

* The example results look reasonable (high similarity scores for same-publisher / same-genre titles).
* Multiple entries for the same game (different platforms) are handled; you can:

  * **Filter by platform** before returning recommendations, or
  * **Post-filter** results to prefer games on the same platform (recommended for most user scenarios).

## Conclusion

Content-based TF-IDF + cosine works well for metadata-based suggestions. For real-world usefulness, prefer or boost same-platform recommendations and use the full deduplicated dataset instead of sampling.
