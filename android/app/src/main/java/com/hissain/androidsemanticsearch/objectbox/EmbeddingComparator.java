package com.hissain.androidsemanticsearch.objectbox;

import java.util.Comparator;

public class EmbeddingComparator implements Comparator<SearchDocument> {
    private final float[] queryEmbedding;

    public EmbeddingComparator(float[] queryEmbedding) {
        this.queryEmbedding = queryEmbedding;
    }

    @Override
    public int compare(SearchDocument doc1, SearchDocument doc2) {
        float score1 = computeSimilarity(queryEmbedding, doc1.embedding);
        float score2 = computeSimilarity(queryEmbedding, doc2.embedding);
        return Float.compare(score2, score1); // Sort in descending order of similarity
    }

    private float computeSimilarity(float[] queryEmbedding, float[] docEmbedding) {
        float dotProduct = 0;
        float normA = 0;
        float normB = 0;
        for (int i = 0; i < queryEmbedding.length; i++) {
            dotProduct += queryEmbedding[i] * docEmbedding[i];
            normA += (float) Math.pow(queryEmbedding[i], 2);
            normB += (float) Math.pow(docEmbedding[i], 2);
        }
        return dotProduct / (float) (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
