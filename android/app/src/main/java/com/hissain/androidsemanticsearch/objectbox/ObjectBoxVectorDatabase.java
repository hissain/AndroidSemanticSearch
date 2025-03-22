package com.hissain.androidsemanticsearch.objectbox;

import android.content.Context;

import com.hissain.androidsemanticsearch.App;
import com.hissain.androidsemanticsearch.interfaces.VectorDatabase;
import com.hissain.androidsemanticsearch.presentation.SearchResult;

import java.util.ArrayList;
import java.util.List;
import io.objectbox.Box;
import io.objectbox.BoxStore;

public class ObjectBoxVectorDatabase implements VectorDatabase {
    private final Box<SearchDocument> searchDocumentBox;

    public ObjectBoxVectorDatabase(Context context) {
        BoxStore boxStore = App.getBoxStore();
        searchDocumentBox = boxStore.boxFor(SearchDocument.class);
    }

    @Override
    public void addDocument(String id, String content, float[] embedding) {
        SearchDocument doc = new SearchDocument();
        doc.id = id;
        doc.content = content;
        doc.embedding = embedding;
        searchDocumentBox.put(doc);
    }

    @Override
    public List<SearchResult> search(float[] queryEmbedding, int limit) {
        List<SearchResult> results = new ArrayList<>();
        List<SearchDocument> documents = searchDocumentBox.query().build().find();
        documents.sort(new EmbeddingComparator(queryEmbedding));

        for (int i = 0; i < Math.min(limit, documents.size()); i++) {
            SearchDocument doc = documents.get(i);
            float score = computeSimilarity(queryEmbedding, doc.embedding);
            results.add(new SearchResult(doc.id, doc.content, score));
        }

        return results;
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