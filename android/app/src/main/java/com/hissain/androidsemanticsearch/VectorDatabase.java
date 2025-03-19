package com.hissain.androidsemanticsearch;

import java.io.IOException;
import java.util.List;

public interface VectorDatabase {
    void addDocument(String id, String content, float[] embedding) throws IOException;
    List<SearchResult> search(float[] queryEmbedding, int limit) throws IOException;
}