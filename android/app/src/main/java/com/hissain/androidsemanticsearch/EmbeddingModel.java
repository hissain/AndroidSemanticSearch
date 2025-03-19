package com.hissain.androidsemanticsearch;

import java.io.IOException;

public interface EmbeddingModel {
    float[] generateEmbedding(String text) throws IOException;
}