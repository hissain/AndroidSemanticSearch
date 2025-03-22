package com.hissain.androidsemanticsearch.interfaces;

import java.io.IOException;

public interface EmbeddingModel {
    float[] generateEmbedding(String text) throws IOException;
}