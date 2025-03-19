package com.hissain.androidsemanticsearch;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MiniLMEmbeddingModel implements EmbeddingModel {
    private static final int MAX_SEQ_LENGTH = 128;
    private static final int EMBEDDING_DIM = 384; // MiniLM L6 embedding dimension

    private final Interpreter interpreter;
    private final BertTokenizer tokenizer;

    public MiniLMEmbeddingModel(Context context) throws IOException {
        // Load TensorFlow Lite model
        interpreter = new Interpreter(loadModelFile(context, "minilm_l6_v2.tflite"));
        tokenizer = new BertTokenizer(context);
    }

    @Override
    public float[] generateEmbedding(String text) throws IOException {
        // Tokenize input text
        int[] inputIds = tokenizer.tokenize(text, MAX_SEQ_LENGTH);
        int[] attentionMask = new int[MAX_SEQ_LENGTH];
        for (int i = 0; i < MAX_SEQ_LENGTH; i++) {
            attentionMask[i] = inputIds[i] != 0 ? 1 : 0;
        }

        // Prepare input for the model
        Object[] inputs = new Object[2];
        inputs[0] = new int[1][MAX_SEQ_LENGTH];
        inputs[1] = new int[1][MAX_SEQ_LENGTH];

        for (int i = 0; i < MAX_SEQ_LENGTH; i++) {
            ((int[][])inputs[0])[0][i] = inputIds[i];
            ((int[][])inputs[1])[0][i] = attentionMask[i];
        }

        // Prepare output buffer
        float[][][] outputs = new float[1][1][EMBEDDING_DIM];

        // Run inference
        interpreter.run(inputs, outputs);

        // Return the embedding (CLS token)
        return outputs[0][0];
    }

    private MappedByteBuffer loadModelFile(Context context, String modelFileName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelFileName);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
