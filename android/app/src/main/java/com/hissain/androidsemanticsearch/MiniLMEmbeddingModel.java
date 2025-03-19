package com.hissain.androidsemanticsearch;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

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

        // Create input tensors
        ByteBuffer inputIdsBuffer = ByteBuffer.allocateDirect(MAX_SEQ_LENGTH * 4);
        inputIdsBuffer.order(ByteOrder.nativeOrder());
        ByteBuffer attentionMaskBuffer = ByteBuffer.allocateDirect(MAX_SEQ_LENGTH * 4);
        attentionMaskBuffer.order(ByteOrder.nativeOrder());

        // Fill input buffers
        for (int i = 0; i < MAX_SEQ_LENGTH; i++) {
            inputIdsBuffer.putInt(inputIds[i]);
            attentionMaskBuffer.putInt(attentionMask[i]);
        }

        // Reset position to start
        inputIdsBuffer.rewind();
        attentionMaskBuffer.rewind();

        // Create output buffer
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(EMBEDDING_DIM * 4);
        outputBuffer.order(ByteOrder.nativeOrder());

        // Set up input and output maps
        Map<Integer, Object> inputs = new HashMap<>();
        inputs.put(0, inputIdsBuffer);
        inputs.put(1, attentionMaskBuffer);

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, outputBuffer);

        // Run inference
//        interpreter.run(inputs, outputs);
        interpreter.runForMultipleInputsOutputs(new Object[]{inputIdsBuffer, attentionMaskBuffer}, outputs);

        // Extract results
        outputBuffer.rewind();
        float[] embeddings = new float[EMBEDDING_DIM];
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            embeddings[i] = outputBuffer.getFloat();
        }

        return embeddings;
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