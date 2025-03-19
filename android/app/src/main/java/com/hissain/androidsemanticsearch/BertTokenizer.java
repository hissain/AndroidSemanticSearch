package com.hissain.androidsemanticsearch;

import android.content.Context;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BertTokenizer {
    private static final String VOCAB_FILE = "bert_uncased_vocab.txt";
    private static final String UNK_TOKEN = "[UNK]";
    private static final String CLS_TOKEN = "[CLS]";
    private static final String SEP_TOKEN = "[SEP]";

    private final Map<String, Integer> vocabMap = new HashMap<>();

    public BertTokenizer(Context context) throws IOException {
        loadVocabulary(context);
    }

    private void loadVocabulary(Context context) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(
                context.getAssets().open(VOCAB_FILE)));
        String line;
        int index = 0;
        while ((line = reader.readLine()) != null) {
            vocabMap.put(line.trim(), index++);
        }
        reader.close();
    }

    public int[] tokenize(String text, int maxLength) {
        // Simple whitespace tokenization for demo
        // A real implementation would use WordPiece tokenization
        List<String> tokens = new ArrayList<>();
        tokens.add(CLS_TOKEN);

        String[] words = text.toLowerCase().split("\\s+");
        for (String word : words) {
            if (tokens.size() >= maxLength - 1) {
                break;
            }
            tokens.add(vocabMap.containsKey(word) ? word : UNK_TOKEN);
        }

        tokens.add(SEP_TOKEN);

        // Convert tokens to IDs
        int[] ids = new int[maxLength];
        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = vocabMap.getOrDefault(tokens.get(i), vocabMap.get(UNK_TOKEN));
        }

        return ids;
    }
}
