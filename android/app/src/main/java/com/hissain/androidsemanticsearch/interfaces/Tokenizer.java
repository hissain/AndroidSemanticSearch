package com.hissain.androidsemanticsearch.interfaces;

import java.io.IOException;

public interface Tokenizer {
    int[] tokenize(String text, int maxLength);
}