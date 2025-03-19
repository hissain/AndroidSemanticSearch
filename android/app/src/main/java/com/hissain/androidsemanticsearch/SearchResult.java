package com.hissain.androidsemanticsearch;

public class SearchResult {
    private String id;
    private String content;
    private float score;

    public SearchResult(String id, String content, float score) {
        this.id = id;
        this.content = content;
        this.score = score;
    }

    public String getId() {
        return id;
    }

    public String getContent() {
        return content;
    }

    public float getScore() {
        return score;
    }
}