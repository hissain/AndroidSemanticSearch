package com.hissain.androidsemanticsearch;

public class DocumentItem {
    private String id;
    private String content;

    public DocumentItem(String id, String content) {
        this.id = id;
        this.content = content;
    }

    public String getId() {
        return id;
    }

    public String getContent() {
        return content;
    }
}
