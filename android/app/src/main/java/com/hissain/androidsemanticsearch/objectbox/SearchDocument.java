package com.hissain.androidsemanticsearch.objectbox;

import io.objectbox.annotation.Entity;
import io.objectbox.annotation.Id;

@Entity
public class SearchDocument {
    @Id
    public long objectBoxId;
    public String id;
    public String content;
    public float[] embedding;
}
