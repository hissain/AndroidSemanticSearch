package com.hissain.androidsemanticsearch;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class LuceneVectorDatabase implements VectorDatabase {
    private static final String ID_FIELD = "id";
    private static final String CONTENT_FIELD = "content";
    private static final String VECTOR_FIELD = "vector";

    private final Directory directory;
    private final StandardAnalyzer analyzer;
    private final IndexWriterConfig indexWriterConfig;
    private IndexWriter indexWriter;

    public LuceneVectorDatabase() throws IOException {
        directory = new ByteBuffersDirectory();
        analyzer = new StandardAnalyzer();
        indexWriterConfig = new IndexWriterConfig(analyzer);
        indexWriter = new IndexWriter(directory, indexWriterConfig);
    }

    @Override
    public void addDocument(String id, String content, float[] embedding) throws IOException {
        Document doc = new Document();
        doc.add(new StringField(ID_FIELD, id, Field.Store.YES));
        doc.add(new TextField(CONTENT_FIELD, content, Field.Store.YES));
        doc.add(new KnnVectorField(VECTOR_FIELD, embedding, VectorSimilarityFunction.DOT_PRODUCT));

        indexWriter.addDocument(doc);
        indexWriter.commit();
    }

    @Override
    public List<SearchResult> search(float[] queryEmbedding, int limit) throws IOException {
        List<SearchResult> results = new ArrayList<>();

        // Refresh reader to see latest changes
        indexWriter.commit();
        IndexReader reader = DirectoryReader.open(directory);
        IndexSearcher searcher = new IndexSearcher(reader);

        // Create vector query
        KnnVectorQuery query = new KnnVectorQuery(VECTOR_FIELD, queryEmbedding, limit);
        TopDocs topDocs = searcher.search(query, limit);

        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            Document doc = searcher.doc(scoreDoc.doc);
            String id = doc.get(ID_FIELD);
            String content = doc.get(CONTENT_FIELD);
            float score = scoreDoc.score;

            results.add(new SearchResult(id, content, score));
        }

        reader.close();
        return results;
    }
}
