package com.hissain.androidsemanticsearch;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {
    private EditText searchQueryInput;
    private Button searchButton;
    private TextView statusText;
    private RecyclerView resultsRecyclerView;
    private ResultsAdapter resultsAdapter;

    private EmbeddingModel embeddingModel;
    private VectorDatabase vectorDatabase;
    private final Executor executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI components
        searchQueryInput = findViewById(R.id.search_query_input);
        searchButton = findViewById(R.id.search_button);
        statusText = findViewById(R.id.status_text);
        resultsRecyclerView = findViewById(R.id.results_recycler_view);

        // Set up RecyclerView
        resultsAdapter = new ResultsAdapter(new ArrayList<>());
        resultsRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        resultsRecyclerView.setAdapter(resultsAdapter);

        // Initialize embedding model and vector database
        initializeModels();

        // Set up search button click listener
        searchButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String query = searchQueryInput.getText().toString().trim();
                if (!query.isEmpty()) {
                    performSearch(query);
                }
            }
        });
    }

    private void initializeModels() {
        statusText.setText("Initializing models...");
        searchButton.setEnabled(false);

        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    // Initialize the embedding model
                    embeddingModel = new MiniLMEmbeddingModel(getApplicationContext());

                    // Initialize and populate the vector database with sample data
                    vectorDatabase = new LuceneVectorDatabase();
                    List<DocumentItem> sampleData = generateSampleData();

                    for (DocumentItem item : sampleData) {
                        float[] embedding = embeddingModel.generateEmbedding(item.getContent());
                        vectorDatabase.addDocument(item.getId(), item.getContent(), embedding);
                    }

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            statusText.setText("Ready to search");
                            searchButton.setEnabled(true);
                        }
                    });
                } catch (IOException e) {
                    final String errorMessage = "Error initializing models: " + e.getMessage();
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            statusText.setText(errorMessage);
                        }
                    });
                }
            }
        });
    }

    private void performSearch(final String query) {
        statusText.setText("Searching...");
        searchButton.setEnabled(false);

        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    float[] queryEmbedding = embeddingModel.generateEmbedding(query);
                    List<SearchResult> results = vectorDatabase.search(queryEmbedding, 5);

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            resultsAdapter.updateResults(results);
                            statusText.setText("Found " + results.size() + " results");
                            searchButton.setEnabled(true);
                        }
                    });
                } catch (IOException e) {
                    final String errorMessage = "Error during search: " + e.getMessage();
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            statusText.setText(errorMessage);
                            searchButton.setEnabled(true);
                        }
                    });
                }
            }
        });
    }

    private List<DocumentItem> generateSampleData() {
        List<DocumentItem> data = new ArrayList<>();
        data.add(new DocumentItem("1", "Machine learning algorithms are transforming how we analyze data"));
        data.add(new DocumentItem("2", "Natural language processing helps computers understand human language"));
        data.add(new DocumentItem("3", "Vector databases store and retrieve high-dimensional vectors efficiently"));
        data.add(new DocumentItem("4", "Embeddings represent words or documents as dense vectors"));
        data.add(new DocumentItem("5", "Semantic search understands the meaning behind search queries"));
        data.add(new DocumentItem("6", "Android development requires understanding of Java or Kotlin"));
        data.add(new DocumentItem("7", "Mobile applications run on smartphones and tablets"));
        data.add(new DocumentItem("8", "User experience design focuses on creating intuitive interfaces"));
        data.add(new DocumentItem("9", "Artificial intelligence mimics human cognitive functions"));
        data.add(new DocumentItem("10", "Deep learning uses neural networks with many layers"));
        return data;
    }
}