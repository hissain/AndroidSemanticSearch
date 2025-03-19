# Android Semantic Search Sample App

This sample app demonstrates how to implement vector search with embeddings for English keywords in an Android application using Java.

## Features

- Transforms text queries into vector embeddings
- Performs semantic search using vector similarity
- Displays search results in a sorted list based on relevance
- Uses MiniLM as the embedding model
- Uses Apache Lucene for vector search capabilities

## Requirements

- Android Studio 2024
- Android SDK 30+ (Android 11.0 or higher)
- JDK 8+
- Tested with Samsung S-24

## Setup Instructions

1. Clone the repository or download the source code
2. Download the required model files:
   - Create an `assets` folder in your Android project's `src/main` directory
   - Download the MiniLM model file and save it as `minilm_l6_v2.tflite` in the assets folder
   - Create a `bert_vocab.txt` file in the assets folder with BERT's vocabulary

3. Sync the project with Gradle files
4. Build and run the app on your device or emulator

## Model Information

This sample uses the MiniLM-L6-v2 model which creates 384-dimensional embeddings. It's a smaller and faster alternative to larger language models while still providing excellent performance for semantic search tasks.

## Implementation Details

### Components

1. **EmbeddingModel**: Interface defining the embedding generation functionality
2. **MiniLMEmbeddingModel**: TensorFlow Lite implementation of the embedding model
3. **BertTokenizer**: Simple tokenizer for processing text input
4. **VectorDatabase**: Interface for vector database operations
5. **LuceneVectorDatabase**: Implementation using Apache Lucene's KNN vector search
6. **MainActivity**: Main UI for the application
7. **ResultsAdapter**: Adapter for displaying search results

### How It Works

1. The app initializes the embedding model and vector database on startup
2. Sample text documents are added to the database with their vector embeddings
3. When a user enters a search query, it's converted to an embedding vector
4. The app performs a KNN vector search to find the most semantically similar documents
5. Results are displayed in a RecyclerView sorted by similarity score

## Dependencies

- **TensorFlow Lite**: For running the embedding model
- **Apache Lucene**: For vector search capabilities
- **AndroidX**: For UI components and RecyclerView

## Notes

- This is a simplified implementation for demonstration purposes
- For production use, consider:
  - Using a more sophisticated tokenizer
  - Implementing asynchronous data loading
  - Adding caching mechanisms for embeddings
  - Storing the vector database persistently

## Contact

Authors email: hissain.khan@gmail.com

Thanks
