package com.hissain.androidsemanticsearch.presentation;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import com.hissain.androidsemanticsearch.R;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class ResultsAdapter extends RecyclerView.Adapter<ResultsAdapter.ViewHolder> {
    private List<SearchResult> results;

    public ResultsAdapter(List<SearchResult> results) {
        this.results = results;
    }

    public void updateResults(List<SearchResult> newResults) {
        this.results = newResults;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.result_item, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        SearchResult result = results.get(position);
        holder.contentTextView.setText(result.getContent());
        holder.scoreTextView.setText(String.format("Score: %.4f", result.getScore()));
    }

    @Override
    public int getItemCount() {
        return results.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        TextView contentTextView;
        TextView scoreTextView;

        ViewHolder(View itemView) {
            super(itemView);
            contentTextView = itemView.findViewById(R.id.result_content);
            scoreTextView = itemView.findViewById(R.id.result_score);
        }
    }
}
