package com.hissain.androidsemanticsearch;

import android.app.Application;

import com.hissain.androidsemanticsearch.objectbox.MyObjectBox;

import io.objectbox.BoxStore;

public class App extends Application {
    private static BoxStore boxStore;

    @Override
    public void onCreate() {
        super.onCreate();
        boxStore = MyObjectBox.builder().androidContext(this).build();
    }

    public static BoxStore getBoxStore() {
        return boxStore;
    }
}

