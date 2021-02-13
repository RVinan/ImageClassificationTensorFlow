    package com.example.appfinalrv;


import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

import com.example.appfinalrv.classifier.*;


public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 1000;
    private static final int CAMERA_REQEUST_CODE = 10001;

    private ImageView imageView;
    private ListView listView;
    private ImageClassifier imageClassifier;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initializeUIElements();
    }

    private void initializeUIElements() {
        imageView = findViewById(R.id.iv_capture);
        listView = findViewById(R.id.lv_probabilities);
        Button takepicture = findViewById(R.id.bt_take_picture);

        try {
            imageClassifier = new ImageClassifier(this);
        } catch (IOException e) {
            Log.e("Image Classifier Error", "ERROR: " + e);
        }

        takepicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if (hasPermission()) {
                    openCamera();
                } else {
                    requestPermission();
                }
            }
        });
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        if (requestCode == CAMERA_REQEUST_CODE) {

            Bitmap photo = (Bitmap) Objects.requireNonNull(Objects.requireNonNull(data).getExtras()).get("data");

            imageView.setImageBitmap(photo);

            List<ImageClassifier.Recognition> predicitons = imageClassifier.recognizeImage(
                    photo, 0);

            final List<String> predicitonsList = new ArrayList<>();
            for (ImageClassifier.Recognition recog : predicitons) {
                predicitonsList.add(recog.getName() + "  ::::::::::  " + recog.getConfidence());
            }

            ArrayAdapter<String> predictionsAdapter = new ArrayAdapter<>(
                    this, R.layout.support_simple_spinner_dropdown_item, predicitonsList);
            listView.setAdapter(predictionsAdapter);
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (hasAllPermissions(grantResults)) {
                openCamera();
            } else {
                requestPermission();
            }
        }
    }

    private boolean hasAllPermissions(int[] grantResults) {
        for (int result : grantResults) {
            if (result == PackageManager.PERMISSION_DENIED)
                return false;
        }
        return true;
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {

            if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
                Toast.makeText(this, "Camera Permission Required", Toast.LENGTH_SHORT).show();
            }

            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        }
    }

    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, CAMERA_REQEUST_CODE);
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
        }
        return true;
    }
}
