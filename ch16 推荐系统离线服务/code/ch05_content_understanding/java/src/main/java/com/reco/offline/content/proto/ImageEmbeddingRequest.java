package com.reco.offline.content.proto;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 手写的 ImageEmbeddingRequest — 对应 proto image_embedding_service.proto。
 *
 * 工业落地时替换为 protoc 自动生成的版本：
 *   protoc --java_out=... --grpc-java_out=... image_embedding_service.proto
 */
public final class ImageEmbeddingRequest {

    private final List<Float> imageList;
    private final int imageDim;
    private final int batchSize;

    private ImageEmbeddingRequest(Builder builder) {
        this.imageList = Collections.unmodifiableList(new ArrayList<>(builder.imageList));
        this.imageDim = builder.imageDim;
        this.batchSize = builder.batchSize;
    }

    public List<Float> getImageListList() { return imageList; }
    public int getImageDim()              { return imageDim; }
    public int getBatchSize()             { return batchSize; }

    public static Builder newBuilder() { return new Builder(); }

    public static final class Builder {
        private final List<Float> imageList = new ArrayList<>();
        private int imageDim;
        private int batchSize;

        public Builder addImageList(float v) { imageList.add(v); return this; }
        public Builder addAllImageList(List<Float> vs) { imageList.addAll(vs); return this; }
        public Builder setImageDim(int v)    { imageDim = v; return this; }
        public Builder setBatchSize(int v)   { batchSize = v; return this; }
        public ImageEmbeddingRequest build() { return new ImageEmbeddingRequest(this); }
    }
}
