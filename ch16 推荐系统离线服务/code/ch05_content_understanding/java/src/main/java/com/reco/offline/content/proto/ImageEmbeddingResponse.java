package com.reco.offline.content.proto;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 手写的 ImageEmbeddingResponse — 对应 proto image_embedding_service.proto。
 */
public final class ImageEmbeddingResponse {

    private final List<Float> imageEmbedding;
    private final int embDim;
    private final int batchSize;

    private ImageEmbeddingResponse(Builder builder) {
        this.imageEmbedding = Collections.unmodifiableList(new ArrayList<>(builder.imageEmbedding));
        this.embDim = builder.embDim;
        this.batchSize = builder.batchSize;
    }

    public List<Float> getImageEmbeddingList()  { return imageEmbedding; }
    public int getImageEmbeddingCount()         { return imageEmbedding.size(); }
    public int getEmbDim()                      { return embDim; }
    public int getBatchSize()                   { return batchSize; }

    public static Builder newBuilder() { return new Builder(); }

    public static final class Builder {
        private final List<Float> imageEmbedding = new ArrayList<>();
        private int embDim;
        private int batchSize;

        public Builder addAllImageEmbedding(List<Float> vs) { imageEmbedding.addAll(vs); return this; }
        public Builder setEmbDim(int v)    { embDim = v; return this; }
        public Builder setBatchSize(int v) { batchSize = v; return this; }
        public ImageEmbeddingResponse build() { return new ImageEmbeddingResponse(this); }
    }
}
