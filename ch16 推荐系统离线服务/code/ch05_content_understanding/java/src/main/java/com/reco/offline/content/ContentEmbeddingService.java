package com.reco.offline.content;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * 内容理解 Java 端服务 — 对应 test_chapter.tex 中 1.5 节的 Java 代码片段。
 *
 * 包含以下 tex 代码片段的完整可运行实现：
 *   code_content_embedding  - 直播切片内容向量消费者（LiveSliceEmbeddingConsumer）
 *
 * 对应 Python RPC 服务（code_qwen_vl_rpc）见同目录：
 *   ../src/qwen_vl_rpc_service.py
 *
 * [MOCK] ImageDownloader、ImageEmbeddingClient、ContentVectorLog、ExperimentBucketSelector
 *        均使用内存模拟实现；工业落地时替换为真实 gRPC 客户端和 Kafka 日志写入器。
 */
public class ContentEmbeddingService {

    // ─────────────────────────────────────────────────────────
    //  数据结构：对应 tex snippet code_content_embedding
    // ─────────────────────────────────────────────────────────

    /** 直播切片元数据（来自上游采集服务） */
    public record LiveSlice(long authorId, long capturedAtMs, String imageUrl) {}

    /** 图片下载后保留作者、时间和归一化像素 */
    public record ImageInput(long authorId, long capturedAtMs, List<Double> pixels) {}

    /** 向量 RPC 响应（对应 Python 服务的 ImageEmbeddingResponse protobuf） */
    public record ImageEmbeddingResponse(
            List<Double> values,
            int embeddingDimension,
            int batchSize) {}

    // ─────────────────────────────────────────────────────────
    //  接口定义
    // ─────────────────────────────────────────────────────────

    /** 图片下载 + Resize + Normalize 接口（工业落地时调用 CDN 下载） */
    public interface ImageDownloader {
        List<Double> downloadResizeAndNormalize(String imageUrl) throws Exception;
    }

    /** 图片 Embedding gRPC 客户端接口（工业落地时对接 Python gRPC 服务） */
    public interface ImageEmbeddingClient {
        ImageEmbeddingResponse embed(
                List<Double> imageValues,
                int imageDimension,
                int batchSize) throws Exception;
    }

    /** 内容向量日志写入接口（工业落地时写 Kafka 或对象存储） */
    public interface ContentVectorLog {
        void append(long authorId, long capturedAtMs, List<Double> embedding);
    }

    /** 实验桶选择接口（工业落地时查询实验平台） */
    public interface ExperimentBucketSelector {
        boolean isEnabled(long authorId);
    }

    // ─────────────────────────────────────────────────────────
    //  [MOCK] 内存模拟实现
    // ─────────────────────────────────────────────────────────

    private static final int MOCK_IMAGE_DIM = 3 * 448 * 448;
    private static final int MOCK_EMB_DIM = 512;

    /** [MOCK] 图片下载器：返回随机像素向量 */
    public static class MockImageDownloader implements ImageDownloader {
        private final int dim;
        public MockImageDownloader() { this.dim = MOCK_IMAGE_DIM; }
        public MockImageDownloader(int dim) { this.dim = dim; }

        @Override
        public List<Double> downloadResizeAndNormalize(String imageUrl) {
            // [MOCK] 用确定性伪随机数模拟归一化像素
            List<Double> pixels = new ArrayList<>(dim);
            long seed = imageUrl.hashCode();
            for (int i = 0; i < dim; i++) {
                seed = seed * 6364136223846793005L + 1442695040888963407L;
                pixels.add(((seed >> 33) & 0xFFFF) / 65535.0 * 2.0 - 1.0);
            }
            return pixels;
        }
    }

    /** [MOCK] Embedding 客户端：返回随机向量 */
    public static class MockImageEmbeddingClient implements ImageEmbeddingClient {
        private final int embDim;
        public MockImageEmbeddingClient() { this.embDim = MOCK_EMB_DIM; }

        @Override
        public ImageEmbeddingResponse embed(List<Double> imageValues, int imageDimension, int batchSize) {
            List<Double> vectors = new ArrayList<>(batchSize * embDim);
            for (int i = 0; i < batchSize * embDim; i++) {
                vectors.add(Math.sin(i * 0.01 + batchSize));
            }
            return new ImageEmbeddingResponse(vectors, embDim, batchSize);
        }
    }

    /** [MOCK] 内容向量日志：写入内存列表 */
    public static class InMemoryContentVectorLog implements ContentVectorLog {
        private final List<long[]> meta = new ArrayList<>();
        private final List<List<Double>> embeddings = new ArrayList<>();

        @Override
        public void append(long authorId, long capturedAtMs, List<Double> embedding) {
            meta.add(new long[]{authorId, capturedAtMs});
            embeddings.add(new ArrayList<>(embedding));
        }

        public int size() { return embeddings.size(); }
        public List<Double> getEmbedding(int idx) { return embeddings.get(idx); }
        public long getAuthorId(int idx) { return meta.get(idx)[0]; }
    }

    /** [MOCK] 实验桶选择：全量开放 */
    public static class AlwaysEnabledBucketSelector implements ExperimentBucketSelector {
        @Override
        public boolean isEnabled(long authorId) { return authorId > 0; }
    }

    // ─────────────────────────────────────────────────────────
    //  code_content_embedding
    //  直播切片内容向量消费者
    // ─────────────────────────────────────────────────────────

    /**
     * 直播切片内容向量消费者 — 对应 tex 代码片段 code_content_embedding。
     *
     * 处理流程：
     *   1. 筛选最近切片 + 实验桶
     *   2. 并发下载 + Resize + Normalize
     *   3. 拼接像素数组，调用 Embedding RPC
     *   4. 校验响应，按图片位置拆分向量并写入日志
     */
    public static final class LiveSliceEmbeddingConsumer {
        /** 切片时效性：5 分钟 */
        private static final long RECENT_SLICE_MILLIS = 5 * 60 * 1000L;
        /** 单图像素维度：3 通道 × 448 × 448 */
        private static final int IMAGE_DIMENSION = 3 * 448 * 448;

        private final ExecutorService downloadExecutor;
        private final ImageDownloader downloader;
        private final ImageEmbeddingClient embeddingClient;
        private final ContentVectorLog outputLog;
        private final ExperimentBucketSelector bucketSelector;

        public LiveSliceEmbeddingConsumer(
                ExecutorService downloadExecutor,
                ImageDownloader downloader,
                ImageEmbeddingClient embeddingClient,
                ContentVectorLog outputLog,
                ExperimentBucketSelector bucketSelector) {
            this.downloadExecutor = downloadExecutor;
            this.downloader = downloader;
            this.embeddingClient = embeddingClient;
            this.outputLog = outputLog;
            this.bucketSelector = bucketSelector;
        }

        // ── tex snippet: code_content_embedding ──
        public void consume(List<LiveSlice> slices, long nowMs) {
            List<Future<ImageInput>> futures = new ArrayList<>();
            for (LiveSlice slice : slices) {
                if (!isRecent(slice, nowMs) || !bucketSelector.isEnabled(slice.authorId())) {
                    continue;
                }
                futures.add(downloadExecutor.submit(download(slice)));
            }

            List<ImageInput> inputs = collectInputs(futures);
            if (inputs.isEmpty()) {
                return;
            }

            List<Double> imageValues = new ArrayList<>(inputs.size() * IMAGE_DIMENSION);
            for (ImageInput input : inputs) {
                imageValues.addAll(input.pixels());
            }

            ImageEmbeddingResponse response;
            try {
                response = embeddingClient.embed(
                        imageValues,
                        IMAGE_DIMENSION,
                        inputs.size());
            } catch (Exception error) {
                recordFailure("embedding_rpc", error);
                return;
            }

            if (!isComplete(response, inputs.size())) {
                recordInvalidResponse(response);
                return;
            }
            writeEmbeddings(inputs, response);
        }

        private Callable<ImageInput> download(LiveSlice slice) {
            return () -> new ImageInput(
                    slice.authorId(),
                    slice.capturedAtMs(),
                    downloader.downloadResizeAndNormalize(slice.imageUrl()));
        }

        private List<ImageInput> collectInputs(List<Future<ImageInput>> futures) {
            List<ImageInput> inputs = new ArrayList<>();
            for (Future<ImageInput> future : futures) {
                try {
                    inputs.add(future.get());
                } catch (Exception error) {
                    recordFailure("image_download_or_preprocess", error);
                }
            }
            return inputs;
        }

        private boolean isRecent(LiveSlice slice, long nowMs) {
            return slice.authorId() > 0
                    && slice.capturedAtMs() <= nowMs
                    && nowMs - slice.capturedAtMs() <= RECENT_SLICE_MILLIS;
        }

        private boolean isComplete(ImageEmbeddingResponse response, int inputCount) {
            return response.batchSize() == inputCount && response.embeddingDimension() > 0
                    && response.values().size() == inputCount * response.embeddingDimension();
        }

        private void writeEmbeddings(List<ImageInput> inputs, ImageEmbeddingResponse response) {
            int dimension = response.embeddingDimension();
            for (int index = 0; index < inputs.size(); index++) {
                int from = index * dimension;
                int to = from + dimension;
                ImageInput input = inputs.get(index);
                outputLog.append(input.authorId(), input.capturedAtMs(),
                        response.values().subList(from, to));
            }
        }

        private void recordFailure(String stage, Exception error) {
            System.err.println("[ContentEmbedding] " + stage + ": " + error.getMessage());
        }

        private void recordInvalidResponse(ImageEmbeddingResponse response) {
            System.err.println("[ContentEmbedding] incomplete embedding response: " + response);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Demo 入口
    // ─────────────────────────────────────────────────────────

    public static InMemoryContentVectorLog runDemo() {
        long now = System.currentTimeMillis();
        ExecutorService executor = Executors.newFixedThreadPool(4);
        InMemoryContentVectorLog log = new InMemoryContentVectorLog();

        // [MOCK] 使用小维度加快测试速度
        int smallDim = 3 * 4 * 4; // 3 channels 4x4
        MockImageDownloader downloader = new MockImageDownloader(smallDim) {
            @Override
            public List<Double> downloadResizeAndNormalize(String url) {
                List<Double> px = new ArrayList<>(smallDim);
                for (int i = 0; i < smallDim; i++) px.add(0.5);
                return px;
            }
        };
        MockImageEmbeddingClient client = new MockImageEmbeddingClient() {
            @Override
            public ImageEmbeddingResponse embed(List<Double> vals, int dim, int batchSize) {
                List<Double> vecs = new ArrayList<>();
                for (int i = 0; i < batchSize * MOCK_EMB_DIM; i++) vecs.add((double) i / 1000.0);
                return new ImageEmbeddingResponse(vecs, MOCK_EMB_DIM, batchSize);
            }
        };

        LiveSliceEmbeddingConsumer consumer = new LiveSliceEmbeddingConsumer(
                executor, downloader, client, log, new AlwaysEnabledBucketSelector());

        List<LiveSlice> slices = List.of(
                new LiveSlice(1001L, now - 60_000L, "http://cdn/live/1001/slice1.jpg"),
                new LiveSlice(1002L, now - 120_000L, "http://cdn/live/1002/slice2.jpg"),
                new LiveSlice(0L, now - 10_000L, "http://cdn/invalid/0/bad.jpg"),  // invalid authorId
                new LiveSlice(1003L, now - 999_000L, "http://cdn/live/1003/old.jpg") // too old
        );
        consumer.consume(slices, now);

        executor.shutdown();
        System.out.println("[ContentEmbeddingService] embeddings written: " + log.size());
        return log;
    }

    public static void main(String[] args) {
        System.out.println("=== ch05 ContentEmbeddingService Demo ===");
        InMemoryContentVectorLog result = runDemo();
        System.out.println("Total embeddings: " + result.size());
        if (result.size() > 0) {
            System.out.println("First embedding dim: " + result.getEmbedding(0).size());
        }
    }
}
