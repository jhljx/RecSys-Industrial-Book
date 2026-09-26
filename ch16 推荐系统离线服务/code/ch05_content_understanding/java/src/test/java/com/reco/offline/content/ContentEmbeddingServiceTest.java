package com.reco.offline.content;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.concurrent.Executors;

public class ContentEmbeddingServiceTest {

    /** 工具：创建小维度 mock Consumer（加快测试） */
    private ContentEmbeddingService.LiveSliceEmbeddingConsumer buildConsumer(
            ContentEmbeddingService.InMemoryContentVectorLog log) {
        int dim = 3 * 4 * 4;
        return new ContentEmbeddingService.LiveSliceEmbeddingConsumer(
                Executors.newFixedThreadPool(2),
                url -> {
                    List<Double> px = new java.util.ArrayList<>(dim);
                    for (int i = 0; i < dim; i++) px.add(0.5);
                    return px;
                },
                (vals, imageDim, batchSize) -> {
                    int embDim = 16;
                    List<Double> vecs = new java.util.ArrayList<>(batchSize * embDim);
                    for (int i = 0; i < batchSize * embDim; i++) vecs.add((double) i);
                    return new ContentEmbeddingService.ImageEmbeddingResponse(vecs, embDim, batchSize);
                },
                log,
                new ContentEmbeddingService.AlwaysEnabledBucketSelector()
        );
    }

    @Test
    void testConsumeValidSlices() {
        long now = System.currentTimeMillis();
        ContentEmbeddingService.InMemoryContentVectorLog log =
                new ContentEmbeddingService.InMemoryContentVectorLog();
        var consumer = buildConsumer(log);

        consumer.consume(List.of(
                new ContentEmbeddingService.LiveSlice(1001L, now - 60_000L, "http://cdn/a.jpg"),
                new ContentEmbeddingService.LiveSlice(1002L, now - 120_000L, "http://cdn/b.jpg")
        ), now);

        assertEquals(2, log.size(), "应产出 2 个向量");
        assertEquals(16, log.getEmbedding(0).size(), "向量维度应为 16");
    }

    @Test
    void testInvalidAuthorIdFiltered() {
        long now = System.currentTimeMillis();
        ContentEmbeddingService.InMemoryContentVectorLog log =
                new ContentEmbeddingService.InMemoryContentVectorLog();
        var consumer = buildConsumer(log);

        consumer.consume(List.of(
                new ContentEmbeddingService.LiveSlice(0L, now - 10_000L, "http://cdn/bad.jpg")
        ), now);

        assertEquals(0, log.size(), "invalid authorId 应被过滤");
    }

    @Test
    void testTooOldSliceFiltered() {
        long now = System.currentTimeMillis();
        ContentEmbeddingService.InMemoryContentVectorLog log =
                new ContentEmbeddingService.InMemoryContentVectorLog();
        var consumer = buildConsumer(log);

        // 超过 5 分钟
        consumer.consume(List.of(
                new ContentEmbeddingService.LiveSlice(1001L, now - 600_000L, "http://cdn/old.jpg")
        ), now);

        assertEquals(0, log.size(), "过旧的切片应被过滤");
    }

    @Test
    void testEmptyInputProducesNoOutput() {
        long now = System.currentTimeMillis();
        ContentEmbeddingService.InMemoryContentVectorLog log =
                new ContentEmbeddingService.InMemoryContentVectorLog();
        var consumer = buildConsumer(log);

        consumer.consume(List.of(), now);
        assertEquals(0, log.size(), "空输入不应产出向量");
    }

    @Test
    void testRpcErrorHandled() {
        long now = System.currentTimeMillis();
        ContentEmbeddingService.InMemoryContentVectorLog log =
                new ContentEmbeddingService.InMemoryContentVectorLog();

        // RPC 总是失败
        var consumer = new ContentEmbeddingService.LiveSliceEmbeddingConsumer(
                Executors.newFixedThreadPool(1),
                url -> List.of(0.5),
                (vals, dim, batchSize) -> { throw new RuntimeException("rpc timeout"); },
                log,
                new ContentEmbeddingService.AlwaysEnabledBucketSelector()
        );

        // 不应抛出异常
        assertDoesNotThrow(() -> consumer.consume(List.of(
                new ContentEmbeddingService.LiveSlice(1001L, now - 10_000L, "http://cdn/a.jpg")
        ), now));
        assertEquals(0, log.size(), "RPC 失败不应产出向量");
    }

    @Test
    void testDemoEndToEnd() {
        ContentEmbeddingService.InMemoryContentVectorLog result = ContentEmbeddingService.runDemo();
        assertNotNull(result, "Demo 不应返回 null");
        // 有效切片为 2 个（authorId>0 且时间在 5 分钟内）
        assertEquals(2, result.size(), "Demo 应产出 2 个向量");
    }
}
