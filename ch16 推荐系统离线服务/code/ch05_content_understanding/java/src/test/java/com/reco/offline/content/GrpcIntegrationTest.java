package com.reco.offline.content;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Java-Python gRPC 集成测试。
 *
 * 测试策略：
 *   1. 在 @BeforeAll 中启动 Python gRPC 服务（USE_MOCK_MODEL=true）
 *   2. 使用 GrpcImageEmbeddingClient 发起真实 gRPC 调用
 *   3. 在 @AfterAll 中关闭 Python 进程
 *
 * 默认通过系统属性 -Dgrpc.integration=true 才运行（避免 CI 无 Python 环境时失败）。
 * 运行方式：
 *   mvn test -Dgrpc.integration=true
 * 或者直接运行此测试类。
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class GrpcIntegrationTest {

    private static final int GRPC_PORT = 50052;  // 使用 50052 避免与已运行服务冲突
    private static final String PYTHON_SERVICE_PATH =
            Paths.get(System.getProperty("user.dir"))  // .../java
                 .getParent()                           // .../ch05_content_understanding
                 .resolve("src/qwen_vl_rpc_service.py")
                 .toString();

    private Process pythonProcess;
    private GrpcImageEmbeddingClient client;

    @BeforeAll
    void startPythonServer() throws Exception {
        // 启动 Python gRPC 服务（USE_MOCK_MODEL=true，不加载真实模型）
        ProcessBuilder pb = new ProcessBuilder(
                "python3", PYTHON_SERVICE_PATH
        );
        pb.environment().put("USE_MOCK_MODEL", "true");
        // 修改 serve() 使用的端口：通过环境变量传入
        pb.environment().put("GRPC_PORT", String.valueOf(GRPC_PORT));
        pb.redirectErrorStream(true);
        pb.redirectOutput(ProcessBuilder.Redirect.INHERIT);

        // 找到 src 目录作为工作目录
        File srcDir = new File(PYTHON_SERVICE_PATH).getParentFile();
        pb.directory(srcDir);

        try {
            pythonProcess = pb.start();
        } catch (Exception e) {
            System.out.println("[GrpcIntegrationTest] Python not available: " + e.getMessage());
            return;
        }

        // 等待服务启动（最多 10 秒）
        waitForServer(GRPC_PORT, 10);

        // 创建 gRPC 客户端
        client = new GrpcImageEmbeddingClient("localhost", GRPC_PORT, 5);
    }

    @AfterAll
    void stopPythonServer() throws Exception {
        if (client != null) {
            client.close();
        }
        if (pythonProcess != null && pythonProcess.isAlive()) {
            pythonProcess.destroy();
            pythonProcess.waitFor(5, TimeUnit.SECONDS);
        }
    }

    /**
     * 核心集成测试：Java 通过 gRPC 向 Python 服务请求 embedding，验证响应完整性。
     */
    @Test
    void testRealGrpcEmbeddingCall() throws Exception {
        if (client == null) {
            System.out.println("[GrpcIntegrationTest] Skipping: Python server not available");
            return;
        }

        int batchSize = 2;
        // 使用小尺寸（3*64*64）避免超 gRPC 默认消息大小限制
        int imageDim = 3 * 64 * 64;
        List<Double> imageValues = new ArrayList<>(batchSize * imageDim);
        for (int i = 0; i < batchSize * imageDim; i++) {
            imageValues.add(0.5);  // 归一化像素（mock）
        }

        ContentEmbeddingService.ImageEmbeddingResponse response =
                client.embed(imageValues, imageDim, batchSize);

        assertNotNull(response, "响应不应为 null");
        assertEquals(batchSize, response.batchSize(), "batch_size 应与请求一致");
        assertTrue(response.embeddingDimension() > 0, "embedding 维度应大于 0");
        assertEquals(batchSize * response.embeddingDimension(), response.values().size(),
                "向量数组长度应等于 batch_size * emb_dim");

        System.out.printf("[GrpcIntegrationTest] PASS: batch=%d, emb_dim=%d, values=%d%n",
                response.batchSize(), response.embeddingDimension(), response.values().size());
    }

    /**
     * 通过 LiveSliceEmbeddingConsumer 端到端集成测试：
     * Java 消费者 → 并发下载（mock）→ 真实 gRPC 调用 Python 服务 → 写入日志。
     */
    @Test
    void testConsumerWithRealGrpcClient() throws Exception {
        if (client == null) {
            System.out.println("[GrpcIntegrationTest] Skipping: Python server not available");
            return;
        }

        long now = System.currentTimeMillis();
        ContentEmbeddingService.InMemoryContentVectorLog log =
                new ContentEmbeddingService.InMemoryContentVectorLog();

        // mock 下载器（返回小尺寸像素，避免超 gRPC 消息大小限制）
        int imageDim = 3 * 64 * 64;
        ContentEmbeddingService.ImageDownloader mockDownloader = url -> {
            List<Double> px = new ArrayList<>(imageDim);
            for (int i = 0; i < imageDim; i++) px.add(0.5);
            return px;
        };

        ContentEmbeddingService.LiveSliceEmbeddingConsumer consumer =
                new ContentEmbeddingService.LiveSliceEmbeddingConsumer(
                        Executors.newFixedThreadPool(2),
                        mockDownloader,
                        client,  // ← 真实 gRPC 客户端
                        log,
                        new ContentEmbeddingService.AlwaysEnabledBucketSelector());

        consumer.consume(List.of(
                new ContentEmbeddingService.LiveSlice(1001L, now - 30_000L, "http://cdn/a.jpg"),
                new ContentEmbeddingService.LiveSlice(1002L, now - 60_000L, "http://cdn/b.jpg")
        ), now);

        assertEquals(2, log.size(), "应产出 2 个向量");
        assertTrue(log.getEmbedding(0).size() > 0, "向量维度应大于 0");
        System.out.printf("[GrpcIntegrationTest] PASS: consumer wrote %d embeddings, dim=%d%n",
                log.size(), log.getEmbedding(0).size());
    }

    // ── 辅助：等待端口开放 ──
    private void waitForServer(int port, int maxSeconds) throws Exception {
        for (int i = 0; i < maxSeconds * 10; i++) {
            try (java.net.Socket s = new java.net.Socket("localhost", port)) {
                return;  // 端口已开放
            } catch (java.net.ConnectException ignored) {
                Thread.sleep(100);
            }
        }
        System.out.println("[GrpcIntegrationTest] Warning: server not ready on port " + port);
    }
}
