package com.reco.offline.content;

import com.reco.offline.content.proto.ImageEmbeddingRequest;
import com.reco.offline.content.proto.ImageEmbeddingResponse;
import com.reco.offline.content.proto.ImageEmbeddingServiceGrpc;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * GrpcImageEmbeddingClient — 通过 gRPC 调用 Python Qwen2.5-VL RPC 服务。
 *
 * 对应 proto 文件：../proto/image_embedding_service.proto
 * 对应 Python 服务：../src/qwen_vl_rpc_service.py
 *
 * 使用方式：
 *   try (GrpcImageEmbeddingClient client = new GrpcImageEmbeddingClient("localhost", 50051)) {
 *       List<Double> embedding = client.embed(imageValues, imageDim, batchSize).values();
 *   }
 */
public class GrpcImageEmbeddingClient
        implements ContentEmbeddingService.ImageEmbeddingClient, AutoCloseable {

    private final ManagedChannel channel;
    private final ImageEmbeddingServiceGrpc.ImageEmbeddingServiceBlockingStub stub;
    private final int timeoutSeconds;

    public GrpcImageEmbeddingClient(String host, int port) {
        this(host, port, 5);
    }

    public GrpcImageEmbeddingClient(String host, int port, int timeoutSeconds) {
        this.channel = ManagedChannelBuilder
                .forAddress(host, port)
                .usePlaintext()
                .maxInboundMessageSize(64 * 1024 * 1024)  // 64 MB，支持大批量图片
                .build();
        this.stub = ImageEmbeddingServiceGrpc.newBlockingStub(channel);
        this.timeoutSeconds = timeoutSeconds;
    }

    /**
     * 调用 Python gRPC 服务获取图片 embedding。
     *
     * @param imageValues  展平的归一化像素数组（float），长度 = imageDimension * batchSize
     * @param imageDimension 单张图片像素维度（e.g. 3 * 448 * 448 = 602112）
     * @param batchSize    批大小
     * @return ImageEmbeddingResponse 包含向量数组、维度和批大小
     * @throws Exception gRPC 调用失败时抛出
     */
    @Override
    public ContentEmbeddingService.ImageEmbeddingResponse embed(
            List<Double> imageValues,
            int imageDimension,
            int batchSize) throws Exception {

        // 将 List<Double> 转换为 repeated float（proto 字段）
        ImageEmbeddingRequest.Builder requestBuilder = ImageEmbeddingRequest.newBuilder()
                .setImageDim(imageDimension)
                .setBatchSize(batchSize);
        for (Double v : imageValues) {
            requestBuilder.addImageList(v.floatValue());
        }
        ImageEmbeddingRequest request = requestBuilder.build();

        // 带超时的阻塞调用
        com.reco.offline.content.proto.ImageEmbeddingResponse protoResponse;
        try {
            protoResponse = stub
                    .withDeadlineAfter(timeoutSeconds, TimeUnit.SECONDS)
                    .get(request);
        } catch (StatusRuntimeException e) {
            throw new RuntimeException("gRPC call failed: " + e.getStatus(), e);
        }

        // 将 proto repeated float 转回 List<Double>
        List<Double> values = new ArrayList<>(protoResponse.getImageEmbeddingCount());
        for (float f : protoResponse.getImageEmbeddingList()) {
            values.add((double) f);
        }

        return new ContentEmbeddingService.ImageEmbeddingResponse(
                values,
                protoResponse.getEmbDim(),
                protoResponse.getBatchSize());
    }

    @Override
    public void close() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }
}
