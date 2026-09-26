package com.reco.offline.content.proto;

import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.MethodDescriptor;
import io.grpc.protobuf.ProtoUtils;
import io.grpc.stub.AbstractBlockingStub;
import io.grpc.stub.ClientCalls;

import com.google.protobuf.Descriptors;
import com.google.protobuf.DynamicMessage;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

/**
 * 手写的 gRPC 服务 Stub — 对应 proto image_embedding_service.proto。
 *
 * 使用 raw bytes（varint 手动编解码）实现 gRPC proto 通信，
 * 无需 protoc-gen-grpc-java，仅依赖 grpc-stub + protobuf-java。
 *
 * 工业落地时替换为 protoc 自动生成的 ImageEmbeddingServiceGrpc 类。
 */
public final class ImageEmbeddingServiceGrpc {

    private static final String SERVICE_NAME = "image_embedding.ImageEmbeddingService";

    private ImageEmbeddingServiceGrpc() {}

    /** 手写的请求序列化：将 ImageEmbeddingRequest 序列化为 proto wire format */
    private static byte[] serializeRequest(ImageEmbeddingRequest req) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        CodedOutputStream cos = CodedOutputStream.newInstance(baos);
        // field 1: repeated float image_list (tag = 1 << 3 | 2 = 0x0a for length-delimited)
        // 使用 packed encoding: tag=0x0A(field 1, wire type 2), then varint length, then floats
        if (!req.getImageListList().isEmpty()) {
            cos.writeRawByte(0x0A);  // field 1, wire type 2 (length-delimited)
            int byteCount = req.getImageListList().size() * 4;
            cos.writeRawVarint32(byteCount);
            for (float f : req.getImageListList()) {
                cos.writeRawLittleEndian32(Float.floatToRawIntBits(f));
            }
        }
        // field 2: int32 image_dim (tag = 0x10)
        if (req.getImageDim() != 0) {
            cos.writeRawByte(0x10);  // field 2, wire type 0 (varint)
            cos.writeRawVarint32(req.getImageDim());
        }
        // field 3: int32 batch_size (tag = 0x18)
        if (req.getBatchSize() != 0) {
            cos.writeRawByte(0x18);  // field 3, wire type 0
            cos.writeRawVarint32(req.getBatchSize());
        }
        cos.flush();
        return baos.toByteArray();
    }

    /** 手写的响应反序列化：从 proto wire format 解析 ImageEmbeddingResponse */
    private static ImageEmbeddingResponse deserializeResponse(byte[] bytes) throws IOException {
        ImageEmbeddingResponse.Builder builder = ImageEmbeddingResponse.newBuilder();
        CodedInputStream cis = CodedInputStream.newInstance(bytes);
        while (!cis.isAtEnd()) {
            int tag = cis.readTag();
            int fieldNumber = tag >>> 3;
            int wireType = tag & 0x7;
            switch (fieldNumber) {
                case 1 -> {
                    // repeated float image_embedding (packed)
                    if (wireType == 2) {
                        int length = cis.readRawVarint32();
                        int limit = cis.pushLimit(length);
                        java.util.List<Float> floats = new java.util.ArrayList<>();
                        while (cis.getBytesUntilLimit() > 0) {
                            floats.add(Float.intBitsToFloat(cis.readRawLittleEndian32()));
                        }
                        cis.popLimit(limit);
                        builder.addAllImageEmbedding(floats);
                    } else if (wireType == 5) {
                        // non-packed float
                        builder.addAllImageEmbedding(java.util.List.of(
                                Float.intBitsToFloat(cis.readRawLittleEndian32())));
                    }
                }
                case 2 -> builder.setEmbDim(cis.readRawVarint32());
                case 3 -> builder.setBatchSize(cis.readRawVarint32());
                default -> cis.skipField(tag);
            }
        }
        return builder.build();
    }

    /**
     * 获取 blocking stub — 对应生成代码中的 newBlockingStub(channel)。
     */
    public static ImageEmbeddingServiceBlockingStub newBlockingStub(Channel channel) {
        return new ImageEmbeddingServiceBlockingStub(channel, CallOptions.DEFAULT);
    }

    /**
     * Blocking stub — 同步调用 ImageEmbeddingService.Get RPC。
     */
    public static final class ImageEmbeddingServiceBlockingStub {

        private final Channel channel;
        private final CallOptions callOptions;

        private ImageEmbeddingServiceBlockingStub(Channel channel, CallOptions callOptions) {
            this.channel = channel;
            this.callOptions = callOptions;
        }

        public ImageEmbeddingServiceBlockingStub withDeadlineAfter(long duration, java.util.concurrent.TimeUnit unit) {
            return new ImageEmbeddingServiceBlockingStub(channel,
                    callOptions.withDeadlineAfter(duration, unit));
        }

        /**
         * 调用 Get RPC。
         */
        public ImageEmbeddingResponse get(ImageEmbeddingRequest request) throws IOException {
            // 序列化请求
            byte[] requestBytes = serializeRequest(request);

            // 构造 MethodDescriptor（使用 raw bytes marshaller）
            MethodDescriptor<byte[], byte[]> methodDescriptor =
                    MethodDescriptor.<byte[], byte[]>newBuilder()
                            .setType(MethodDescriptor.MethodType.UNARY)
                            .setFullMethodName(SERVICE_NAME + "/Get")
                            .setRequestMarshaller(ByteArrayMarshaller.INSTANCE)
                            .setResponseMarshaller(ByteArrayMarshaller.INSTANCE)
                            .build();

            byte[] responseBytes = ClientCalls.blockingUnaryCall(
                    channel.newCall(methodDescriptor, callOptions), requestBytes);

            return deserializeResponse(responseBytes);
        }
    }

    /** 简单的 byte[] marshaller for gRPC */
    private enum ByteArrayMarshaller implements MethodDescriptor.Marshaller<byte[]> {
        INSTANCE;

        @Override
        public InputStream stream(byte[] value) {
            return new ByteArrayInputStream(value);
        }

        @Override
        public byte[] parse(InputStream stream) {
            try {
                return stream.readAllBytes();
            } catch (IOException e) {
                throw new RuntimeException("Failed to read gRPC response", e);
            }
        }
    }
}
