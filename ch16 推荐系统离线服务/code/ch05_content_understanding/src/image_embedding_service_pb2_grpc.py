# image_embedding_service_pb2_grpc.py
# 手写的 gRPC servicer/stub（兼容 grpcio >= 1.32）
# 对应 proto/image_embedding_service.proto

import grpc

import image_embedding_service_pb2 as _api

SERVICE_NAME = "image_embedding.ImageEmbeddingService"


class ImageEmbeddingServiceStub:
    """Python gRPC client stub（供测试使用）。"""

    def __init__(self, channel):
        self.Get = channel.unary_unary(
            f"/{SERVICE_NAME}/Get",
            request_serializer=_api.ImageEmbeddingRequest.SerializeToString,
            response_deserializer=_api.ImageEmbeddingResponse.FromString,
        )


class ImageEmbeddingServiceServicer:
    """gRPC server servicer 基类。子类实现 Get 方法。"""

    def Get(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


class _ImageEmbeddingGenericHandler(grpc.ServiceRpcHandler):
    """通用 RPC handler — 兼容 grpcio 1.32+ 的注册方式。"""

    def __init__(self, servicer):
        self._servicer = servicer

    def service_name(self):
        return SERVICE_NAME

    def service(self, handler_call_details):
        method = handler_call_details.method
        if method.endswith("/Get"):
            return grpc.unary_unary_rpc_method_handler(
                self._servicer.Get,
                request_deserializer=_api.ImageEmbeddingRequest.FromString,
                response_serializer=_api.ImageEmbeddingResponse.SerializeToString,
            )
        return None


def add_ImageEmbeddingServiceServicer_to_server(servicer, server):
    """将 servicer 注册到 gRPC server。"""
    server.add_generic_rpc_handlers((_ImageEmbeddingGenericHandler(servicer),))
