# image_embedding_service_pb2.py
# 手写的 proto 消息类（兼容任意 grpcio 版本，无需 protoc-gen）
# 对应 proto/image_embedding_service.proto
#
# 实现了 protobuf wire format 的最小子集：
#   - field 1: repeated float image_list (packed, tag=0x0A)
#   - field 2: int32 image_dim (tag=0x10)
#   - field 3: int32 batch_size (tag=0x18)
#   - field 1: repeated float image_embedding (packed, tag=0x0A)
#   - field 2: int32 emb_dim (tag=0x10)
#   - field 3: int32 batch_size (tag=0x18)

import struct


def _encode_varint(value: int) -> bytes:
    """将 int 编码为 protobuf varint。"""
    result = b""
    while value > 0x7F:
        result += bytes([(value & 0x7F) | 0x80])
        value >>= 7
    result += bytes([value])
    return result


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """从 data[pos] 开始解码 varint，返回 (value, new_pos)。"""
    result, shift = 0, 0
    while True:
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


class ImageEmbeddingRequest:
    """对应 proto ImageEmbeddingRequest。"""

    def __init__(self, image_list=None, image_dim=0, batch_size=0):
        self.image_list = list(image_list) if image_list else []
        self.image_dim = image_dim
        self.batch_size = batch_size

    def SerializeToString(self) -> bytes:
        result = b""
        if self.image_list:
            # field 1, wire type 2 (length-delimited packed floats)
            payload = struct.pack(f"<{len(self.image_list)}f", *self.image_list)
            result += b"\x0A" + _encode_varint(len(payload)) + payload
        if self.image_dim:
            result += b"\x10" + _encode_varint(self.image_dim)
        if self.batch_size:
            result += b"\x18" + _encode_varint(self.batch_size)
        return result

    @classmethod
    def FromString(cls, data: bytes) -> "ImageEmbeddingRequest":
        obj = cls()
        pos = 0
        while pos < len(data):
            tag, pos = _decode_varint(data, pos)
            field_number = tag >> 3
            wire_type = tag & 0x7
            if field_number == 1 and wire_type == 2:
                length, pos = _decode_varint(data, pos)
                floats = struct.unpack(f"<{length // 4}f", data[pos:pos + length])
                obj.image_list.extend(floats)
                pos += length
            elif field_number == 2 and wire_type == 0:
                obj.image_dim, pos = _decode_varint(data, pos)
            elif field_number == 3 and wire_type == 0:
                obj.batch_size, pos = _decode_varint(data, pos)
            else:
                # skip unknown field
                if wire_type == 0:
                    _, pos = _decode_varint(data, pos)
                elif wire_type == 2:
                    length, pos = _decode_varint(data, pos)
                    pos += length
                elif wire_type == 5:
                    pos += 4
                else:
                    break
        return obj


class ImageEmbeddingResponse:
    """对应 proto ImageEmbeddingResponse。"""

    def __init__(self, image_embedding=None, emb_dim=0, batch_size=0):
        self.image_embedding = list(image_embedding) if image_embedding else []
        self.emb_dim = emb_dim
        self.batch_size = batch_size

    def SerializeToString(self) -> bytes:
        result = b""
        if self.image_embedding:
            payload = struct.pack(f"<{len(self.image_embedding)}f", *self.image_embedding)
            result += b"\x0A" + _encode_varint(len(payload)) + payload
        if self.emb_dim:
            result += b"\x10" + _encode_varint(self.emb_dim)
        if self.batch_size:
            result += b"\x18" + _encode_varint(self.batch_size)
        return result

    @classmethod
    def FromString(cls, data: bytes) -> "ImageEmbeddingResponse":
        obj = cls()
        pos = 0
        while pos < len(data):
            tag, pos = _decode_varint(data, pos)
            field_number = tag >> 3
            wire_type = tag & 0x7
            if field_number == 1 and wire_type == 2:
                length, pos = _decode_varint(data, pos)
                floats = struct.unpack(f"<{length // 4}f", data[pos:pos + length])
                obj.image_embedding.extend(floats)
                pos += length
            elif field_number == 2 and wire_type == 0:
                obj.emb_dim, pos = _decode_varint(data, pos)
            elif field_number == 3 and wire_type == 0:
                obj.batch_size, pos = _decode_varint(data, pos)
            else:
                if wire_type == 0:
                    _, pos = _decode_varint(data, pos)
                elif wire_type == 2:
                    length, pos = _decode_varint(data, pos)
                    pos += length
                elif wire_type == 5:
                    pos += 4
                else:
                    break
        return obj
