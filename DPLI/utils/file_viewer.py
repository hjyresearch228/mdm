import os
import shutil
import struct

def load_list(path):
    with open(path, 'r') as reader:
        lines = reader.readlines()
    res = [s.strip() for s in lines]
    return res

def detect_and_create_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

def detect_and_delete_dir(dir: str):
    if os.path.exists(dir):
        shutil.rmtree(dir)

def detect_and_delete_empty_dir(dir: str):
    if os.path.exists(dir):
        os.removedirs(dir)

def save_bytes_to_file(src: str, buffer: bytes):
    with open(src, 'w') as writer:
        pass
    with open(src, 'wb+') as writer:
        writer.write(buffer)

# def save_bytes_file_(src: str, fmt: str, pack_value):
#     buffer = struct.pack(fmt, pack_value)
#     save_bytes_to_file(src, buffer)

def save_elem_to_file(src: str, elem_fmt: str, *pack_values):
    elem_num = len(pack_values)
    buffer = struct.pack(elem_fmt * elem_num, *pack_values)
    save_bytes_to_file(src, buffer)

def load_bytes_file(src: str, fmt: str) -> tuple:
    with open(src, 'rb') as reader:
        content = reader.read()
    tuple_ = struct.unpack(fmt, content)
    return tuple_

def load_bytes_file_to_list(src: str, fmt: str, unpack_size: int) -> list:
    with open(src, 'rb') as reader:
        content = reader.read()
    ele_num = int(len(content) / unpack_size)  # 元素个数 = 总长度 / 元素大小
    tuple_ = struct.unpack(fmt * ele_num, content)
    return list(tuple_)

def load_bytes_file_to_list_slow(src: str, fmt: str, unpack_size: int) -> list:
    with open(src, 'rb') as reader:
        content = reader.read()
    res = []
    for i in range(0, len(content), unpack_size):
        part_bytes = content[i: i + unpack_size]
        tuple_ = struct.unpack(fmt, part_bytes)
        res.extend(tuple_)
    return res

