import heapq
import math
import struct
import zlib
from collections import Counter
from pathlib import Path

MAGIC = b"ZMH1"
FIXED_HEADER = 19


def lengths(data):
    frequencies = Counter(data)
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        return {next(iter(frequencies)): 1}
    heap = [(count, i, symbol) for i, (symbol, count) in enumerate(frequencies.items())]
    heapq.heapify(heap)
    serial = len(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        heapq.heappush(heap, (left[0] + right[0], serial, (left[2], right[2])))
        serial += 1
    result = {}

    def visit(node, depth):
        if isinstance(node, int):
            result[node] = max(1, depth)
        else:
            visit(node[0], depth + 1)
            visit(node[1], depth + 1)

    visit(heap[0][2], 0)
    return result


def codes(code_lengths):
    value = 0
    previous = 0
    result = {}
    for symbol, length in sorted(code_lengths.items(), key=lambda item: (item[1], item[0])):
        value <<= length - previous
        result[symbol] = value, length
        value += 1
        previous = length
    return result


def header(size, checksum, code_lengths):
    table = b"".join(struct.pack(">BB", symbol, length) for symbol, length in sorted(code_lengths.items()))
    return MAGIC + bytes([1]) + struct.pack(">QIH", size, checksum, len(code_lengths)) + table


def read_header(blob):
    if len(blob) < 4 or blob[:4] != MAGIC:
        raise ValueError("чужой формат")
    if len(blob) < FIXED_HEADER:
        raise ValueError("обрезанный файл")
    if blob[4] != 1:
        raise ValueError("повреждённый файл")
    size, checksum, count = struct.unpack(">QIH", blob[5:FIXED_HEADER])
    end = FIXED_HEADER + count * 2
    if len(blob) < end:
        raise ValueError("обрезанный файл")
    code_lengths = {}
    for offset in range(FIXED_HEADER, end, 2):
        symbol, length = blob[offset:offset + 2]
        if not length or symbol in code_lengths:
            raise ValueError("повреждённый файл")
        code_lengths[symbol] = length
    if bool(size) != bool(code_lengths):
        raise ValueError("повреждённый файл")
    return size, checksum, code_lengths, end


def stats(data, code_lengths, header_bytes, data_bytes):
    count = Counter(data)
    size = len(data)
    probabilities = [value / size for value in count.values()] if size else []
    entropy = -sum(p * math.log2(p) for p in probabilities) if size else 0
    average = sum(count[symbol] / size * length for symbol, length in code_lengths.items()) if size else 0
    total = header_bytes + data_bytes
    return {
        "entropy": entropy,
        "average_code_length": average,
        "header_bytes": header_bytes,
        "data_bytes": data_bytes,
        "compressed_bytes": total,
        "input_bytes": size,
        "compression_ratio": total / size if size else None,
    }


def compress(input_file, output_file):
    data = Path(input_file).read_bytes()
    code_lengths = lengths(data)
    code = codes(code_lengths)
    body = bytearray()
    value = bits = 0
    for symbol in data:
        part, width = code[symbol]
        value = value << width | part
        bits += width
        while bits >= 8:
            bits -= 8
            body.append(value >> bits & 255)
            value &= (1 << bits) - 1 if bits else 0
    if bits:
        body.append(value << (8 - bits) & 255)
    head = header(len(data), zlib.crc32(data) & 0xffffffff, code_lengths)
    Path(output_file).write_bytes(head + body)
    return stats(data, code_lengths, len(head), len(body))


def decompress(input_file, output_file):
    blob = Path(input_file).read_bytes()
    size, checksum, code_lengths, start = read_header(blob)
    reverse = {item: symbol for symbol, item in codes(code_lengths).items()}
    result = bytearray()
    value = width = 0
    for byte in blob[start:]:
        for shift in range(7, -1, -1):
            value = value << 1 | byte >> shift & 1
            width += 1
            symbol = reverse.get((value, width))
            if symbol is not None:
                result.append(symbol)
                value = width = 0
                if len(result) == size:
                    break
        if len(result) == size:
            break
    if len(result) != size:
        raise ValueError("обрезанный файл")
    data = bytes(result)
    if zlib.crc32(data) & 0xffffffff != checksum:
        raise ValueError("повреждённый файл")
    Path(output_file).write_bytes(data)
    return {"size": size, "crc32": f"{checksum:08x}"}


def file_stats(filename):
    data = Path(filename).read_bytes()
    code_lengths = lengths(data)
    total_bits = sum(Counter(data)[symbol] * length for symbol, (_, length) in codes(code_lengths).items())
    return stats(data, code_lengths, FIXED_HEADER + 2 * len(code_lengths), (total_bits + 7) // 8)
