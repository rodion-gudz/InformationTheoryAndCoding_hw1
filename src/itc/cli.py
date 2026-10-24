import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

from .archive import compress, decompress, file_stats
from .source import block_probability, load, sample
from .uniform import analyze, entropy, memoryless, write_code


def sequence(values):
    if len(values) == 1 and values[0].startswith("["):
        value = json.loads(values[0])
        if not isinstance(value, list):
            raise ValueError("последовательность должна быть JSON-списком")
        return [str(item) for item in value]
    return values


def source_generate(args):
    print("".join(sample(load(args.input), args.length, args.seed)))


def source_probability(args):
    source, target = load(args.input), sequence(args.sequence)
    if args.n <= 0:
        raise ValueError("N должно быть положительным")
    exact_first, exact_average = block_probability(source, target), average(source, target)
    realization = sample(source, args.n, args.seed)
    denominator = args.n - len(target) + 1
    hits = sum(realization[i:i + len(target)] == target for i in range(max(0, denominator)))
    estimate = Fraction(hits, denominator) if denominator > 0 else Fraction()
    print(json.dumps({"exact_first": str(exact_first), "exact_period_average": str(exact_average),
                      "frequency": f"{float(estimate):.12g}", "frequency_fraction": str(estimate),
                      "N": args.n, "sequence": target}, ensure_ascii=False, indent=2))
    for n in [100, 1000, 10000, 100000, 1000000]:
        realization = sample(source, n, args.seed)
        denominator = n - len(target) + 1
        value = sum(realization[i:i + len(target)] == target for i in range(denominator)) / denominator
        print(f"N={n} estimate={value:.8f} d_first={abs(value-float(exact_first)):.8f} d_average={abs(value-float(exact_average)):.8f}")


def average(source, sequence):
    period = source.get("period", len(source.get("p", [1])))
    return sum((block_probability(source, sequence, i) for i in range(period)), Fraction()) / period


def uniform_entropy(args):
    distribution = memoryless(load(args.input))
    print(f"H={entropy(distribution):.12f}")
    print(json.dumps({key: str(value) for key, value in distribution.items()}, ensure_ascii=False))


def uniform_analyze(args):
    result = analyze(load(args.input), args.rate, args.epsilon, args.q, args.nmax)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.code_out:
        write_code(args.code_out, result)


def huffman_compress(args):
    source = Path(args.input)
    target = Path(args.output) if args.output else source.with_suffix(source.suffix + ".zmh")
    result = compress(source, target)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.stats else target)


def huffman_decompress(args):
    source = Path(args.input)
    if source.suffix != ".zmh":
        raise ValueError("чужой формат")
    target = Path(args.output) if args.output else source.with_suffix("")
    print(json.dumps(decompress(source, target), ensure_ascii=False))


def huffman_stats(args):
    print(json.dumps(file_stats(args.input), ensure_ascii=False, indent=2))


def parser():
    root = argparse.ArgumentParser(description="Домашнее задание по теории информации")
    root.add_argument("--variant", type=int, default=7)
    groups = root.add_subparsers(dest="group", required=True)
    source = groups.add_parser("source").add_subparsers(dest="command", required=True)
    generate = source.add_parser("generate")
    generate.add_argument("--input", required=True); generate.add_argument("--length", type=int); generate.add_argument("--seed", type=int, default=0)
    generate.set_defaults(run=source_generate)
    probability = source.add_parser("probability")
    probability.add_argument("--input", required=True); probability.add_argument("--n", type=int, required=True); probability.add_argument("--seed", type=int, default=0); probability.add_argument("sequence", nargs="+")
    probability.set_defaults(run=source_probability)

    uniform = groups.add_parser("uniform").add_subparsers(dest="command", required=True)
    entropy_command = uniform.add_parser("entropy"); entropy_command.add_argument("--input", required=True); entropy_command.set_defaults(run=uniform_entropy)
    analyze_command = uniform.add_parser("analyze")
    analyze_command.add_argument("--input", required=True); analyze_command.add_argument("--rate", type=float, required=True); analyze_command.add_argument("--epsilon", type=float, required=True); analyze_command.add_argument("--q", type=int, required=True); analyze_command.add_argument("--nmax", type=int, default=50); analyze_command.add_argument("--code-out")
    analyze_command.set_defaults(run=uniform_analyze)

    archive = groups.add_parser("huffman").add_subparsers(dest="command", required=True)
    compress_command = archive.add_parser("compress"); compress_command.add_argument("input"); compress_command.add_argument("--output"); compress_command.add_argument("--stats", action="store_true"); compress_command.set_defaults(run=huffman_compress)
    decompress_command = archive.add_parser("decompress"); decompress_command.add_argument("input"); decompress_command.add_argument("--output"); decompress_command.set_defaults(run=huffman_decompress)
    stats_command = archive.add_parser("stats"); stats_command.add_argument("input"); stats_command.set_defaults(run=huffman_stats)
    return root


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        return args.run(args) or 0
    except Exception as error:
        print(f"ошибка: {error}", file=sys.stderr)
        return 2
