import math
from fractions import Fraction
from pathlib import Path
from .source import distribution_at, stationary


def entropy(probabilities):
    return -sum(float(p) * math.log2(float(p)) for p in probabilities.values() if p)


def memoryless(source):
    if not stationary(source):
        raise ValueError("источник нестационарен")
    if source["kind"] == "mixture":
        parts = source["parts"]
        distributions = [distribution_at(part, 0) for _, part in parts]
        if not all(memoryless(part) for _, part in parts) or len({tuple(p.items()) for p in distributions}) != 1:
            raise ValueError("источник имеет память")
        return distributions[0]
    if source["kind"] in {"memoryless", "periodic"}:
        return distribution_at(source, 0)
    raise ValueError("источник имеет память")


def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for tail in compositions(total - first, parts - 1):
            yield (first,) + tail


def type_count(counts):
    value = math.factorial(sum(counts))
    for count in counts:
        value //= math.factorial(count)
    return value


def error_probability(probabilities, q, n, rate):
    length = max(0, math.floor(n * rate / math.log2(q) + 1e-12))
    capacity, success = q ** length, Fraction()
    types = []
    for counts in compositions(n, len(probabilities)):
        probability = math.prod(p ** count for p, count in zip(probabilities, counts))
        types.append((probability, type_count(counts)))
    selected = 0
    for probability, amount in sorted(types, reverse=True):
        take = min(amount, capacity - selected)
        success += take * probability
        selected += take
        if selected == capacity:
            break
    return length, Fraction(1) - success


def analyze(source, rate, epsilon, q, nmax):
    distribution = memoryless(source)
    rows, n_min = [], None
    probabilities = [distribution[symbol] for symbol in source["a"]]
    for n in range(1, nmax + 1):
        length, error = error_probability(probabilities, q, n, rate)
        n_min = n if n_min is None and error <= epsilon else n_min
        rows.append({"n": n, "ell": length, "Pe": float(error), "Pe_fraction": str(error)})
    return {"distribution": {s: str(distribution[s]) for s in source["a"]}, "H": entropy(distribution),
            "rate": rate, "q": q, "epsilon": epsilon, "n_min": n_min,
            "eventually_tends_to_zero": rate > entropy(distribution), "rows": rows}


def write_code(filename, result):
    lines = ["# optimal type selection", f"alphabet={list(result['distribution'])}",
             f"distribution={result['distribution']}", f"q={result['q']}", f"rate={result['rate']}", ""]
    lines += [f"n={row['n']} ell={row['ell']} Pe={row['Pe_fraction']}" for row in result["rows"]]
    Path(filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
