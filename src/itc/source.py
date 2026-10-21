import json
import math
import random
from fractions import Fraction
from itertools import product
from pathlib import Path


def frac(value):
    try:
        return Fraction(str(value).strip().replace("∕", "/").replace(" ", ""))
    except (ValueError, ZeroDivisionError, TypeError):
        raise ValueError(f"некорректная вероятность {value!r}")


def dist(raw, name="distribution"):
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"пустое распределение: {name}")
    result = {str(key): frac(value) for key, value in raw.items()}
    if any(value < 0 for value in result.values()) or sum(result.values(), Fraction()) != 1:
        raise ValueError(f"некорректное распределение: {name}")
    return result


def choose(items, rng):
    point, total = rng.random(), 0
    last = next(iter(items))
    for key, probability in items.items():
        last, total = key, total + float(probability)
        if point < total:
            return key
    return last


def key(context):
    return "|".join(context)


def mixed(raw, models, alphabet, name):
    weights = dist(raw, name)
    if any(model not in models for model in weights):
        raise ValueError(f"неизвестная модель в {name}")
    return {symbol: sum((weight * models[model][symbol] for model, weight in weights.items()), Fraction())
            for symbol in alphabet}


def parse(data):
    if not isinstance(data, dict):
        raise ValueError("описание источника должно быть объектом")
    if "models" in data:
        models = {str(name): dist(value, f"model {name}") for name, value in data.get("models", {}).items()}
        switches, names = data.get("switches"), data.get("source")
        if not models or not isinstance(switches, dict) or not isinstance(names, list) or not names:
            raise ValueError("нужны models, switches и source")
        alphabet = list(next(iter(models.values())))
        if any(set(model) != set(alphabet) for model in models.values()):
            raise ValueError("алфавиты моделей различаются")
        switches = {str(name): mixed(raw, models, alphabet, f"switch {name}")
                    for name, raw in switches.items()}
        if any(str(name) not in switches for name in names):
            raise ValueError("неизвестный переключатель в source")
        return {"kind": "periodic", "a": alphabet, "p": [switches[str(name)] for name in names]}

    kind = data.get("kind", "memoryless")
    if kind == "memoryless":
        probabilities = dist(data.get("distribution", {}))
        alphabet = [str(x) for x in data.get("alphabet", probabilities)]
        if set(alphabet) != set(probabilities):
            raise ValueError("alphabet и distribution различаются")
        return {"kind": kind, "a": alphabet, "p": [probabilities]}
    if kind == "periodic":
        values = data.get("distributions")
        if not isinstance(values, list) or not values:
            raise ValueError("для periodic нужны distributions")
        values = [dist(value, f"distributions[{i}]") for i, value in enumerate(values)]
        alphabet = [str(x) for x in data.get("alphabet", values[0])]
        if any(set(alphabet) != set(value) for value in values):
            raise ValueError("алфавиты periodic различаются")
        return {"kind": kind, "a": alphabet, "p": values}
    if kind == "markov":
        return markov(data)
    if kind == "mixture":
        parts = data.get("components")
        if not isinstance(parts, list) or not parts:
            raise ValueError("для mixture нужны components")
        weights = dist({str(i): part.get("weight") for i, part in enumerate(parts)}, "mixture")
        sources = [parse(part.get("source", {})) for part in parts]
        alphabet = [str(x) for x in data.get("alphabet", sources[0]["a"])]
        if any(source["a"] != alphabet for source in sources):
            raise ValueError("алфавиты mixture различаются")
        period = math.lcm(*(len(source["p"]) if source["kind"] in {"memoryless", "periodic"} else 1
                           for source in sources))
        return {"kind": kind, "a": alphabet, "parts": list(zip(weights.values(), sources)), "period": period}
    raise ValueError(f"неизвестный тип источника {kind!r}")


def markov(data):
    memory = int(data.get("memory", 0))
    alphabet = [str(x) for x in data.get("alphabet", [])]
    initial = dist(data.get("initial", {}), "initial")
    raw = {str(context): dist(value, f"transition {context}")
           for context, value in data.get("transitions", {}).items()}
    expected = {key(context) for context in product(alphabet, repeat=memory)}
    if memory < 1 or not alphabet or not initial or not set(initial) <= expected or not set(raw) <= expected:
        raise ValueError("некорректный марковский источник")
    transitions = {context: {symbol: Fraction(1, len(alphabet)) for symbol in alphabet} for context in expected}
    transitions.update(raw)
    if any(set(value) != set(alphabet) for value in transitions.values()):
        raise ValueError("некорректный алфавит перехода")
    return {"kind": "markov", "a": alphabet, "d": memory, "initial": initial,
            "transitions": transitions, "period": 1}


def state_distribution(source, index):
    states = source["initial"].copy()
    for _ in range(index):
        next_states = {}
        for context, probability in states.items():
            history = context.split("|")
            for symbol, weight in source["transitions"][context].items():
                context = key(history[1:] + [symbol])
                next_states[context] = next_states.get(context, Fraction()) + probability * weight
        states = next_states
    return states


def distribution_at(source, index):
    if source["kind"] in {"memoryless", "periodic"}:
        return source["p"][index % len(source["p"])]
    if source["kind"] == "mixture":
        result = {symbol: Fraction() for symbol in source["a"]}
        for weight, part in source["parts"]:
            for symbol, probability in distribution_at(part, index).items():
                result[symbol] += weight * probability
        return result
    result = {symbol: Fraction() for symbol in source["a"]}
    position = min(index, source["d"] - 1)
    for context, probability in state_distribution(source, index).items():
        result[context.split("|")[position]] += probability
    return result


def block_probability(source, sequence, start=0):
    if any(symbol not in source["a"] for symbol in sequence):
        return Fraction()
    if source["kind"] in {"memoryless", "periodic"}:
        result = Fraction(1)
        for index, symbol in enumerate(sequence):
            result *= distribution_at(source, start + index).get(symbol, Fraction())
        return result
    if source["kind"] == "mixture":
        return sum((weight * block_probability(part, sequence, start) for weight, part in source["parts"]), Fraction())
    if not sequence:
        return Fraction(1)
    result, memory = Fraction(), source["d"]
    prefix = min(memory, len(sequence))
    for context, probability in source["initial"].items():
        history = context.split("|")
        if sequence[:prefix] != history[:prefix]:
            continue
        for symbol in sequence[memory:]:
            probability *= source["transitions"][key(history)].get(symbol, Fraction())
            history = history[1:] + [symbol]
        result += probability
    return result


def stationary(source):
    kind = source["kind"]
    if kind in {"memoryless", "periodic"}:
        return all(source["p"][0] == value for value in source["p"])
    if kind == "mixture":
        return all(stationary(part) for _, part in source["parts"])
    return state_distribution(source, 1) == source["initial"]


def memoryless(source):
    if not stationary(source):
        raise ValueError("source is not stationary")
    if source["kind"] in {"memoryless", "periodic"}:
        return distribution_at(source, 0)
    if source["kind"] == "mixture":
        distributions = [distribution_at(part, 0) for _, part in source["parts"]]
        if all(memoryless(part) for _, part in source["parts"]) and len({tuple(item.items()) for item in distributions}) == 1:
            return distributions[0]
    raise ValueError("source has memory")


def sample(source, length, seed):
    if length is None and source["kind"] == "markov":
        raise ValueError("для markov нужен --length")
    rng = random.Random(seed)
    if source["kind"] == "mixture":
        weights = {str(i): weight for i, (weight, _) in enumerate(source["parts"])}
        index = int(choose(weights, rng))
        return sample(source["parts"][index][1], length, rng.randrange(2**63))
    if length is not None and length < 0:
        raise ValueError("длина не может быть отрицательной")
    if source["kind"] == "markov":
        if not length:
            return []
        history = choose(source["initial"], rng).split("|")
        result = history[:length]
        while len(result) < length:
            symbol = choose(source["transitions"][key(history)], rng)
            result.append(symbol)
            history = history[1:] + [symbol]
        return result
    if length is None:
        result = []
        try:
            while True:
                result.append(choose(distribution_at(source, len(result)), rng))
        except KeyboardInterrupt:
            return result
    return [choose(distribution_at(source, index), rng) for index in range(length)]


def load(filename):
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    return parse(data)
