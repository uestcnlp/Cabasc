def samples2list(samples, attr):
    rets = []
    for sample in samples:
        rets.append(getattr(sample, attr))
    return rets