def fixed(*trainable_params):
    def get(net):
        return ((n, p) for n, p in net.named_parameters() if n in trainable_params)
    return get

def headtail(n_head, n_tail):
    def get(net):
        pp = list(net.named_parameters())
        take_first = min(n_head, len(pp) - n_tail)
        return pp[:take_first] + pp[-n_tail:]
    return get

def all():
    def get(net):
        return net.named_parameters()
    return get

def none():
    def get(net):
        return dict()
    return get
