import yaml


def _get_key(x, y):
    pair = sorted([x, y])
    key = ''.join(pair)
    return key


class RDMCache:

    def __init__(self):
        self.cache_dict = {}

    def load_from_file(self, fp_cache):
        with open(fp_cache, 'r') as h:
            self.cache_dict = yaml.safe_load(h)

    def save_to_file(self, fp_dst):
        with open(fp_dst, 'w') as h:
            h.write(yaml.dump(self.cache_dict))

    # def is_in(self, x, y):
    #
    #     key = self._get_key(x, y)
    #     return key in self.cache_dict

    def add(self, x, y, value):
        key = _get_key(x, y)
        self.cache_dict[key] = value

    def get(self, x, y, default_value):
        key = _get_key(x, y)
        return self.cache_dict.get(key, default_value)
