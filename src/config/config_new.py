from __future__ import annotations


class Cfg:

    def __init__(self, ):
        pass

    @staticmethod
    def _dict_crawler(self: object, d: object, attr: object = None) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                self.__setattr__(key, Cfg())
                self._dict_crawler(self=self.__getattribute__(key), d=value, attr=key)
            else:
                self.__setattr__(key, value)

    def merge_from_dict(self, d: dict) -> None:
        self._dict_crawler(self, d)

    def __str__(self):
        """TODO: Print all nested configs from current node."""
        return "" 


def main():

    config = {
        "cfg_transformer": {
            "cfg_img2seq": {
                "patch_size": 2,
                "n_channels": 1,
            },
            "cfg_transformer_block": {
                "cfg_multi_head_self_att": {
                    "n_heads": 32,
                    "seq_size": 16,
                },
                "hidden_expansion": 3,
            },
            "cfg_classification": {
                "bla": 3,
            }
        }
    }

    cfg = Cfg()
    cfg.merge_from_dict(config)

    print(cfg.cfg_transformer.cfg_img2seq)
    print(cfg.cfg_transformer.cfg_transformer_block.cfg_multi_head_self_att.n_heads)
    print(dir(cfg))


if __name__ == "__main__":
    main()
