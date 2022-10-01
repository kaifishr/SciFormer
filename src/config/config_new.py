from __future__ import annotations
from dataclasses import dataclass

from attr import has


# @dataclass
class Cfg:
    """Configuration class.

    Creates configuration from dictionary for better handling.
    TODO: Use dataclass for this?
    """

    # def __init__(self):
    #     """Initializes config class."""
    #     pass

    @staticmethod
    def _crawl_dict(self: object, d: object) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                if not hasattr(self, key):
                    self.__setattr__(key, Cfg())
                self._crawl_dict(self=self.__getattribute__(key), d=value)
            else:
                self.__setattr__(key, value)

    def merge_from_dict(self, d: dict) -> None:
        self._crawl_dict(self, d)

    def __str__(self) -> str:
        """Prints nested config."""
        cfg = []
        self._print_helper(self, cfg)
        return "".join(cfg)

    @staticmethod
    def _print_helper(self: object, cfg: list, indent: int = 0) -> None:
        """Recursively iterates through all configuration nodes."""
        for key, value in self.__dict__.items():
            indent_ = 4 * indent * " "
            if isinstance(value, Cfg):
                cfg.append(f"{indent_}{key}\n")
                self._print_helper(
                    self=self.__getattribute__(key), cfg=cfg, indent=indent + 1
                )
            else:
                cfg.append(f"{indent_}{key}: {value}\n")


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
            },
        }
    }

    config2 = {
        "cfg_transformer": {
            "cfg_img2seq": {
                "patch_size": 3,
            },
            "cfg_transformer_block": {
                "cfg_multi_head_self_att": {
                    "n_heads": 64,
                },
                "hidden_expansion": 3,
            },
        }
    }

    config3 = {
        "cfg_img2seq": {
            "foo": 69,
        },
        "cfg_transformer_block": {
            "cfg_multi_head_self_att": {
                "n_heads": 128,
            },
            "hidden_foo": 8,
        },
    }

    cfg = Cfg()
    cfg.merge_from_dict(config)
    print(cfg.cfg_transformer.cfg_img2seq)
    print(cfg)
    print(cfg.cfg_transformer)

    cfg.merge_from_dict(config2)
    print(cfg)

    cfg.cfg_transformer.merge_from_dict(config3)
    print(cfg)


if __name__ == "__main__":
    main()