import json
import tempfile
import unittest
from pathlib import Path

from server.config import _validate, get_level_ids, get_savestate_files, load_config
from server.main import _backend_override_for_mode


def minimal_config() -> dict:
    return {
        "paths": {
            "bizhawk_exe": "./BizHawk/EmuHawk.exe",
            "rom": "./roms/Super Mario World.sfc",
            "savestate_dir": "./savestates",
            "model_dir": "./models",
            "log_dir": "./logs",
            "video_dir": "./videos",
            "lua_script": "./lua/smw_agent.lua",
        },
        "emulator": {
            "num_instances": 1,
            "base_port": 9000,
            "frame_skip": 4,
        },
        "normalization": {
            "grid_size": 21,
            "tile_categories_count": 16,
        },
        "ppo": {
            "n_steps": 128,
            "batch_size": 64,
        },
        "level_loading": {
            "savestate_files": [],
        },
    }


class ConfigTests(unittest.TestCase):
    def test_load_config_resolves_paths_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(minimal_config()), encoding="utf-8")

            config = load_config(str(config_path))

            self.assertEqual(
                Path(config["paths"]["rom"]),
                root / "roms" / "Super Mario World.sfc",
            )

    def test_explicit_savestate_name_is_relative_to_savestate_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            savestate_dir = Path(temp_dir) / "savestates"
            savestate_dir.mkdir()
            state_path = savestate_dir / "level-one.State"
            state_path.write_bytes(b"state")
            config = {
                "paths": {"savestate_dir": str(savestate_dir)},
                "level_loading": {"savestate_files": ["level-one.State"]},
            }

            self.assertEqual(get_savestate_files(config), [state_path.as_posix()])

    def test_grid_size_must_be_odd(self):
        config = minimal_config()
        config["normalization"]["grid_size"] = 20

        with self.assertRaisesRegex(ValueError, "grid_size"):
            _validate(config)

    def test_sprite_hitbox_normalization_scale_must_be_positive(self):
        config = minimal_config()
        config["normalization"]["max_sprite_hitbox_dimension"] = 0

        with self.assertRaisesRegex(ValueError, "max_sprite_hitbox_dimension"):
            _validate(config)

    def test_performance_thread_counts_must_be_positive_integers(self):
        for setting in ("torch_threads", "retrojet_threads"):
            for invalid_value in (0, -1, 2.5, "4", True):
                with self.subTest(setting=setting, value=invalid_value):
                    config = minimal_config()
                    config["performance"] = {setting: invalid_value}

                    with self.assertRaisesRegex(ValueError, setting):
                        _validate(config)

    def test_rollout_size_must_be_divisible_by_batch_size(self):
        config = minimal_config()
        config["ppo"]["n_steps"] = 100
        config["ppo"]["batch_size"] = 64

        with self.assertRaisesRegex(ValueError, "divisible"):
            _validate(config)

    def test_lunar_magic_level_ids_are_parsed_as_hex_strings(self):
        config = {"level_loading": {"levels": ["0x105", "106"]}}

        self.assertEqual(get_level_ids(config), [0x105, 0x106])

    def test_full_level_loading_rejects_decimal_lunar_magic_ids(self):
        config = minimal_config()
        config["level_loading"] = {
            "mode": "level_loading",
            "levels": [105, 106],
        }

        with self.assertRaisesRegex(ValueError, 'quote them, for example "0x105"'):
            _validate(config)

    def test_full_level_loading_accepts_overworld_entry_levels(self):
        config = minimal_config()
        config["level_loading"] = {
            "mode": "level_loading",
            "levels": ["0x105", "0x106"],
        }

        _validate(config)

    def test_retrojet_backend_uses_retrojet_env_count_for_rollout_validation(self):
        config = minimal_config()
        config["emulator"]["num_instances"] = 1
        config["ppo"]["n_steps"] = 128
        config["ppo"]["batch_size"] = 256
        config["backend"] = {
            "type": "retrojet",
            "retrojet": {
                "num_envs": 2,
            },
        }

        _validate(config)

    def test_backend_override_is_applied_before_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = minimal_config()
            config["backend"] = {"type": "bizhawk", "retrojet": {"num_envs": 2}}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            loaded = load_config(str(config_path), backend="retrojet")

            self.assertEqual(loaded["backend"]["type"], "retrojet")

    def test_live_demo_validates_the_forced_retrojet_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = minimal_config()
            config["backend"] = {
                "type": "bizhawk",
                "retrojet": {"num_envs": 0},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            backend = _backend_override_for_mode("live-demo", None)
            with self.assertRaisesRegex(ValueError, "num_envs"):
                load_config(str(config_path), backend=backend)

    def test_conflicting_retrojet_rom_override_is_rejected(self):
        config = minimal_config()
        config["backend"] = {
            "type": "retrojet",
            "retrojet": {"rom_path": "./roms/another.sfc"},
        }

        with self.assertRaisesRegex(ValueError, "configure the ROM once"):
            _validate(config)

    def test_conflicting_retrojet_level_override_is_rejected(self):
        config = minimal_config()
        config["level_loading"] = {
            "mode": "level_loading",
            "levels": ["0x105"],
        }
        config["backend"] = {
            "type": "retrojet",
            "retrojet": {"level_ids": ["0x106"]},
        }

        with self.assertRaisesRegex(ValueError, "configure levels once"):
            _validate(config)


if __name__ == "__main__":
    unittest.main()
