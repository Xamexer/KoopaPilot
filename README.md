<p align="center">
  <img src="docs/images/logo.png" alt="KoopaPilot logo" width="128">
</p>

<h1 align="center">KoopaPilot</h1>

<p align="center">
  A reinforcement learning project that teaches a PPO policy to play <em>Super Mario World</em> through BizHawk/Lua or the headless RetroJet libretro backend.
</p>

## About the Project

I have been passionate about _Super Mario World_ for a long time and have been active in the SMW scene since 2014. Over the years, I have contributed to and organized countless SMW-related projects. Eventually, I wanted to explore the game from a different angle: build a detailed reinforcement learning environment and teach Mario to play levels through training.

This repository is the result. It can connect one or more BizHawk emulator instances to a Python server, or use the headless RetroJet backend for faster rollout collection. In both cases it reads the current game state from WRAM, converts it into normalized observations, selects controller inputs with a PPO policy, and tracks training progress in a live dashboard.

## Preview

<p align="center">
  <img src="docs/images/example-run.gif" alt="KoopaPilot training run" width="568">
</p>

| Emulator View | Dashboard |
| --- | --- |
| <img src="docs/images/emulator-view.png" alt="BizHawk emulator view with KoopaPilot overlays"> | <img src="docs/images/dashboard-example.png" alt="KoopaPilot training dashboard"> |

## Features

- Parallel training with configurable BizHawk instances
- TCP communication between BizHawk Lua scripts and the Python server
- Headless RetroJet backend for faster libretro/Snes9x rollout collection
- Savestate-based resets and optional level warping
- PPO training with checkpointing, frame stacking, and resume support
- Tile-grid, sprite, movement, and player-state observations
- Reward shaping for progress, goals, coins, powerups, pipes, doors, enemies, deaths, and inactivity
- Evaluation mode with visible overlays and MP4 recording
- Demo mode for watching a saved model play in visible BizHawk at 100% speed
- Live-demo mode for RetroJet training with one separate visible BizHawk viewer
- Human-play mode for checking reward behavior manually
- Flask dashboard for live metrics and run comparisons

## Architecture

```text
+-----------------------------+        TCP sockets        +-----------------------------+
| Python PPO training server  | <-----------------------> | BizHawk emulator + Lua      |
|                             |                           | smw_agent.lua               |
| - observations + rewards    |                           | - WRAM reads / overlays     |
| - metrics + Flask dashboard |                           | - controller input          |
+-----------------------------+                           +-----------------------------+
              |
              | optional headless backend
              v
+-----------------------------+
| RetroJet libretro runner    |
| - Snes9x cores              |
| - direct WRAM read/write    |
| - native batched stepping   |
+-----------------------------+
```

## Requirements

- Windows 10 or Windows 11
- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [BizHawk](https://tasvideos.org/BizHawk) with a SNES-compatible core
- Your own legally obtained _Super Mario World_ ROM
- Optional for RetroJet: Rust, Cargo, and the separate `RetroJet` repository

BizHawk binaries and ROM files are intentionally not included in this repository.

## Setup

1. Clone the repository and enter the project directory.

   ```powershell
   git clone https://github.com/Xamexer/KoopaPilot.git
   cd KoopaPilot
   ```

2. Install the Python environment and locked dependencies.

   ```powershell
   uv sync --locked
   ```

3. Download a current BizHawk release from the [official release page](https://github.com/TASEmulators/BizHawk/releases) and extract it locally. The default configuration expects:

   ```text
   ./BizHawk/EmuHawk.exe
   ```

4. Place your own ROM at:

   ```text
   ./roms/Super Mario World.sfc
   ```

5. Create one or more BizHawk savestates and place the `.State` files in:

   ```text
   ./savestates/
   ```

6. Review `config.json`, especially the paths, emulator count, ports, savestate settings, reward values, and PPO hyperparameters.

`uv` creates and manages the local `.venv` automatically. Dependencies are declared in `pyproject.toml`; exact resolved versions live in `uv.lock`.

## RetroJet Setup

RetroJet is a separate private repository that should live next to KoopaPilot:

```text
D:\Hobby\Programmieren\
|-- KoopaPilot\
`-- RetroJet\
```

Build RetroJet:

```powershell
cd D:\Hobby\Programmieren\RetroJet
uv sync --extra dev
.\scripts\download_cores.ps1
$env:CONDA_PREFIX=$null
uv run maturin develop --release
```

Make sure the ROM exists at:

```text
D:\Hobby\Programmieren\RetroJet\roms\Super Mario World.sfc
```

Install RetroJet into KoopaPilot's environment:

```powershell
cd D:\Hobby\Programmieren\KoopaPilot
$env:CONDA_PREFIX=$null
uv pip install -e ..\RetroJet
```

Quick RetroJet benchmark:

```powershell
cd D:\Hobby\Programmieren\RetroJet
uv run retrojet-benchmark --core ./cores/snes9x2010_libretro.dll --rom "./roms/Super Mario World.sfc" --envs 32 --frames 1200 --frame-skip 4 --level 0x105
```

On the current local machine, `snes9x2010` with 32 envs and `frame_skip=4`
measured roughly `4,000` RetroJet env steps/s on level `0x105`.

## Typical RetroJet Workflow

1. Build RetroJet and install it into KoopaPilot's `.venv` as shown above.
2. Configure `backend.type` as `retrojet`, or pass `--backend retrojet`.
3. Start fast headless training:

   ```powershell
   uv run koopapilot --mode training --backend retrojet
   ```

4. To watch training progress live, use the viewer mode instead:

   ```powershell
   uv run koopapilot --mode live-demo
   ```

5. To inspect a saved checkpoint without training:

   ```powershell
   uv run koopapilot --mode demo --model ./models/model_best.zip
   ```

## Creating Savestates

Savestates make episode resets reliable and allow training on selected levels.

1. Open the ROM in BizHawk.
2. Navigate to the desired starting position.
3. Save a named state through BizHawk.
4. Copy the resulting `.State` file into `./savestates/`.
5. Repeat this for each training start you want to sample.

When `level_loading.savestate_files` is empty, the server automatically scans the top level of `./savestates/`.

For full ROM-backed resets instead of savestates, use quoted hexadecimal
Lunar Magic level IDs:

```json
"level_loading": {
  "mode": "level_loading",
  "levels": ["0x105", "0x106"],
  "savestate_files": []
}
```

This starts SMW's real overworld-to-level game-mode sequence, so level
headers, Map16, graphics, music, and sprites are loaded again after every
episode reset or registered death. Select only first rooms that can be
entered from the overworld (`0x001`-`0x024` or `0x101`-`0x1DB`), not
sublevels that are reachable only through doors or pipes.

## Commands

Start a fresh training run:

```powershell
uv run koopapilot --mode training
```

Show the colored tile grid and sprite markers during training:

```powershell
uv run koopapilot --mode training --vis
```

Resume training from a checkpoint:

```powershell
uv run koopapilot --mode training --model ./models/model_best.zip
```

Checkpoints require the same observation-space, action-space, and PPO rollout
settings that were used when they were created.

Evaluate a trained model with visible overlays and video recording:

```powershell
uv run koopapilot --mode evaluation --model ./models/model_best.zip
```

Evaluate a fixed number of episodes:

```powershell
uv run koopapilot --mode evaluation --model ./models/model_best.zip --episodes 5
```

Watch a trained model play normally in BizHawk at 100% speed:

```powershell
uv run koopapilot --mode demo --model ./models/model_best.zip
```

Run a longer demo with multiple visible BizHawk instances:

```powershell
uv run koopapilot --mode demo --model ./models/model_best.zip --demo-emulators 2 --episodes 10
```

Demo mode does not train. It loads the selected PPO checkpoint, applies the
same observation, reward, reset, and controller pipeline as training, and lets
the agent play visibly in BizHawk. If `--model` is omitted, KoopaPilot uses
`./models/model_best.zip`. If `--episodes` is omitted, the demo runs until
Ctrl+C.

Play manually while inspecting reward events:

```powershell
uv run koopapilot --mode human
```

Run only the metrics dashboard:

```powershell
uv run koopapilot --mode dashboard
```

Connect to emulator instances that are already running:

```powershell
uv run koopapilot --mode training --no-launch
```

Run training through the experimental RetroJet headless backend:

```powershell
uv run koopapilot --mode training --backend retrojet
```

Train through RetroJet while watching one live BizHawk demo instance:

```powershell
uv run koopapilot --mode live-demo
```

`live-demo` forces the RetroJet backend for training and starts one extra
BizHawk instance at 100% speed. That viewer waits for `./models/model_best.zip`
by default, then reloads it whenever the file changes, so you can see whether
the current best model is improving without slowing the headless rollout
collection. The viewer uses a separate socket range at `emulator.base_port +
1000`; override it with `--live-demo-port` if needed.

RetroJet is a separate native libretro/Snes9x batch runner intended for
high-throughput headless rollout collection. It uses libretro-compatible
savestates, not BizHawk `.State` files. BizHawk remains the recommended backend
for visual debugging, overlays, human reward inspection, and evaluation.
When `level_loading.mode` is `level_loading`, RetroJet can use the configured
Lunar Magic level IDs through its native ROM-backed level-load path.

For faster RetroJet training, add this section to `config.json`:

```json
"backend": {
  "type": "retrojet",
  "retrojet": {
    "core_path": "../RetroJet/cores/snes9x2010_libretro.dll",
    "rom_path": "../RetroJet/roms/Super Mario World.sfc",
    "num_envs": 32,
    "frame_skip": 4,
    "boot_frames": 300,
    "level_ids": ["0x105", "0x106"],
    "savestate_paths": []
  }
}
```

With `backend.type` set to `retrojet`, this is enough:

```powershell
uv run koopapilot --mode training
```

Force BizHawk instead:

```powershell
uv run koopapilot --mode training --backend bizhawk
```

Use another configuration file:

```powershell
uv run koopapilot --mode training --config ./my_config.json
```

Show all CLI options:

```powershell
uv run koopapilot --help
```

The dashboard is available at [http://127.0.0.1:8080](http://127.0.0.1:8080) by default.

## Configuration

All runtime settings live in `config.json`.

| Section           | Purpose                                                                        |
| ----------------- | ------------------------------------------------------------------------------ |
| `paths`           | Local BizHawk, ROM, Lua script, savestate, model, log, and video paths         |
| `emulator`        | Instance count, socket ports, speed, frame skip, sound, and window layout      |
| `backend`         | Optional backend selection: `bizhawk` or `retrojet`                            |
| `flags`           | Tile, reward, and controller overlay visibility                                |
| `level_loading`   | Savestate scanning, explicit state files, or Lunar Magic level IDs for warping |
| `ram_addresses`   | WRAM addresses used as a readable reference for the integration                |
| `tile_categories` | Numeric labels used for Map16 tile classes                                     |
| `normalization`   | Screen dimensions, tile-grid size, and normalization constants                 |
| `rewards`         | Reward weights, penalties, teleport threshold, and inactivity handling         |
| `ppo`             | PPO hyperparameters, frame stack, checkpoints, and episode limits              |
| `dashboard`       | Dashboard bind host, port, and refresh interval                                |

Important variables:

| Variable                       | Default   | Notes                                                     |
| ------------------------------ | --------- | --------------------------------------------------------- |
| `emulator.num_instances`       | `8`       | Number of parallel BizHawk processes                      |
| `backend.type`                 | unset     | Use `retrojet` for headless libretro training             |
| `backend.retrojet.num_envs`    | fallback  | Number of headless RetroJet environments                  |
| `backend.retrojet.core_path`   | auto      | Prefer `../RetroJet/cores/snes9x2010_libretro.dll`        |
| `backend.retrojet.frame_skip`  | fallback  | Frames repeated for each RetroJet action                  |
| `emulator.base_port`           | `9000`    | First TCP port; following instances use consecutive ports |
| `--live-demo-port`             | `10000`   | First TCP port for the optional live-demo BizHawk viewer  |
| `emulator.speed_percent`       | `6400`    | BizHawk speed during training                             |
| demo speed                     | `100`     | Demo and live-demo BizHawk instances always run normally  |
| `emulator.frame_skip`          | `4`       | Frames repeated for each selected action                  |
| `normalization.grid_size`      | `21`      | Odd-sized tile grid centered around Mario                 |
| `normalization.max_sprite_hitbox_dimension` | `128` | Scale reserved for normalized sprite footprint dimensions |
| `ppo.frame_stack`              | `4`       | Consecutive observation frames exposed to PPO             |
| `ppo.n_steps`                  | `128`     | Steps collected per emulator before each PPO update       |
| `ppo.batch_size`               | `256`     | Minibatch size; four minibatches per update with 8 emulators |
| `ppo.gamma`                    | `0.99`    | Reward discount factor                                    |
| `ppo.gae_lambda`               | `0.95`    | Bias-variance tradeoff for advantage estimation           |
| `ppo.clip_range`               | `0.1`     | Linearly decaying PPO policy-update clip range             |
| `ppo.ent_coef`                 | `0.01`    | Entropy bonus coefficient for exploration                 |
| `ppo.target_kl`                | `0.03`    | Safety stop for unusually large PPO updates               |
| `ppo.total_timesteps`          | `2500000` | Total training budget                                     |
| `ppo.save_interval_steps`      | `100000`  | Checkpoint interval                                       |
| `ppo.max_episode_steps`        | `1024`    | Hard episode limit                                        |
| `ppo.stagnation_timeout_steps` | `300`     | Stop episodes that make no progress                       |

The Lua integration contains the active Map16 classification logic. If tile categories are extended or remapped, keep `config.json` and `lua/smw_agent.lua` aligned.

## Observation Space

With the default `21 x 21` tile grid, one observation contains `562` values before frame stacking:

| Component             | Size | Description                                                        |
| --------------------- | ---: | ------------------------------------------------------------------ |
| Mario screen position |    2 | Normalized X/Y coordinates                                         |
| Mario velocity        |    2 | Signed normalized X/Y speed                                        |
| Powerup state         |    4 | One-hot small, big, cape, or fire state                            |
| Movement state        |    4 | One-hot ground, air, water, or climbing state                      |
| Level orientation     |    1 | Horizontal or vertical level flag                                  |
| Tile grid             |  441 | `21 x 21` normalized Map16 categories                              |
| Sprites               |  108 | 12 slots with active state, ID, position, native or platform-specific interaction footprint, velocity, and misc state |

With the default four-frame stack, PPO receives `2248` values per environment step.

## Action Space

The policy selects one of 12 discrete controller combinations. The action table covers idle, left/right running, jumps, spin jumps, ducking, door or climbing input, and releasing `Y`. Run is held by default so that movement stays responsive at training speed.

## Reward Design

The reward calculator favors new level progress and successful exits while discouraging deaths, damage, inactivity, and endless episodes.

| Event                                    |    Default reward |
| ---------------------------------------- | ----------------: |
| New horizontal progress                  |  `+0.3` per pixel |
| New vertical progress in vertical levels |  `+0.3` per pixel |
| Newly explored tile cell                 |         `+0.25` |
| Goal reached                             |           `+1000` |
| Coin collected                           |              `+0` |
| Powerup upgrade                          |             `+20` |
| 1-UP collected                           |              `+0` |
| Pipe or door transition                  |             `+50` |
| Enemy defeated                           |             `+10` |
| Enemy stunned                            |             `+10` |
| Death                                    |             `-30` |
| Powerup loss                             |             `-10` |
| Time penalty                             |  `-0.05` per step |

Horizontal reward is granted only for new per-episode maximum X positions.
Returning to previously visited ground therefore cannot farm reward. Large
coordinate jumps are treated as teleports to prevent savestate loads and
transitions from creating false progress rewards.

Each coarse tile cell also grants a small episode-local exploration reward
the first time Mario visits it. This makes necessary detours and climbs
learnable without allowing repeated movement between known cells to farm
reward. Newly explored cells reset the stagnation timeout as well.

## Training Profile

The default PPO profile uses short rollouts: `128` steps across each of the
eight emulator instances, producing `1024` transitions per update. With a
minibatch size of `256`, PPO trains on four minibatches for each of four
epochs. This keeps policy updates frequent while preserving a useful amount
of parallel experience.

Timeouts and maximum-step limits are treated as truncated episodes. Deaths
and goals remain real terminal states. This distinction allows the value
function to bootstrap correctly when an episode ends only because of a
configured time limit.

## Dashboard

Training runs write JSON metrics below `./logs/`. The Flask dashboard can:

- display timesteps, episodes, rewards, and horizontal progress;
- plot reward, episode length, and episode maximum X over time;
- load the latest run automatically;
- inspect a previous run;
- compare reward graphs across multiple runs.

## Project Structure

```text
.
|-- config.json
|-- pyproject.toml
|-- uv.lock
|-- lua/
|   `-- smw_agent.lua
|-- server/
|   |-- main.py
|   |-- demo.py
|   |-- environment.py
|   |-- vec_env.py
|   |-- socket_server.py
|   |-- retrojet_backend.py
|   |-- observation.py
|   |-- reward.py
|   |-- training.py
|   |-- evaluation.py
|   |-- human_play.py
|   |-- emulator_manager.py
|   |-- metrics.py
|   `-- dashboard/
|-- tests/
|-- docs/images/
|-- models/       # local checkpoints, ignored by Git
|-- logs/         # local training metrics, ignored by Git
|-- roms/         # local ROM files, ignored by Git
|-- savestates/   # local BizHawk states, ignored by Git
`-- videos/       # local evaluation recordings, ignored by Git
```

## RAM Map References

The WRAM addresses used by this project were researched with the SMWCentral documentation:

- [SMWCentral SMW RAM Memory Map](https://www.smwcentral.net/?p=memorymap&game=smw&region=ram)

Useful related references:

- [BizHawk project page](https://tasvideos.org/BizHawk)
- [BizHawk releases](https://github.com/TASEmulators/BizHawk/releases)
- [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

## Tests

Run the lightweight unit suite:

```powershell
uv run python -m unittest discover -s tests -v
```

Run a syntax check:

```powershell
uv run python -m compileall -q server tests
```

Full end-to-end training and evaluation checks require a local BizHawk installation, a ROM, and savestates.

## Legal Notice

This is an independent research and hobby project. It is not affiliated with or endorsed by Nintendo. No ROM, commercial game data, BizHawk binaries, savestates, or trained checkpoints are distributed through this repository.

## License

Released under the [MIT License](LICENSE).
