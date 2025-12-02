from datetime import datetime
import hydra
import json
import numpy as np
from omegaconf import OmegaConf, DictConfig
import os
import pyspiel
import random
import time
import torch
import traceback
import uuid

from algorithms.eas_exploitability import compute_exploitability, build_traverser
from algorithms.runner import get_runner_cls
from utils import log_to_csv, get_metadata, log_memory_usage_periodically


OPENSPIEL_GAMES = {
    "classical_phantom_ttt": "phantom_ttt(obstype=reveal-nothing)",
    "abrupt_phantom_ttt": "phantom_ttt(obstype=reveal-nothing,gameversion=abrupt)",
    "classical_dark_hex": "dark_hex(gameversion=cdh,board_size=3,obstype=reveal-nothing)",
    "abrupt_dark_hex": "dark_hex(gameversion=adh,board_size=3,obstype=reveal-nothing)",
    "kuhn_poker": "kuhn_poker(players=2)",
    "leduc_poker": "leduc_poker(players=2)",
}


def set_seed(seed):
    # set the random seed for torch, numpy, and python
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def compute_exploitability_wrapper(
    traverser,
    log_file,
    game_name,
    model_p0,
    model_p1,
    step,
    action_selection=["sto", "sto"],
):
    print("Computing exploitability... (this can take a few minutes)")
    t0_exploitability = time.time()
    ev0, expl0, expl1 = compute_exploitability(
        model_p0,
        model_p1,
        traverser=traverser,
        batch_size=400_000,
        action_selection=action_selection,
        game_name=game_name,
    )
    log_data = {
        "global_step": step,
        "avg_score_response": (expl0 + expl1) / 2,
        "avg_score_p0": ev0 + expl1,
        "avg_score_p1": -ev0 + expl0,
        "ev0": ev0,
        "expl0": expl0,
        "expl1": expl1,
        "timestamp": time.time(),
        "computation_duration": time.time() - t0_exploitability,
    }
    print(f'avg_score_response={(expl0 + expl1) / 2}')
    log_to_csv(log_data, log_file)


@hydra.main(version_base=None, config_path="configs", config_name="experiment")
def main(cfg: DictConfig):
    # checks
    assert cfg.game in list(OPENSPIEL_GAMES.keys())
    assert cfg.compute_exploitability in [True, False]
    print(f"Seed: {cfg.seed}")
    # seed
    set_seed(cfg.seed)

    # setup logging
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    random_str = uuid.uuid4().hex[:6]
    experiment_dir = os.path.join(
        cfg.save_dir, cfg.group_name, cfg.algorithm.algorithm_name, cfg.game, f'{time_str}_{random_str}'
    )
    os.makedirs(experiment_dir, exist_ok=False)
    cfg.experiment_dir = experiment_dir
    print(f"Logging at {cfg.experiment_dir}")

    # save config
    config_path = os.path.join(cfg.experiment_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)

    # save metadata
    metadata = get_metadata()
    metadata['status'] = 'Running'
    metadata['start_date'] = datetime.now().strftime("%B %d, %Y at %-I:%M:%S %p")
    metadata['start_timestamp'] = time.time()
    metadata_path = os.path.join(cfg.experiment_dir, "metadata.json")
    with open(metadata_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    # setup exploitability computation
    if cfg.compute_exploitability:
        t0_traverser = time.time()
        print("Building traverser... (this can take a few minutes)")
        traverser = build_traverser(cfg.game)
        print("Done")
        metadata['eas_traverser_build_time'] = time.time() - t0_traverser
        with open(metadata_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=4)
        exploitability_log_file = os.path.join(cfg.experiment_dir, "exploitability.csv")

        def compute_exploitability_callback(*args, **kwargs):
            compute_exploitability_wrapper(
                traverser, exploitability_log_file, cfg.game, *args, **kwargs
            )
    else:
        compute_exploitability_callback = None

    # load game
    print("Loading game")
    openspiel_game = OPENSPIEL_GAMES[cfg.game]
    game = pyspiel.load_game(openspiel_game)

    # load algorithm
    print("Loading runner")
    algorithm = cfg.algorithm.algorithm_name
    runner = get_runner_cls(algorithm)(cfg, game, compute_exploitability_callback)
    
    

    # start memory logging thread
    memory_usage_file_path = os.path.join(cfg.experiment_dir, "memory_usage.log")
    thread, stop_event = log_memory_usage_periodically(log_file_path=memory_usage_file_path, log_interval_s=10, runner=runner)

    # train
    print(
        f'Training {cfg.algorithm.algorithm_name.upper()} on {cfg.game} for {cfg.max_steps} steps '
        f'{"with" if cfg.compute_exploitability else "without"} exploitability computation'
    )

    if cfg.run_in_safe_mode:
        print("Running in safe mode")
        try:
            runner.run()
        # update metadata
        except KeyboardInterrupt:
            metadata['status'] = 'Stopped'
            print('Training stopped')
        except Exception as e:
            metadata['status'] = 'Crashed'
            metadata['exception'] = repr(e)
            print('Training crashed')
            tb = traceback.format_exc()
            print(tb)
            err_path = os.path.join(cfg.experiment_dir, "error.txt")
            with open(err_path, 'w') as f:
                f.write(tb)
            metadata['error'] = tb
        else:
            metadata['status'] = 'Finished'
            print('Training finished!')
    else:
        runner.run()
        metadata['status'] = 'Finished'
        print('Training finished!')
    metadata['end_date'] = datetime.now().strftime("%B %d, %Y at %-I:%M:%S %p")
    metadata['end_timestamp'] = time.time()
    with open(metadata_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    # stop memory loggging thread peacefully
    stop_event.set()
    thread.join()


if __name__ == "__main__":
    main()
