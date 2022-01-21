#!/usr/bin/env python3

import byoc
import autoprop

from relax_b import init_logging, logger, kv
from byoc import Key, DocoptConfig
from pathlib import Path

@autoprop.cache
class PickBestModel(byoc.App):
    """\
Create a symlink to the best-scoring relaxed model.

Usage:
    pick_best_model.py <pdbs> [-o <pdb>] [-f]

Arguments:
    <pdbs>
        A directory containing the PDB files to pick from.  Only files with the 
        '*.pdb' suffix will be considered.

Options:
    -o --output <pdb>   [default: best_model.pdb]
        The name of the symlink to create.

    -f --force
        Overwrite the output symlink if it exists already.  Without this 
        option, the program will refuse to overwrite existing results.
"""
    __config__ = [
            DocoptConfig,
    ]

    input_dir = byoc.param(
            Key(DocoptConfig, '<pdbs>'),
            cast=Path,
    )
    output_path = byoc.param(
            Key(DocoptConfig, '--output'),
            cast=Path,
    )
    overwrite = byoc.param(
            Key(DocoptConfig, '--force'),
            default=False,
    )

    def main(self):
        self.load(DocoptConfig)
        init_logging()

        best_model = pick_best_relaxed_model(self.pdb_paths)
        record_best_relaxed_model(best_model, self.output_path, self.overwrite)

    def get_pdb_paths(self):
        pdb_paths = list(self.input_dir.glob('*.pdb'))
        if not pdb_paths:
            logger.error("no *.pdb files in '{self.input_dir}'")
        return pdb_paths

def pick_best_relaxed_model(pdb_paths):
    scores = {
            Path(x): score_from_pdb(x)
            for x in pdb_paths
    }
    models = sorted(scores.keys(), key=lambda x: scores[x])
    best_model = models[0]

    logger.info(f"{len(models)} relaxed models.")
    logger.info(f"sorted scores:")
    for model in models:
        logger.info(f"{model.name:>10s}: {scores[model]:.3f}")

    return best_model

def score_from_pdb(pdb_path):
    with open(pdb_path) as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('pose'):
            return float(line.split()[-1])

def record_best_relaxed_model(src_path, dest_path, overwrite=False):
    if dest_path.exists():
        if overwrite:
            dest_path.unlink()
        else:
            logger.fatal(f"output path already exists: {dest_path}")
            logger.fatal(f"use the '-f' flag to overwrite")
            return False

    logger.info(f"symlinking '{src_path}' to '{dest_path}'.")
    dest_path.symlink_to(src_path)
    return True


if __name__ == '__main__':
    PickBestModel.entry_point()
