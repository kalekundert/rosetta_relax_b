#!/usr/bin/env python3

import pyrosetta
import numpy as np
import pandas as pd
import logging
import byoc
import autoprop

from pyrosetta.io import pose_from_pdb
from pyrosetta.rosetta import core, protocols, basic
from byoc import Key, Value, DocoptConfig, JsonConfig
from pathlib import Path
from math import pi

logger = logging.getLogger('rosetta_relax_b')

@autoprop.cache
class Relax(byoc.App):
    """\
Relax a protein structure.

Usage:
    relax_b.py [<pdb>] [-c <cst>] [-i <json>] [-o <pdb>] [-fd]

Arguments:
    <pdb>
        The PDB model to relax.  Not required if `--input` is specified.

Options:
    -c --cst <stdev>
        The weight of the all-atom starting-position constraint to apply during 
        the relax.  This weight is more precisely the standard deviation of the 
        harmonic restraint applied to each atom.  This option is not required 
        if `--input` is specified.
    
    -i --input <json>
        The path to the JSON file produced by the `optimize_cst.py` script.  
        This file specifies an unrelaxed PDB file and an optimized constraint 
        weight, so this option conveys the same information as `<pdb>` and 
        `--cst`.

    -o --output <pdb>
        The path where the relaxed PDB model should be written.

    -f --force
        Overwrite the output PDB file if it exists already.  Without this 
        option, the program will refuse to overwrite existing results.

    -d --dry-run
        Run an extremely abbreviated protocol for the purpose of debugging the 
        pipeline.  The results from such runs should not be used for any 
        scientific purpose.  This option implies '--force' and is enabled 
        by default when using `--input` from an optimization dry run.
"""
    __config__ = [
            JsonConfig.setup(
                path_getter=lambda self: self.optimization_json,
                ),
            DocoptConfig,
            ]

    input_pdb_path = byoc.param(
            Key(DocoptConfig, '<pdb>'),
            Key(JsonConfig, 'unrelaxed_pdb'),
    )
    output_pdb_path = byoc.param(
            Key(DocoptConfig, '--output'),
            Value('relaxed.pdb'),
            cast=Path,
    )
    optimization_json = byoc.param(
            Key(DocoptConfig, '--input'),
            cast=Path,
    )
    cst_stdev = byoc.param(
            Key(DocoptConfig, '--cst'),
            Key(JsonConfig, 'cst_stdev'),
            cast=float,
    )
    overwrite = byoc.param(
            Key(DocoptConfig, '--force'),
            Key(DocoptConfig, '--dry-run'),
            default=False,
    )
    dry_run = byoc.param(
            Key(DocoptConfig, '--dry-run'),
            Key(JsonConfig, 'dry_run'),
            default=False,
    )

    def main(self):
        self.load(DocoptConfig)
        self.reload(JsonConfig)

        init_logging()
        init_rosetta()

        self.relax()

    def relax(self):
        if self.output_pdb_path.exists() and not self.overwrite:
            logger.fatal(f"output file already exists: {self.output_pdb_path}")
            logger.fatal(f"use the '-f' flag to overwrite")
            return

        logger.info("begin relax simulation:")
        logger.info(kv("restraint weight", self.cst_stdev))
        logger.info(kv("dry run", self.dry_run))

        initial_pose = self.unrelaxed_pose
        relaxed_pose = core.pose.Pose(initial_pose)

        sfxn = relax_pose(
                relaxed_pose,
                cst_stdev=self.cst_stdev,
                dry_run=self.dry_run,
        )

        df = calc_atoms_within_b(initial_pose, relaxed_pose)
        initial_score = sfxn(initial_pose)
        relaxed_score = sfxn(relaxed_pose)
        score_diff = relaxed_score - initial_score

        logger.info("end relax simulation:")
        logger.info(kv("mean distance (actual)", df.dist_actual.mean()))
        logger.info(kv("mean distance (B-factor)", df.dist_b_factor.mean()))
        logger.info(kv("score (REU)", relaxed_score))
        logger.info(kv("score improvement (REU)", score_diff))

        record_pdb(relaxed_pose, self.output_pdb_path)

    def get_unrelaxed_pose(self):
        return load_pdb(self.input_pdb_path)

def init_rosetta(constant_seed=False):
    flags = []

    if constant_seed:
        flags += ['-constant_seed']

    pyrosetta.init(' '.join(flags), set_logging_handler='logging')

def relax_pose(pose, cst_stdev=1.0, dry_run=False):
    # Some options can only be set via the command-line.  Welcome to Rosetta...
    basic.options.set_real_option('relax:coord_cst_stdev', cst_stdev)
    basic.options.set_boolean_option('run:test_cycles', dry_run)

    sfxn = load_score_function()
    tf = load_task_factory(extra_rotamers=not dry_run)
    mm = load_move_map()
    relax = load_fast_relax(sfxn, tf, mm)

    relax.apply(pose)

    return sfxn

def load_score_function():
    return core.scoring.ScoreFunctionFactory.create_score_function('ref2015')

def load_task_factory(extra_rotamers=True):
    tf = core.pack.task.TaskFactory()

    # FastRelax automatically adds the `IncludeCurrent` task operation, so 
    # adding it here is redundant, but hopefully helpful in terms of clarity.
    tf.push_back(core.pack.task.operation.IncludeCurrent())
    tf.push_back(core.pack.task.operation.RestrictToRepacking())

    if extra_rotamers:
        ex = core.pack.task.operation.ExtraRotamersGeneric()
        ex.ex1(True); ex.ex2(True)
        tf.push_back(ex)

    return tf

def load_move_map():
    mm = core.kinematics.MoveMap()
    mm.set_bb(True)
    mm.set_chi(True)

    return mm

def load_fast_relax(sfxn, tf, mm):
    relax = protocols.relax.FastRelax(sfxn)
    relax.set_task_factory(tf)
    relax.set_movemap(mm)

    # Both of the following must be set, or no restraints will be applied.
    relax.constrain_coords(True)
    relax.constrain_relax_to_start_coords(True)
    relax.ramp_down_constraints(False)

    return relax

def load_pdb(path):
    return pose_from_pdb(str(path))

def record_pdb(pose, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pose.dump_pdb(str(path))

def calc_atoms_within_b(initial_pose, relaxed_pose):
    rows = []

    for i in range(1, initial_pose.size() + 1):
        residue = initial_pose.residue(i)

        for j in range(1, residue.natoms() + 1):
            id = core.id.AtomID(j, i)
            v1 = initial_pose.xyz(id)
            v2 = relaxed_pose.xyz(id)
            d = v1.distance(v2)

            rows.append(dict(
                residue=i,
                atom=j,
                distance = v1.distance(v2),
                b_factor = initial_pose.pdb_info().bfactor(i, j),
                is_backbone = j <= residue.last_backbone_atom(),
                is_dna = residue.is_DNA(),
            ))

    df = pd.DataFrame(rows)
    df = df[df.b_factor > 0]
    df = df[df.is_backbone | df.is_dna]
    df['dist_actual'] = df.distance
    df['dist_b_factor'] = np.sqrt(df.b_factor / (8*pi**2))
    df['dist_diff'] = df.dist_actual - df.dist_b_factor
    return df


def init_logging():
    # The reason for using the logging module is to get uniform output 
    # formatting by taking advantage of the fact that pyrosetta can redirect 
    # all of rosetta's output to a logger.
    log = logging.getLogger()
    log.setLevel('WARNING')

    logger.setLevel('DEBUG')
    logging.getLogger('rosetta').setLevel('DEBUG')

    formatter = logging.Formatter('{asctime}\t{name}\t{levelname}\t{message}', style='{')
    handler = logging.StreamHandler()

    for handler in log.handlers[:]:
        log.removeHandler(handler)

    handler.setFormatter(formatter)
    log.addHandler(handler)

def kv(key, value):
    """Utility for logging key/value pairs in a consistent format."""
    return f"- {key+':':<35} {value}"


if __name__ == '__main__':
    Relax.entry_point()
