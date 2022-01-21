#!/usr/bin/env python3

import pyrosetta
import numpy as np
import matplotlib.pyplot as plt
import byoc
import autoprop
import json

from relax_b import (
        init_rosetta, relax_pose, load_pdb, record_pdb, calc_atoms_within_b,
        init_logging, logger, kv,
)
from pyrosetta.rosetta import core
from skopt import gp_minimize, expected_minimum
from skopt.callbacks import DeadlineStopper
from skopt.acquisition import _gaussian_acquisition
from byoc import Key, Method, Value, DocoptConfig, NtConfig
from more_itertools import one
from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path
from color_scheme import *

@autoprop.cache
class OptimizeCst(byoc.App):
    """\
Determine how strong the all-atom coordinates need to be...

Usage:
    optimize_cst.py <pdb> [-c <config>] [-i <prior>]... [-o <dir>] [-fd]

Arguments:
    <pdb>
        The PDB model to relax.

Options:
    -c --config <path>
        Provide a file containing configuration options not specified on the 
        command line.  See below for a description of the options that can be 
        specified in this file.

    -i --input <path>
        Provide the output from a previous run of this script as input to this 
        run.  More specifically, any (constraint stdev, objective function) 
        pairs that were measured in the previous run will be copied into this 
        one.  Note that these pairs will not be included in the output for this 
        run.  This option can be specified multiple times to combine 
        information from several previous runs.

    -o --output <dir>                   [default: out]
        The directory where all output files will be written.

    -f --force
        Overwrite the output directory if it exists already.  Without this 
        option, the program will refuse to overwrite existing results.

    -d --dry-run
        Run an extremely abbreviated protocol for the purpose of debugging the 
        pipeline.  The results from such runs should not be used for any 
        scientific purpose.  This option implies '--force'.

Config File:
    The following options can be specified in the file provided by the 
    `--config` option.  This file should be in the Nested Text format.

    target_diff: <float>                [default: ${app.target_diff}]
        The desired distance between the distance an atom actually moves, and 
        the distance implied by the B-factor for that atom.  A value of 0 
        indicates that atoms should move as much as their B-factors allow.

    target_percentile: <float>          [default: ${app.target_percentile}]
        The `target_diff` (see above) is calculated for every atom in the 
        model.  This option specifies which percentile of this distribution to 
        use when calculating the objective function.  In other words, a value 
        of 90 would indicate that exactly 90% of atoms would have to have 
        distances within the target in order for a constraint stdev to be 
        considered optimal.
        
    cst_limits: <float, float>          [default: ${app.cst_limits}]
        The minimum and maximum (respectively) constraint stdevs that will be 
        considered.

    n_max_calls: <int>                  [default: ${app.n_max_calls}]
        The maximum number of relax simulations to invoke.  Ideally this number 
        would never be reached, and the optimization would end when the 
        convergence condition is reached.

    n_initial_points: <int>             [default: ${app.n_initial_points}]
        The number of initial points to evaluate before letting the 
        optimization algorithm pick which points to evaluate.

    initial_point_generator: <str>      [default: ${app.initial_point_generator}]
        The algorithm to use when generating initial points.  For the list of 
        algorithms, see: https://tinyurl.com/pbrdhp8b

    convergence_cst_delta: <float>      [default: ${app.convergence_cst_delta}]
        Consider the optimization complete once the predicted best constraint 
        stdev has not changed by more than this amount in the number of 
        iterations indicated by the following option.

    convergence_n_most_recent: <int>    [default: ${app.convergence_n_most_recent}]
        How many of the most recent iterations to consider when deciding is 
        the convergence criterion (see above) has been satisfied.

    time_budget_s: <float>
        How much time to allow the optimization simulation, in seconds.  The 
        optimizer will keep track of how long each evaluation takes, and 
        attempt to stop before this time is exceeded.  By default, there is no 
        time limit.
"""
    __config__ = [
            DocoptConfig,
            NtConfig.setup(path_getter=lambda self: self.config_path)
            ]

    unrelaxed_pdb_path = byoc.param(
            Key(DocoptConfig, '<pdb>'),
    )
    config_path = byoc.param(
            Key(DocoptConfig, '--config'),
            default_factory=list
    )
    prior_results_paths = byoc.param(
            Key(DocoptConfig, '--input'),
            default_factory=list,
    )
    output_dir = byoc.param(
            Key(DocoptConfig, '--output'),
            cast=Path,
    )
    target_diff = byoc.param(
            Key(NtConfig, 'target_diff', cast=float),
            default=0,
    )
    target_percentile = byoc.param(
            Key(NtConfig, 'target_percentile', cast=float),
            default=90,
    )
    cst_limits = byoc.param(
            Key(NtConfig, 'cst_limits', cast=lambda x: tuple(map(float, x))),
            default=(0.5, 10.0),
    )
    n_max_calls = byoc.param(
            Key(NtConfig, 'n_max_calls', cast=int),
            default=100,
    )
    n_initial_points = byoc.param(
            Key(NtConfig, 'n_initial_points', cast=int),
            default=4,
    )
    initial_point_generator = byoc.param(
            Key(NtConfig, 'initial_point_generator'),
            default='sobol',
    )
    convergence_cst_delta = byoc.param(
            Key(NtConfig, 'convergence_cst_delta', cast=float),
            default=0.1,
    )
    convergence_n_most_recent = byoc.param(
            Key(NtConfig, 'convergence_n_most_recent', cast=int),
            default=10,
    )
    time_budget_s = byoc.param(
            Key(NtConfig, 'time_budget_s', cast=float),
            default=None,
    )
    overwrite = byoc.param(
            Key(DocoptConfig, '--force'),
            Key(DocoptConfig, '--dry-run'),
            default=False,
    )
    dry_run = byoc.param(
            Key(DocoptConfig, '--dry-run'),
            default=False,
    )

    def main(self):
        self.load(DocoptConfig)

        init_logging()
        init_rosetta()

        self.optimize(self.unrelaxed_pose)

    def optimize(self, initial_pose):
        if self.output_dir.exists() and not self.overwrite:
            logger.fatal(f"output directory already exists: {self.output_dir}")
            logger.fatal(f"use the '-f' flag to overwrite")
            return

        iteration = 0
        trajectory = {
                'cst_stdev': [],
                'pdb_path': [],
                'objective': [],
                'score_reu': [],
                'score_diff_reu': [],
        }

        class ExpectedMinimumStopper:

            def __init__(self, dx, n_iterations):
                self.dx = dx
                self.n_iterations = n_iterations
                self.expected_minima = []

            def __call__(self, result):
                if not result.models:
                    return None

                n = self.n_iterations
                (x,), y = expected_minimum(result)
                self.expected_minima.append(x)

                logger.info("check convergence:")
                logger.info(kv("constraint stdev guess", x))

                if len(self.expected_minima) < n:
                    logger.info(kv("converged", False))
                    return False

                most_recent = self.expected_minima[-n:]
                most_recent_dx = max(most_recent) - min(most_recent)
                is_converged = most_recent_dx < self.dx

                logger.info(kv(f"change over last {n} guesses:", most_recent_dx))
                logger.info(kv("convergence threshold", self.dx))
                logger.info(kv("converged", is_converged))

                return is_converged

            @property
            def most_recent(self):
                return self.expected_minima[-1]

        def objective(params):
            nonlocal iteration
            iteration += 1

            cst_stdev = params[0]

            logger.info("begin relax simulation:")
            logger.info(kv("optimization iteration", iteration))
            logger.info(kv("constraint stdev", cst_stdev))

            # Relax the pose with the given constraint stdev.
            relaxed_pose = core.pose.Pose(initial_pose)
            sfxn = relax_pose(
                    relaxed_pose,
                    cst_stdev=params[0],
                    dry_run=self.dry_run,
            )

            # Determine how well the relaxed structure agrees with the B-factors.
            df = calc_atoms_within_b(initial_pose, relaxed_pose)
            dist_from_target = np.percentile(
                    df.dist_diff - self.target_diff,
                    self.target_percentile,
            )
            objective = abs(dist_from_target)

            # Record/log this iteration.
            pdb_path = self.get_pdb_output_path(iteration)
            record_pdb(relaxed_pose, pdb_path)

            initial_score = sfxn(initial_pose)
            relaxed_score = sfxn(relaxed_pose)
            score_diff = relaxed_score - initial_score

            trajectory['cst_stdev'].append(cst_stdev)
            trajectory['pdb_path'].append(str(pdb_path))
            trajectory['objective'].append(objective)
            trajectory['score_reu'].append(relaxed_score)
            trajectory['score_diff_reu'].append(score_diff)

            logger.info("end relax simulation:")
            logger.info(kv(f"mean distance (actual)", df.dist_actual.mean()))
            logger.info(kv(f"mean distance (B-factor)", df.dist_b_factor.mean()))
            logger.info(kv(f"distance diff ({self.target_percentile:.0f}%)", dist_from_target))
            logger.info(kv("objective", objective))
            logger.info(kv("score (REU)", relaxed_score))
            logger.info(kv("score improvement (REU)", score_diff))

            return objective

        logger.info("begin optimization:")
        logger.info(kv("target distance diff", f'{self.target_diff}%'))
        logger.info(kv("target percentile", f'{self.target_percentile}%'))
        logger.info(kv("min constraint stdev", self.cst_limits[0]))
        logger.info(kv("max constraint stdev", self.cst_limits[1]))
        logger.info(kv("prior evaluations", len(self.prior_trajectories['objective'])))
        logger.info(kv("dry run", self.dry_run))

        expected_min = ExpectedMinimumStopper(
                self.convergence_cst_delta,
                self.convergence_n_most_recent,
        )
        callbacks = [expected_min]
        if t := self.time_budget_s:
            callbacks.append(DeadlineStopper(t))

        result = gp_minimize(
                objective,
                dimensions=[(*self.cst_limits, "uniform")],
                n_calls=self.n_max_calls \
                        if not self.dry_run else 1,
                n_initial_points=self.n_initial_points \
                        if not self.dry_run else 1,
                initial_point_generator=self.initial_point_generator,
                x0=self.prior_trajectories['cst_stdev'] or None,
                y0=self.prior_trajectories['objective'] or None,
                callback=callbacks,
        )

        x_best = expected_min.most_recent

        logger.info("end optimization:")
        logger.info(kv("best constraint stdev", x_best))
        logger.info(kv("function calls", iteration))

        record_optimization_results(self, x_best, trajectory, self.json_output_path)
        plot_optimization_results(result, self.svg_output_path)

    def get_unrelaxed_pose(self):
        return load_pdb(self.unrelaxed_pdb_path)

    def get_prior_trajectories(self):
        trajectory = {
                'cst_stdev': [],
                'objective': [],
        }
        for path in self.prior_results_paths:
            results = load_optimization_results(path)
            trajectory['cst_stdev'] += results['trajectory']['cst_stdev']
            trajectory['objective'] += results['trajectory']['objective']

        return trajectory

    def get_json_output_path(self):
        return self.output_dir / 'results.json'

    def get_pdb_output_path(self, i):
        digits = len(str(self.n_max_calls))
        return self.output_dir / 'pdb' / f'{i:0{digits}}.pdb'

    def get_svg_output_path(self):
        return self.output_dir / 'gaussian_process_model.svg'

def load_optimization_results(path):
    with open(path) as f:
        return json.load(f)

def record_optimization_results(app, cst_stdev, trajectory, path):
    results = {
            'cst_stdev': cst_stdev,
            'unrelaxed_pdb': str(app.unrelaxed_pdb_path),
            'dry_run': app.dry_run,
            'trajectory': trajectory,
    }
    with open(path, 'w') as f:
        json.dump(results, f)

def plot_optimization_results(result, path=None):
    args = result.specs.get('args', {})
    dim = one(result.space.dimensions)
    opt, _ = expected_minimum(result)

    n_calls = len(result.func_vals) 
    n_preset = len(args.get('x0') or [])
    n_random = args.get('n_initial_points', 0)
    n_guided = n_calls - n_preset - n_random
    n_models = len(result.models)
    n_unmodeled = n_calls - n_models

    assert n_models == n_guided + 1
    assert n_unmodeled == n_preset + n_random - 1

    fig, axes = plt.subplots(
        n_models, 2,
        sharex='all',
        sharey='col',
        figsize=(11, 3*n_models),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.set_xlim(dim.low, dim.high)
        opt_handle = ax.axvline(
                opt,
                color=light_grey[0],
                linestyle=':',
                label='expected minimum',
                zorder=3,
        )
    for ax in axes[-1,:]:
        ax.set_xlabel('cst stdev')

    @dataclass
    class Index:
        call: int
        model: int

    for i_model in range(n_models):
        i = Index(i_model + n_unmodeled, i_model)
        handles = [opt_handle]

        axes[i_model,0].set_ylabel(f'it={i_model+1}\nobjective func.')
        axes[i_model,1].set_ylabel('acquisition func.')
        axes[i_model,1].set_yticks([])

        handles += plot_model(
                axes[i_model,0],
                result, i,
                color=light_grey[0],
                linestyle='-',
                zorder=2,
                label="model (mean)",
                fill_fc=light_grey[2],
                fill_ec=None,
                fill_zorder=1,
                fill_label="model (95% CI)",
        )
        handles += plot_samples(
                axes[i_model,0],
                result, i,
                mec=navy[0],
                marker='+',
                linestyle='',
                label='observations',
                zorder=4,
        )

        next_acqs = []

        handles += plot_acq(
                axes[i_model,1],
                result, i,
                'LCB',
                next_acqs,
                color=red[0],
                next_marker='o',
                next_mfc='none',
                next_mec=red[0],
                label='LCB',
        )
        handles += plot_acq(
                axes[i_model,1],
                result, i,
                'EI',
                next_acqs,
                color=olive[0],
                next_marker='o',
                next_mfc='none',
                next_mec=olive[0],
                label='EI',
        )
        handles += plot_acq(
                axes[i_model,1],
                result, i,
                'PI',
                next_acqs,
                color=blue[0],
                next_marker='o',
                next_mfc='none',
                next_mec=blue[0],
                label='PI',
        )

        if next_acqs:
            next_acqs.sort(key=itemgetter(0))
            next_acqs[0][1]()

    axes[0,1].legend(
            handles=handles,
            bbox_to_anchor=(1.04, 1),
            loc="upper left", 
    )
    fig.tight_layout(
            #rect=[0, 0, 0.75, 1],
    )

    if path:
        fig.savefig(path)

    return fig

def plot_samples(ax, result, i, **kwargs):
    return ax.plot(
            result.x_iters[:i.call+1],
            result.func_vals[:i.call+1],
            **kwargs,
    )

def plot_model(ax, result, i, **kwargs):
    plot_kwargs, fill_kwargs = _split_kwargs(kwargs, 'fill')
    x, x_model = _linspace_x(result)
    model = result.models[i.model]

    y_pred, sigma = model.predict(x_model, return_std=True)
    plot_handles = ax.plot(x, y_pred, **plot_kwargs)

    # 95% confidence interval:
    # https://en.wikipedia.org/wiki/1.96
    fill_handles = ax.fill(
            np.concatenate([x, x[::-1]]),
            np.concatenate([
                y_pred - 1.9600 * sigma,
                (y_pred + 1.9600 * sigma)[::-1]
            ]),
            **fill_kwargs,
    )

    return *plot_handles, *fill_handles

def plot_acq(ax, result, i, acq_func, next_acqs, **kwargs):
    plot_kwargs, next_kwargs = _split_kwargs(kwargs, 'next')

    x, x_model = _linspace_x(result)
    res_acq_func = result.specs["args"].get("acq_func", "gp_hedge")
    acq_func_kwargs = result.specs.get("args", {}).get("acq_func_kwargs", {})

    if (res_acq_func != 'gp_hedge') and (acq_func != res_acq_func):
        return

    acq = _gaussian_acquisition(
            x_model, result.models[i.model],
            y_opt=np.min(result.func_vals[:i.call+1]),
            acq_func=acq_func,
            acq_func_kwargs=acq_func_kwargs,
    )
    next_x = x[np.argmin(acq)]
    next_acq = acq[np.argmin(acq)]
    next_i = i.call + 1

    if next_i < len(result.x_iters):
        next_acqs.append((
            next_x - result.x_iters[next_i],
            lambda: ax2.plot(next_x, -next_acq, **next_kwargs),
        ))

    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.set_ymargin(0.1)

    return ax2.plot(x, -acq, **plot_kwargs)

def _linspace_x(result):
    dim = one(result.space.dimensions)
    x = np.linspace(dim.low, dim.high, 100)
    x_model = dim.transform(x)
    return x.reshape(-1, 1), x_model.reshape(-1, 1)

def _split_kwargs(kwargs, *prefixes):
    out = [{} for i in range(len(prefixes) + 1)]

    for k, v in kwargs.items():
        for i, p in enumerate(prefixes, 1):
            if k.startswith(p):
                k = k[len(p)+1:]
                break
        else:
            i = 0

        out[i][k] = v

    return out


if __name__ == '__main__':
    OptimizeCst.entry_point()

