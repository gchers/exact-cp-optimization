"""This library only serves to launch multiple experiments at the same
time and fetch the results back.
It handles multiprocessing as appropriate, all the naming conventions
of the files, and it logs results (and metadata) into .json files.

Please refer to the notebook, exact-cp-optimization.ipynb, for a description
of how this library is used.
The library itself does not contain any detail about our proposals.
"""
import os
import json
import arrow
import inspect
import numpy as np
import pandas as pd
import multiprocessing as mp
from sh.contrib import git
from itertools import product, starmap
from flatten_dict import flatten
from collections import defaultdict
from setproctitle import setproctitle

class Eli:
    """Simple experiment library.
    """
    def __init__(self, results_dir="."):
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        self.results_dir = results_dir

    def _generate_parameters(self, **params):
        params_list = []
        for k in params:
            if not isinstance(params[k], list):
                params[k] = [params[k]]

            # Replace each value with dict
            params_list.append([{k: v} for v in params[k]])
        
        for params in product(*params_list):
            # Into a dict
            params_d = {}
            for p in params:
                params_d.update(p)

            yield (params_d,)

    def run_experiment(self, exp_name, func, repetitions=1, n_jobs=-1,
                       automatic_seed=False, dryrun=False, **params):

        if dryrun:
            self.dryrun(exp_name, repetitions, **params)
            return

        exp_dir = os.path.dirname(job_file_name(self.results_dir, exp_name, None))
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # How many experiments in total?
        total = repetitions
        for p in params.values():
            if isinstance(p, list):
                total *= len(p)
        
        # Generate parameters
        reps = list(range(repetitions))
        parameter_set = self._generate_parameters(**params, repetition=reps)

        # Store the code we will run
        code_fname = os.path.join(exp_dir, "called_function.py")
        if not os.path.exists(code_fname):
            source = inspect.getsource(func)
            with open(code_fname, "w") as f:
                f.write(source)

        # Wrap function to allow storing results
        func = JobWrapper(func, self.results_dir, exp_name, automatic_seed)

        # Run it
        if n_jobs == 1:
            # No multiprocessing. This allows individual jobs to spawn
            # processes as they like.
            list(map(func, parameter_set))
        else:
            if n_jobs == -1:
                n_jobs = mp.cpu_count()

            # `chunksize` defined as in cpython's pool.py:
            chunksize, extra = divmod(total, n_jobs*4)
            if extra:
                chunksize += 1

            with mp.Pool(n_jobs) as p:
                list(p.imap_unordered(func, parameter_set, chunksize=chunksize))

    def dryrun(self, exp_name, repetitions=1, **params):
        # Generate parameters
        reps = list(range(repetitions))
        for params in self._generate_parameters(**params, repetition=reps):
            params = params[0]
            rep = params["repetition"]
            del params["repetition"]

            fname = job_file_name(self.results_dir, exp_name, rep, **params)

            if not os.path.exists(fname):
                print(f"Will run: {params}")
                print(fname)
                print()

    def fetch_results(self, exp_name):
        exp_dir = os.path.dirname(job_file_name(self.results_dir, exp_name, None))
        # TODO: parameters.json
        all_results = defaultdict(list)
        for exp in os.listdir(exp_dir):
            exp = os.path.join(exp_dir, exp)
            if not os.path.isdir(exp):
                continue

            for run in os.listdir(exp):
                run = os.path.join(exp, run)
                with open(run) as f:
                    results = json.load(f)
                try:
                    # This fails if there are duplicate keys
                    results = flatten(results, reducer=lambda _, k2: k2)
                except ValueError:
                    results = flatten(results, reducer="underscore")

                if all_results:
                    # Check that all the keys are there.
                    # Otherwise fill with None.
                    for k in all_results:
                        if not k in results:
                            results[k] = None
                    # Similarly, fill with None the keys that
                    # were missing from all_results.
                    n = len(next(iter(all_results.values())))
                    for k in results:
                        if not k in all_results:
                            all_results[k] = [None]*n

                # Append new results
                for k, v in results.items():
                    all_results[k].append(v)

        return pd.DataFrame(all_results)


def job_file_name(basedir, exp_name, repetition, **params):
    """Derives the file name for the results.
    """
    name = "-".join([f"{k}={v}" for k, v in params.items()])

    return os.path.join(basedir, exp_name, name, f"{repetition}.json")


class JobWrapper(object):
    """Wraps a function to run.

    It equips the function with parameter parsing and result storing.
    """
    def __init__(self, func, basedir, exp_name, automatic_seed):
        self.func = func
        self.basedir = basedir
        self.exp_name = exp_name
        self.automatic_seed = automatic_seed

    def __call__(self, params):
        # Destination file name.
        params = params[0]
        repetition = params["repetition"]
        del params["repetition"]
        fname = job_file_name(self.basedir, self.exp_name, repetition, **params)
        if os.path.exists(fname):
            print("Already run: {}".format(fname))
            return

        # This process name.
        setproctitle(f"eli: {fname}")

        # Log git commit if git repo.
        try:
            git_commit = git.log().split()[1]
        except:
            git_commit = "no-git-repo"

        if self.automatic_seed:
            # TODO: set seeds from other libraries too.
            np.random.seed(repetition)

        dt_started = str(arrow.utcnow())

        # Run.
        try:
            print(f"Running: {fname}")
            res = self.func(**params)
            print(f"Finished: {fname}")
        except Exception as e:
            print("Failed: {}".format(fname))
            print(e)
            return

        # Store results.
        if res is None:
            return

        results = {}
        results["name"] = os.path.basename(os.path.dirname(fname))
        results["results"] = res
        results["parameters"] = params
        results["repetition"] = repetition
        results["git-commit"] = git_commit
        results["started"] = dt_started
        results["finished"] = str(arrow.utcnow())

        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        
        with open(fname, "w") as f:
            json.dump(results, f, indent=4)
