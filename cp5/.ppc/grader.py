#!/usr/bin/env python3

from ppcgrader.cli import cli
import ppccp

if __name__ == "__main__":
    cli(
        ppccp.Config(code='cp5',
                     gpu=True,
                     openmp=False,
                     single_precision=True,
                     vectorize=True))
