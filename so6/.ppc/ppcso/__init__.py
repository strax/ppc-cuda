from typing import Optional
import ppcgrader.config


class Config(ppcgrader.config.Config):
    def __init__(self, openmp: bool, code: str, gpu: bool = False):
        from . import info
        super().__init__(binary='so',
                         cfg_file=__file__,
                         gpu=gpu,
                         openmp=openmp,
                         info=info,
                         code=code)

    def parse_output(self, output):
        input_data = {
            "n": None,
            "data": None,
        }
        output_data = {
            "result": None,
        }
        output_errors = {
            "type": 0,
            "correct": None,
        }

        statistics = {}

        def parse_matrix(string):
            M = string.strip('[]').split(';')
            M = [row.strip() for row in M]
            M = [row.split(" ") for row in M]
            M = [[int(e) for e in row if e != ''] for row in M]
            return M

        for line in output.splitlines():
            splitted = line.split('\t')
            if splitted[0] == 'result':
                errors = {
                    'fail': True,
                    'pass': False,
                    'done': False
                }[splitted[1]]
            elif splitted[0] == 'time':
                time = float(splitted[1])
            elif splitted[0] == 'perf_wall_clock_ns':
                time = int(splitted[1]) / 1e9
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0].startswith('perf_'):
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0] == 'n':
                input_data['n'] = int(splitted[1])
            elif splitted[0] == 'input':
                input_data["data"] = parse_matrix(splitted[1])[0]
            elif splitted[0] == 'output':
                output_data["result"] = parse_matrix(splitted[1])[0]
            elif splitted[0] == 'correct':
                output_errors["correct"] = parse_matrix(splitted[1])[0]
            elif splitted[0] == 'error_type':
                output_errors["type"] = int(splitted[1])

        return time, errors, input_data, output_data, output_errors, statistics

    # TODO: Implement method
    def format_output(self, output) -> Optional[str]:
        raise NotImplementedError
