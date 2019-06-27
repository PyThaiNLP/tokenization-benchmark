# -*- coding: utf-8 -*-

import json
import os
import argparse
import yaml

from pythainlp.benchmarks import word_tokenisation

parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('--input',
    action="store",
    help="""
        path to file that you want to compare against
        a standard dataset or a custom test file
    """
)

parser.add_argument('--test-file',
    action="store",
    help="path to test file"
)

args = parser.parse_args()

def _read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = map(lambda r: r.strip(), f.readlines())
    return list(lines)

print(args.input)
actual = _read_file(args.input)
expected = _read_file(args.test_file)

assert len(actual) == len(expected), \
    'Input and test files do not have the same number of samples'
print('Benchmarking %s against %s with %d samples in total' % (
    args.input, args.test_file, len(actual)
    ))

df_raw = word_tokenisation.benchmark(expected, actual)

df_res =  df_raw\
    .describe() 
df_res = df_res[[
    'char_level:tp',
    'char_level:tn',
    'char_level:fp',
    'char_level:fn',
    'char_level:precision',
    'char_level:recall',
    'char_level:f1',
    'word_level:precision',
    'word_level:recall',
    'word_level:f1',
]]

df_res = df_res.T.reset_index(0)

df_res['mean±std'] = df_res.apply(
    lambda r: '%2.2f±%2.2f' % (r['mean'], r['std']),
    axis=1
)

df_res['metric'] = df_res['index']

print("============== Benchmark Result ==============")
print(df_res[['metric', 'mean±std', 'min', 'max']].to_string(index=False))


# save file to json
data = {}
for r in df_res.to_dict('records'):
    metric = r['index']
    del r['index']
    data[metric] = r

dir_name = os.path.dirname(args.input)
file_name = args.input.split("/")[-1].split(".")[0]

res_path = "%s/eval-%s.yml" % (dir_name, file_name)
print("Evaluation result is saved to %s." % res_path)
with open(res_path, 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)


res_path = "%s/eval-details-%s.json" % (dir_name, file_name)
with open(res_path, "w") as f:
    samples = []
    for i, r in enumerate(df_raw.to_dict("records")):
        expected, actual = r["expected"], r["actual"]
        del r["expected"]
        del r["actual"]

        samples.append(dict(
            metrics=r,
            expected=expected,
            actual=actual,
            id=i
        ))

    details = dict(
        metrics=data,
        samples=samples
    )

    json.dump(details,f, ensure_ascii=False)
