import argparse
from pythainlp.benchmarks import word_tokenisation

parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('--input',
    action="store",
    help="""
        path to file that you want to compare against
        a standard dataset or a custom test file
    """
)

parser.add_argument('--dataset',
    action="store",
    help="[orchid|best]"
)

parser.add_argument('--custom-test-file',
    action="store",
    help="path to your custom test file"
)

args = parser.parse_args()

def _read_file(path):
    with open(path, 'r') as f:
        lines = map(lambda r: r.strip(), f.readlines())
    return list(lines)

# @todo #1 read input file / compute state
# @todo #1 print stats
print(args.input)
actual = _read_file(args.input)
expected = _read_file(args.custom_test_file)

assert len(actual) == len(expected), \
    'Input and test files do not have the same number of samples'
print('Benchmarking %s against %s with %d samples in total' % (
    args.input, args.custom_test_file, len(actual)
    ))

df_res = word_tokenisation.benchmark(expected, actual) \
    .describe() 
df_res.columns = [
    'char_level:tp',
    'char_level:tn',
    'char_level:fp',
    'char_level:fn',
    'char_level:precision',
    'char_level:recall',
    'char_level:f1',
    'word_level:accuracy'
]

df_res = df_res.T.reset_index(0)

df_res['mean±std'] = df_res.apply(
    lambda r: '%2.2f±%2.2f' % (r['mean'], r['std']),
    axis=1
)

df_res['metric'] = df_res['index']

print("============== Benchmark Result ==============")
print(df_res[['metric', 'mean±std', 'min', 'max']].to_string(index=False))