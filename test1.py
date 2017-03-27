import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', action='store', dest='constant_value',
                    help='Store a constant value')
result = parser.parse_args()
print result.constant_value