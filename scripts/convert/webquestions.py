
"""A script to convert the default WebQuestions dataset to the format:

'{"question": "q1", "answer": ["a11", ..., "a1i"]}'
...
'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'

"""

import argparse
import re
import json

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# Reading dataset
with open(args.input) as f:
    dataset = json.load(f)

# Iterating and writing question-answer pairs
with open(args.output, 'w') as f:
    for ex in dataset:
        question = ex['utterance']
        answer = ex['targetValue']
        answer = re.findall(
            r'(?<=\(description )(.+?)(?=\) \(description|\)\)$)', answer
        )
        answer = [a.replace('"', '') for a in answer]
        f.write(json.dumps({'question': question, 'answer': answer}))
        f.write('\n')
