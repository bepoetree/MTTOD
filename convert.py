import json
import argparse
from collections import defaultdict

from utils.io_utils import load_json, save_json


def bspn_to_constraint_dict(bspn):
    bspn = bspn.replace('<bos_belief>', '')
    bspn = bspn.replace('<eos_belief>', '')
    bspn = bspn.strip().split()

    constraint_dict = {}
    domain, slot = None, None
    for token in bspn:
        if token.startswith('['):
            token = token[1:-1]

            if token.startswith('value_'):
                if domain is None:
                    continue

                if domain not in constraint_dict:
                    constraint_dict[domain] = {}

                slot = token.split('_')[1]

                constraint_dict[domain][slot] = []
            else:
                domain = token

        else:
            try:
                constraint_dict[domain][slot].append(token)
            except KeyError:
                continue

    for domain, sv_dict in constraint_dict.items():
        for s, value_tokens in sv_dict.items():
            constraint_dict[domain][s] = ' '.join(value_tokens)

    return constraint_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conversion Output')

    parser.add_argument('-input', type=str, required=True)
    parser.add_argument('-output', type=str, required=True)

    args = parser.parse_args()

    results = load_json(args.input)

    converted_results = defaultdict(list)
    for dial_id, dial in results.items():
        dial_id = dial_id.split('.')[0]
        for turn in dial:
            converted_turn = {'response': '', 'state': {}}
            resp = turn['resp_gen']

            resp = resp.replace('<bos_resp>', '')
            resp = resp.replace('<eos_resp>', '')

            converted_turn['response'] = resp.strip()

            converted_turn['state'] = bspn_to_constraint_dict(turn['bspn_gen'])
            
            converted_results[dial_id].append(converted_turn)

    save_json(converted_results, args.output)