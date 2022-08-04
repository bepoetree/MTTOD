"""
   MTTOD: preprocess.py

   implements data preprocessor for MTTOD

   This code is partially referenced from thu-spmi's damd-multiwoz repository:
   (https://github.com/thu-spmi/damd-multiwoz/blob/master/preprocess.py)

   Copyright 2021 ETRI LIRS, Yohan Lee
   Copyright 2019 Yichi Zhang

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os
import re
import copy
import argparse
from collections import OrderedDict

import spacy

from tqdm import tqdm

from utils import definitions
from utils.io_utils import load_json, save_json, load_text
from utils.clean_dataset import clean_text, clean_slot_values

from external_knowledges import MultiWozDB


class Preprocessor(object):
    def __init__(self, version):
        self.nlp = spacy.load("en_core_web_sm")

        self.data_dir = os.path.join("data/MultiWOZ_{}".format(version))

        self.save_dir = os.path.join(self.data_dir, "processed")
        os.makedirs(self.save_dir, exist_ok=True)

        if version == "2.0":
            data_name = "annotated_user_da_with_span_full.json"
            self.dev_list = load_text(os.path.join(self.data_dir, "valListFile.json"))
            self.test_list = load_text(os.path.join(self.data_dir, "testListFile.json"))
            self.do_tokenize_text = False   # In 2.0, tests have been tokenized already
        else:
            data_name = "data.json"
            self.dev_list = load_text(os.path.join(self.data_dir, "valListFile.txt"))
            self.test_list = load_text(os.path.join(self.data_dir, "testListFile.txt"))
            self.do_tokenize_text = True

        self.version = version

        self.mapping_pair = self.load_mapping_pair()

        self.get_db_values()

        self.preprocess_db()

        self.db = MultiWozDB(os.path.join(self.data_dir, "db"))

        self.data = load_json(os.path.join(self.data_dir, data_name))

        self.delex_sg_valdict_path = os.path.join(self.save_dir, "delex_single_valdict.json")
        self.delex_mt_valdict_path = os.path.join(self.save_dir, "delex_multi_valdict.json")
        self.ambiguous_val_path = os.path.join(self.save_dir, "ambiguous_values.json")

        if (not os.path.exists(self.delex_sg_valdict_path) or
                not os.path.exists(self.delex_mt_valdict_path) or
                not os.path.exists(self.ambiguous_val_path)):
            self.delex_sg_valdict, self.delex_mt_valdict, self.ambiguous_vals = self.get_delex_valdict()
        else:
            self.delex_sg_valdict = load_json(self.delex_sg_valdict_path)
            self.delex_mt_valdict = load_json(self.delex_mt_valdict_path)
            self.ambiguous_vals = load_json(self.ambiguous_val_path)

    def load_mapping_pair(self):
        mapping_pair = []

        curr_dir = os.path.dirname(__file__)
        with open(os.path.join(curr_dir, 'utils/mapping.pair'), 'r') as fin:
            for line in fin.readlines():
                fromx, tox = line.replace('\n', '').split('\t')

                mapping_pair.append((fromx, tox))

        return mapping_pair

    def get_db_values(self):
        processed = {}
        bspn_word = []

        value_set_path = os.path.join(self.data_dir, "db", "value_set.json")

        value_set = load_json(value_set_path)

        ontology_path = os.path.join(self.data_dir, "ontology.json")

        otlg = load_json(ontology_path)

        for domain, slots in value_set.items():
            # add all informable slots to bspn_word, create lists holder for values
            processed[domain] = {}
            bspn_word.append('['+domain+']')
            for slot, values in slots.items():
                s_p = definitions.NORMALIZE_SLOT_NAMES.get(slot, slot)
                if s_p in definitions.INFORMABLE_SLOTS[domain]:
                    bspn_word.append(s_p)
                    processed[domain][s_p] = []

        for domain, slots in value_set.items():
            # add all words of values of informable slots to bspn_word
            for slot, values in slots.items():
                s_p = definitions.NORMALIZE_SLOT_NAMES.get(slot, slot)
                if s_p in definitions.INFORMABLE_SLOTS[domain]:
                    for v in values:
                        _, v_p = clean_slot_values(domain, slot, v, self.mapping_pair)
                        v_p = ' '.join([token.text for token in self.nlp(v_p)]).strip()
                        processed[domain][s_p].append(v_p)
                        for x in v_p.split():
                            if x not in bspn_word:
                                bspn_word.append(x)

        for domain_slot, values in otlg.items():  # split domain-slots to domains and slots
            tokens = domain_slot.split('-')
            if len(tokens) == 3:
                domain = tokens[0]
                slot = tokens[-1]
            else:
                domain, slot = tokens

            if domain == 'bus':
                domain = 'taxi'
            if slot == 'price range':
                slot = 'pricerange'
            if slot == 'book stay':
                slot = 'stay'
            if slot == 'book day':
                slot = 'day'
            if slot == 'book people':
                slot = 'people'
            if slot == 'book time':
                slot = 'time'
            if slot == 'arrive by':
                slot = 'arrive'
            if slot == 'leave at':
                slot = 'leave'
            if slot == 'leaveat':
                slot = 'leave'
            # add all slots and words of values if not already in processed and bspn_word
            if slot not in processed[domain]:
                processed[domain][slot] = []
                bspn_word.append(slot)
            for v in values:
                _, v_p = clean_slot_values(domain, slot, v)
                v_p = ' '.join([token.text for token in self.nlp(v_p)]).strip()
                if v_p not in processed[domain][slot]:
                    processed[domain][slot].append(v_p)
                    for x in v_p.split():
                        if x not in bspn_word:
                            bspn_word.append(x)

        save_json(processed, os.path.join(self.data_dir, "db", "value_set_processed.json"))
        save_json(bspn_word, os.path.join(self.save_dir, "bspn_word_collection.json"))

        print("DB value set processed !")

    def preprocess_db(self):
        dbs = {}
        for domain in definitions.ALL_DOMAINS:
            db_path = os.path.join(self.data_dir, "db", "{}_db.json".format(domain))

            dbs[domain] = load_json(db_path)

            for idx, entry in enumerate(dbs[domain]):
                new_entry = copy.deepcopy(entry)
                for key, value in entry.items():
                    if type(value) is not str:
                        continue
                    del new_entry[key]
                    key, value = clean_slot_values(domain, key, value)
                    tokenize_and_back = ' '.join(
                        [token.text for token in self.nlp(value)]).strip()
                    new_entry[key] = tokenize_and_back
                dbs[domain][idx] = new_entry

            save_json(dbs[domain], os.path.join(self.data_dir, "db", "{}_db_processed.json".format(domain)))

            print("[{}] DB processed !".format(domain))

    def get_delex_valdict(self):
        skip_entry_type = {
            'taxi': ['taxi_phone'],
            'police': ['id'],
            'hospital': ['id'],
            'hotel': ['id', 'location', 'internet', 'parking', 'takesbookings',
                      'stars', 'price', 'n', 'postcode', 'phone'],
            'attraction': ['id', 'location', 'pricerange', 'price', 'openhours', 'postcode', 'phone'],
            'train': ['price', 'id'],
            'restaurant': ['id', 'location', 'introduction', 'signature', 'type', 'postcode', 'phone'],
        }
        entity_value_to_slot = {}
        ambiguous_entities = []
        for domain, db_data in self.db.dbs.items():
            print('Processing entity values in [%s]' % domain)
            if domain != 'taxi':
                for db_entry in db_data:
                    for slot, value in db_entry.items():
                        if slot not in skip_entry_type[domain]:
                            if type(value) is not str:
                                raise TypeError(
                                    "value '%s' in domain '%s' should be rechecked" % (slot, domain))
                            else:
                                slot, value = clean_slot_values(domain, slot, value)
                                value = ' '.join(
                                    [token.text for token in self.nlp(value)]).strip()
                                if value in entity_value_to_slot and entity_value_to_slot[value] != slot:
                                    # print(value, ": ",entity_value_to_slot[value], slot)
                                    ambiguous_entities.append(value)
                                entity_value_to_slot[value] = slot
            else:   # taxi db specific
                db_entry = db_data[0]
                for slot, ent_list in db_entry.items():
                    if slot not in skip_entry_type[domain]:
                        for ent in ent_list:
                            entity_value_to_slot[ent] = 'car'
        ambiguous_entities = set(ambiguous_entities)
        ambiguous_entities.remove('cambridge')
        ambiguous_entities = list(ambiguous_entities)
        for amb_ent in ambiguous_entities:   # departure or destination? arrive time or leave time?
            entity_value_to_slot.pop(amb_ent)
        entity_value_to_slot['parkside'] = 'address'
        entity_value_to_slot['parkside, cambridge'] = 'address'
        entity_value_to_slot['cambridge belfry'] = 'name'
        entity_value_to_slot['hills road'] = 'address'
        entity_value_to_slot['hills rd'] = 'address'
        entity_value_to_slot['Parkside Police Station'] = 'name'

        single_token_values = {}
        multi_token_values = {}
        for val, slt in entity_value_to_slot.items():
            if val in ['cambridge']:
                continue
            if len(val.split()) > 1:
                multi_token_values[val] = slt
            else:
                single_token_values[val] = slt

        single_token_values = OrderedDict(
            sorted(single_token_values.items(), key=lambda kv: len(kv[0]), reverse=True))

        save_json(single_token_values, self.delex_sg_valdict_path)

        print('single delex value dict saved!')

        multi_token_values = OrderedDict(
            sorted(multi_token_values.items(), key=lambda kv: len(kv[0]), reverse=True))

        save_json(multi_token_values, self.delex_mt_valdict_path)

        print('multi delex value dict saved!')

        save_json(ambiguous_entities, self.ambiguous_val_path)

        print('ambiguous value dict saved!')

        return single_token_values, multi_token_values, ambiguous_entities

    def delex_by_annotation(self, dial_turn):
        if self.version == "2.2":
            u = list(dial_turn['text'])
        else:
            u = dial_turn['text'].split()
        span = dial_turn['span_info']
        for s in span:
            slot = s[1]
            if slot == 'open':
                continue
            if definitions.DA_ABBR_TO_SLOT_NAME.get(slot):
                slot = definitions.DA_ABBR_TO_SLOT_NAME[slot]

            ### my code
            if slot == "price" and ("cheap" in s[2] or "moderate" in s[2] or "expensive" in s[2]):
                slot = "pricerange"
            ### my code (end)

            for idx in range(s[3], s[4] + 1):
                if idx >= len(u):
                    break

                u[idx] = ''
            try:
                u[s[3]] = '[value_'+slot+']'
            except (NameError, IndexError):
                u[5] = '[value_'+slot+']'

        if self.version == "2.2":
            u_delex = ''.join([t for t in u if t is not ''])
            u_delex = u_delex.replace("]", "] ")
            u_delex = u_delex.replace("[", " [")
            u_delex = ' '.join(u_delex.strip().split())
        else:
            u_delex = ' '.join([t for t in u if t is not ''])
            
        u_delex = u_delex.replace(
            '[value_address] , [value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace(
            '[value_address] , [value_address]', '[value_address]')
        u_delex = u_delex.replace('[value_name] [value_name]', '[value_name]')
        u_delex = u_delex.replace(
            '[value_name]([value_phone] )', '[value_name] ( [value_phone] )')

        return u_delex

    def delex_by_valdict(self, text):
        text = clean_text(text)

        text = re.sub(r'\d{5}\s?\d{5,7}', '[value_phone]', text)
        text = re.sub(r'\d[\s-]stars?', '[value_stars]', text)
        text = re.sub(r'\$\d+|\$?\d+.?(\d+)?\s(pounds?|gbps?)',
                      '[value_price]', text)
        text = re.sub(r'tr[\d]{4}', '[value_id]', text)
        text = re.sub(
            r'([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
            '[value_postcode]', text)

        for value, slot in self.delex_mt_valdict.items():
            text = text.replace(value, '[value_%s]' % slot)

        for value, slot in self.delex_sg_valdict.items():
            tokens = text.split()
            for idx, tk in enumerate(tokens):
                if tk == value:
                    tokens[idx] = '[value_%s]' % slot
            text = ' '.join(tokens)

        for ambg_ent in self.ambiguous_vals:
            # ely is a place, but appears in words like moderately
            start_idx = text.find(' '+ambg_ent)
            if start_idx == -1:
                continue
            front_words = text[:start_idx].split()
            ent_type = 'time' if ':' in ambg_ent else 'place'

            for fw in front_words[::-1]:
                if fw in ['arrive', 'arrives', 'arrived', 'arriving', 'arrival',
                          'destination', 'there', 'reach', 'to', 'by', 'before']:
                    slot = '[value_arrive]' if ent_type == 'time' else '[value_destination]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)
                elif fw in ['leave', 'leaves', 'leaving', 'depart', 'departs', 'departing', 'departure',
                            'from', 'after', 'pulls']:
                    slot = '[value_leave]' if ent_type == 'time' else '[value_departure]'
                    text = re.sub(' '+ambg_ent, ' '+slot, text)

        text = text.replace('[value_car] [value_car]', '[value_car]')

        return text

    def preprocess(self):
        # preprocess_main
        train_data, dev_data, test_data = {}, {}, {}
        count = 0
        self.unique_da = {}
        ordered_sysact_dict = {}
        # yyy
        nn = 0
        for fn, raw_dial in tqdm(list(self.data.items())):
            nn += 1
            '''
            if nn == 100:
                break
            '''
            if ".json" not in fn:
                fn += ".json"

            if fn in ['pmul4707.json', 'pmul2245.json', 'pmul4776.json', 'pmul3872.json', 'pmul4859.json']:
                continue
            count += 1

            # NOTE: apply clean_slot_value to goal??
            compressed_goal = {}  # for every dialog, keep track the goal, domains, requests
            dial_domains, dial_reqs = [], []
            for dom, g in raw_dial['goal'].items():
                if dom != 'topic' and dom != 'message' and g:
                    if g.get('reqt'):  # request info. eg. postcode/address/phone
                        # normalize request slots
                        for i, req_slot in enumerate(g['reqt']):
                            if definitions.NORMALIZE_SLOT_NAMES.get(req_slot):
                                g['reqt'][i] = definitions.NORMALIZE_SLOT_NAMES[req_slot]
                                dial_reqs.append(g['reqt'][i])
                    compressed_goal[dom] = g
                    if dom in definitions.ALL_DOMAINS:
                        dial_domains.append(dom)

            dial_reqs = list(set(dial_reqs))

            dial = {'goal': compressed_goal, 'log': []}
            single_turn = {}
            constraint_dict = OrderedDict()
            prev_constraint_dict = {}
            prev_turn_domain = ['general']
            ordered_sysact_dict[fn] = {}

            for turn_num, dial_turn in enumerate(raw_dial['log']):
                # for user turn, have text
                # sys turn: text, belief states(metadata), dialog_act, span_info
                dial_state = dial_turn['metadata']

                if self.do_tokenize_text:
                    dial_turn['text'] = ' '.join([t.text for t in self.nlp(dial_turn['text'])])

                    dial_turn["text"] = ' '.join(
                        dial_turn["text"].replace(".", " . ").split())

                if not dial_state:   # user
                    # delexicalize user utterance, either by annotation or by val_dict
                    u = ' '.join(clean_text(dial_turn['text']).split())
                    if 'span_info' in dial_turn and dial_turn['span_info']:
                        u_delex = clean_text(self.delex_by_annotation(dial_turn))
                    else:
                        u_delex = self.delex_by_valdict(dial_turn['text'])

                    single_turn['user'] = u
                    single_turn['user_delex'] = u_delex

                else:   # system
                    # delexicalize system response, either by annotation or by val_dict
                    if 'span_info' in dial_turn and dial_turn['span_info']:
                        s_delex = clean_text(self.delex_by_annotation(dial_turn))
                    else:
                        if not dial_turn['text']:
                            print(fn)
                        s_delex = self.delex_by_valdict(dial_turn['text'])
                    single_turn['resp'] = s_delex
                    single_turn['nodelx_resp'] = ' '.join(clean_text(dial_turn['text']).split())

                    # get belief state, semi=informable/book=requestable, put into constraint_dict

                    # this has no delete operations because it has cumulative property
                    #curr_constraint_dict = OrderedDict()
                    for domain in dial_domains:
                        if not constraint_dict.get(domain):
                            constraint_dict[domain] = OrderedDict()
                        info_sv = dial_state[domain]['semi']
                        for s, v in info_sv.items():
                            s, v = clean_slot_values(domain, s, v)
                            if len(v.split()) > 1:
                                v = ' '.join(
                                    [token.text for token in self.nlp(v)]).strip()
                            if v != '':
                                constraint_dict[domain][s] = v
                        book_sv = dial_state[domain]['book']
                        for s, v in book_sv.items():
                            if s == 'booked':
                                continue
                            s, v = clean_slot_values(domain, s, v)
                            if len(v.split()) > 1:
                                v = ' '.join(
                                    [token.text for token in self.nlp(v)]).strip()
                            if v != '':
                                constraint_dict[domain][s] = v
                    '''
                    # constraint_dict update
                    for domain, sv_dict in curr_constraint_dict.items():
                        if domain not in constraint_dict:
                            constraint_dict[domain] = OrderedDict()

                        for s, v in sv_dict.items():
                            constraint_dict[domain][s] = v
                    '''
                    constraints = []  # list in format of [domain] slot value
                    cons_delex = []
                    turn_dom_bs = []
                    for domain, info_slots in constraint_dict.items():
                        if info_slots:
                            constraints.append('['+domain+']')
                            cons_delex.append('['+domain+']')
                            for slot, value in info_slots.items():
                                constraints.append('[value_' + slot + ']')
                                constraints.extend(value.split())
                                cons_delex.append('[value_' + slot + ']')
                            if domain not in prev_constraint_dict:
                                turn_dom_bs.append(domain)
                            elif prev_constraint_dict[domain] != constraint_dict[domain]:
                                turn_dom_bs.append(domain)

                    sys_act_dict = {}
                    turn_dom_da = set()
                    for act in dial_turn['dialog_act']:
                        d, a = act.split('-')  # split domain-act
                        turn_dom_da.add(d)
                    turn_dom_da = list(turn_dom_da)
                    if len(turn_dom_da) != 1 and 'general' in turn_dom_da:
                        turn_dom_da.remove('general')
                    if len(turn_dom_da) != 1 and 'booking' in turn_dom_da:
                        turn_dom_da.remove('booking')

                    # get turn domain
                    turn_domain = turn_dom_bs
                    for dom in turn_dom_da:
                        if dom != 'booking' and dom not in turn_domain:
                            turn_domain.append(dom)
                    if not turn_domain:
                        turn_domain = prev_turn_domain
                    if len(turn_domain) == 2 and 'general' in turn_domain:
                        turn_domain.remove('general')
                    if len(turn_domain) == 2:
                        if len(prev_turn_domain) == 1 and prev_turn_domain[0] == turn_domain[1]:
                            turn_domain = turn_domain[::-1]

                    # get system action
                    for dom in turn_domain:
                        sys_act_dict[dom] = {}
                    add_to_last_collect = []
                    booking_act_map = {
                        'inform': 'offerbook', 'book': 'offerbooked'}
                    for act, params in dial_turn['dialog_act'].items():
                        if act == 'general-greet':
                            continue
                        d, a = act.split('-')
                        if d == 'general' and d not in sys_act_dict:
                            sys_act_dict[d] = {}
                        if d == 'booking':
                            d = turn_domain[0]
                            a = booking_act_map.get(a, a)
                        add_p = []
                        for param in params:
                            p = param[0]
                            if p == 'none':
                                continue
                            elif definitions.DA_ABBR_TO_SLOT_NAME.get(p):
                                p = definitions.DA_ABBR_TO_SLOT_NAME[p]
                            if p not in add_p:
                                add_p.append(p)
                        add_to_last = True if a in [
                            'request', 'reqmore', 'bye', 'offerbook'] else False
                        if add_to_last:
                            add_to_last_collect.append((d, a, add_p))
                        else:
                            sys_act_dict[d][a] = add_p
                    for d, a, add_p in add_to_last_collect:
                        sys_act_dict[d][a] = add_p

                    for d in copy.copy(sys_act_dict):
                        acts = sys_act_dict[d]
                        if not acts:
                            del sys_act_dict[d]
                        if 'inform' in acts and 'offerbooked' in acts:
                            for s in sys_act_dict[d]['inform']:
                                sys_act_dict[d]['offerbooked'].append(s)
                            del sys_act_dict[d]['inform']

                    ordered_sysact_dict[fn][len(dial['log'])] = sys_act_dict

                    sys_act = []
                    if 'general-greet' in dial_turn['dialog_act']:
                        sys_act.extend(['[general]', '[greet]'])
                    for d, acts in sys_act_dict.items():
                        sys_act += ['[' + d + ']']
                        for a, slots in acts.items():
                            self.unique_da[d+'-'+a] = 1
                            sys_act += ['[' + a + ']']
                            sys_act += slots

                    # get db pointers
                    matnums = self.db.get_match_num(constraint_dict)
                    match_dom = turn_domain[0] if len(
                        turn_domain) == 1 else turn_domain[1]
                    match = matnums[match_dom]
                    dbvec = self.db.addDBPointer(match_dom, match)
                    bkvec = self.db.addBookingPointer(dial_turn['dialog_act'])

                    # 4 database pointer for domains, 2 for booking
                    single_turn['pointer'] = ','.join(
                        [str(d) for d in dbvec + bkvec])
                    single_turn['match'] = str(match)
                    single_turn['constraint'] = ' '.join(constraints)
                    single_turn['cons_delex'] = ' '.join(cons_delex)
                    single_turn['sys_act'] = ' '.join(sys_act)
                    single_turn['turn_num'] = len(dial['log'])
                    single_turn['turn_domain'] = ' '.join(
                        ['['+d+']' for d in turn_domain])

                    prev_turn_domain = copy.deepcopy(turn_domain)
                    prev_constraint_dict = copy.deepcopy(constraint_dict)

                    if 'user' in single_turn:
                        dial['log'].append(single_turn)

                        for t in single_turn['user_delex'].split():
                            if '[' in t and ']' in t and not t.startswith('[') and not t.endswith(']'):
                                single_turn['user_delex'].replace(t, t[t.index('['): t.index(']')+1])

                    single_turn = {}

            if fn in self.dev_list:
                dev_data[fn] = dial
            elif fn in self.test_list:
                test_data[fn] = dial
            else:
                train_data[fn] = dial

        print("Save preprocessed data to {} (#train: {}, #dev: {}, #test: {})"
                .format(self.save_dir, len(train_data), len(dev_data), len(test_data)))

        save_json(train_data, os.path.join(self.save_dir, "train_data.json"))
        save_json(dev_data, os.path.join(self.save_dir, "dev_data.json"))
        save_json(test_data, os.path.join(self.save_dir, "test_data.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument for preprocessing")

    parser.add_argument("-version", type=str, default="2.0", choices=["2.0", "2.1", "2.2"])

    args = parser.parse_args()

    preprocessor = Preprocessor(args.version)

    preprocessor.preprocess()
