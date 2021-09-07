"""
   MTTOD: utils/definitions.py

   Defines slot names and domain names for MTTOD

   This code is referenced from thu-spmi's damd-multiwoz repository:
   (https://github.com/thu-spmi/damd-multiwoz/blob/master/ontology.py)

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

ALL_DOMAINS = ["attraction", "hotel", "restaurant", "taxi", "train", "hospital", "police"]

NORMALIZE_SLOT_NAMES = {
    "car type": "car",
    "entrance fee": "price",
    "duration": "time",
    "leaveat": 'leave',
    'arriveby': 'arrive',
    'trainid': 'id'
}

REQUESTABLE_SLOTS = {
    "taxi": ["car", "phone"],
    "police": ["postcode", "address", "phone"],
    "hospital": ["address", "phone", "postcode"],
    "hotel": ["address", "postcode", "internet", "phone", "parking",
              "type", "pricerange", "stars", "area", "reference"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference"],
    "train": ["time", "leave", "price", "arrive", "id", "reference"],
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
}

ALL_REQSLOT = ["car", "address", "postcode", "phone", "internet", "parking", "type", "pricerange", "food",
                      "stars", "area", "reference", "time", "leave", "price", "arrive", "id"]

INFORMABLE_SLOTS = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}

ALL_INFSLOT = ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name",
               "leave", "destination", "departure", "arrive", "department", "food", "time"]

EXTRACTIVE_SLOT = ["leave", "arrive", "destination", "departure", "type", "name", "food"]

DA_ABBR_TO_SLOT_NAME = {
    'addr': "address",
    'fee': "price",
    'post': "postcode",
    'ref': 'reference',
    'ticket': 'price',
    'depart': "departure",
    'dest': "destination",
}

DIALOG_ACTS = {
    'restaurant': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'hotel': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'attraction': ['inform', 'request', 'nooffer', 'recommend', 'select'],
    'train': ['inform', 'request', 'nooffer', 'offerbook', 'offerbooked', 'select'],
    'taxi': ['inform', 'request'],
    'police': ['inform', 'request'],
    'hospital': ['inform', 'request'],
    'general': ['bye', 'greet', 'reqmore', 'welcome'],
}

BOS_USER_TOKEN = "<bos_user>"
EOS_USER_TOKEN = "<eos_user>"

USER_TOKENS = [BOS_USER_TOKEN, EOS_USER_TOKEN]

BOS_BELIEF_TOKEN = "<bos_belief>"
EOS_BELIEF_TOKEN = "<eos_belief>"

BELIEF_TOKENS = [BOS_BELIEF_TOKEN, EOS_BELIEF_TOKEN]

BOS_DB_TOKEN = "<bos_db>"
EOS_DB_TOKEN = "<eos_db>"

DB_TOKENS = [BOS_DB_TOKEN, EOS_DB_TOKEN]

BOS_ACTION_TOKEN = "<bos_act>"
EOS_ACTION_TOKEN = "<eos_act>"

ACTION_TOKENS = [BOS_ACTION_TOKEN, EOS_ACTION_TOKEN]

BOS_RESP_TOKEN = "<bos_resp>"
EOS_RESP_TOKEN = "<eos_resp>"

RESP_TOKENS = [BOS_RESP_TOKEN, EOS_RESP_TOKEN]

DB_NULL_TOKEN = "[db_null]"
DB_0_TOKEN = "[db_0]"
DB_1_TOKEN = "[db_1]"
DB_2_TOKEN = "[db_2]"
DB_3_TOKEN = "[db_3]"

DB_STATE_TOKENS = [DB_NULL_TOKEN, DB_0_TOKEN, DB_1_TOKEN, DB_2_TOKEN, DB_3_TOKEN]

SPECIAL_TOKENS = USER_TOKENS + BELIEF_TOKENS + DB_TOKENS + ACTION_TOKENS + RESP_TOKENS + DB_STATE_TOKENS
