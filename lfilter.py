#!/usr/bin/env python
'''
filter out a set of labels.
'''
import json

pv0 = json.load(open("data/pubs_validate0.json"))
av = json.load(open("data/assignment_validate.json"))

pv = dict()
for k in av:
    ids = set().union(*[ set(l) for l in av[k] ])
    if len(ids):
        pv.update({k: [au for au in pv0[k] if au['id'] in ids]})

json.dump(pv, open("data/pubs_validate.json", 'w'))
