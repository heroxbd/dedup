SHELL:=/bin/bash

assignment_validate.zip: assignment_validate.json
	ln -sf $^ result.json
	zip -9 $@ result.json
data/stage_train.json: data/pubs_train.json
	./baseline.py $^ -o $@

data/assignment_validate.json: data/pubs_validate.json
	./venue_bag.py $^ -o $@

org_bag.zip: org_bag.json
	ln -sf $^ result.json
	zip -9 $@ result.json
org_bag.json: data/validate/author
	./org_bag.py $^ -o $@

data/train: data/pubs_train.json
	mkdir -p $@/{item,author,abstract,keywords}
	./data_transfer.R $^ -o $@

features/train/c_org/%.h5: data/train/author/%.csv
	mkdir -p $(dir $@)
	./c_org.py $^ -o $@

features/train/c_keywords/%.h5: data/train/keywords/%.csv
	mkdir -p $(dir $@)
	./c_org.py $^ -o $@ --field keywords

features/train/label/%.h5: data/train/item/%.csv
	mkdir -p $(dir $@)
	./label.py $^ -o $@ --ref data/assignment_train.json

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
