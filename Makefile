SHELL:=/bin/bash
# data/pubs_train.json
-include train_names.mk
ifndef train_names
train_names:=$(shell jq -r 'keys[]' < data/pubs_train.json)
train_names.mk:
	echo 'train_names:=$(train_names)' > $@
endif

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

define merge-tpl
features/train/$(1).h5: $$(train_names:%=features/train/$(1)/%.h5)
	./merge.py $$^ -o $$@ --field $(1)
endef
$(foreach k,c_keywords c_org label,$(eval $(call merge-tpl,$(k))))

features/train/idx.csv: $(train_names:%=features/train/c_keywords/%.h5)
	for f in $^; do \
		echo -n $$(basename --suffix=.h5 $$f) >> $@ \
		python -c "import h5py; print(h5py.File($$f)['c_keywords'].size)" >> $@ \
	done

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
