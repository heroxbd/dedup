SHELL:=/bin/bash
DS:=validate

DSP:=train validate0 test
DSA:=train validate validate0 test
-include $(wildcard *_names.mk)

define load-tpl
$(1)_names:=$(shell jq -r 'keys[]' < data/pubs_$(1).json)
$(1)_names.mk:
	echo '$(1)_names:=$($(1)_names)' > $@
endef

ifndef train_names
$(call load-tpl,train)
endif
ifndef validate_names
$(call load-tpl,validate)
endif
ifndef validate0_names
$(call load-tpl,validate0)
endif
ifndef test_names
$(call load-tpl,test)
endif

.PHONY: prepare
prepare: $(DSA:%=%_names.mk)

assignment_validate.zip: assignment_validate.json
	ln -sf $^ result.json
	zip -9 $@ result.json
data/stage_$(DS).json: data/pubs_$(DS).json
	./baseline.py $^ -o $@

data/venue_bag_$(DS).json: data/pubs_$(DS).json
	./venue_bag.py $^ -o $@
data/org_bag_${DS}.json: data/${DS}/author
	./org_bag.py $^ -o $@
data/%.score: data/%.json
	parallel python evaluate.py $^ --names {} ::: $($(DS)_names) > $@
data/%.pdf: data/%.score
	./pscore.R $^ -o $@ > $@.log

data/%.zip: data/%.json
	ln -sf $^ result.json
	zip -9 $@ result.json

data/venue_idf.csv: $(foreach D,${DSP},$($(D)_names:%=data/$(D)/venue/%.csv))
	./IDF.py $^ -o $@ --field venue
data/org_idf.csv: $(foreach D,${DSP},$($(D)_names:%=data/$(D)/org/%.csv))
	./IDF.py $^ -o $@ --field org
data/title_idf.csv: $(foreach D,${DSP},$($(D)_names:%=data/$(D)/title/%.csv))
	./IDF.py $^ -o $@ --field title
data/keywords_idf.csv: $(foreach D,${DSP},$($(D)_names:%=data/$(D)/keywords/%.csv))
	./IDF.py $^ -o $@ --field keywords

define DS-tpl
data/$(1)/author0/%.csv: data/$(1)/csv_flag
	touch $$@
data/$(1)/item0/%.csv: data/$(1)/csv_flag
	touch $$@
data/$(1)/abstract/%.csv: data/$(1)/csv_flag
	touch $$@
data/$(1)/keywords/%.csv: data/$(1)/csv_flag
	touch $$@
data/$(1)/csv_flag: data/pubs_$(1).json
	mkdir -p $$(dir $$@){item0,author0,abstract,keywords}
	./data_transfer.R $$^ -o $$(dir $$@)
	touch $$@
data/$(1)/dual/%.csv: data/$(1)/author/%.csv
	mkdir -p $$(dir $$@)
	./dual_marry.py $$^ -o $$@
data/$(1)/ia.csv: $($(1)_names:%=data/$(1)/item/%.csv) $($(1)_names:%=data/$(1)/abstract/%.csv)
	./combine-at.R $($(1)_names:%=data/$(1)/item/%.csv) --abstract $($(1)_names:%=data/$(1)/abstract/%.csv) -o $$@
data/$(1)/uniglue/%.csv: data/$(1)/item/%.csv data/$(1)/author/%.csv
	mkdir -p $$(dir $$@)
	./uni_glue_baseline.R $$< --author $$(word 2,$$^) -o $$@
data/$(1)/coauthor/%.csv: data/$(1)/author/%.csv
	mkdir -p $$(dir $$@)
	./coauthor_glue.R $$< -o $$@
data/$(1)/author/%.csv: data/$(1)/author0/%.csv
	mkdir -p $$(dir $$@)
	./venue_author_preprocess.R $$^ -o $$@ --field author

data/$(1)/item/%.csv: data/$(1)/item0/%.csv
	mkdir -p $$(dir $$@)
	./venue_author_preprocess.R $$^ -o $$@ --field item

data/$(1)/title/%.csv: data/$(1)/item/%.csv
	mkdir -p $$(dir $$@)
	./wordlist.py $$^ -o $$@ --field title
data/$(1)/venue/%.csv: data/$(1)/item/%.csv
	mkdir -p $$(dir $$@)
	./wordlist.py $$^ -o $$@ --field venue
data/$(1)/org/%.csv: data/$(1)/author/%.csv
	mkdir -p $$(dir $$@)
	./wordlist.py $$^ -o $$@ --field org
endef

# $(eval $(call DS-tpl,$(DS)))
$(foreach D,$(DSA),$(eval $(call DS-tpl,$(D))))

features/d2v_singlet.model: data/train/ia.csv
	python doc2vec.py -i $^ -o $@
features/d2v_doublet.model: data/train/ia.csv data/validate0/ia.csv
	python doc2vec.py -i $^ -o $@
features/d2v_triplet.model: data/train/ia.csv data/validate0/ia.csv data/test/ia.csv
	python doc2vec.py -i $^ -o $@
features/train/doc2vec_singlet_native/%.h5: data/train/item/%.csv features/d2v_singlet.model data/train/ia.csv
	mkdir -p $(dir $@)
	python doc2vec_pair_native.py -i $< -o $@ -m $(word 2,$^) -a $(word 3,$^) > $@.log
features/$(DS)/doc2vec_doublet_native/%.h5: data/$(DS)/item/%.csv features/d2v_doublet.model data/$(DS)/ia.csv
	mkdir -p $(dir $@)
	python doc2vec_pair_native.py -i $< -o $@ -m $(word 2,$^) -a $(word 3,$^) > $@.log
features/$(DS)/doc2vec_triplet_native/%.h5: data/$(DS)/item/%.csv features/d2v_triplet.model data/$(DS)/ia.csv
	mkdir -p $(dir $@)
	python doc2vec_pair_native.py -i $< -o $@ -m $(word 2,$^) -a $(word 3,$^) > $@.log

data/uni_glue_${DS}.json: $($(DS)_names:%=data/$(DS)/uniglue/%.csv)
	./org_bag.py $^ -o $@ --field uniglue
data/coauthor_glue_${DS}.json: $($(DS)_names:%=data/$(DS)/coauthor/%.csv)
	./org_bag.py $^ -o $@ --field uniglue

features/$(DS)/shortpath/%.json: data/$(DS)/author/%.csv
	mkdir -p $(dir $@)
	./short_path.R $< -o $@

#features/$(DS)/c_authorFN/%.h5: data/$(DS)/author/%.csv
#	mkdir -p $(dir $@)
#	./c_org.py $< -o $@ --field authorFN

features/$(DS)/c_org/%.h5: data/$(DS)/org/%.csv data/org_idf.csv
	mkdir -p $(dir $@)
	./c_org.py $< -o $@ --field org --idf $(word 2,$^)
features/$(DS)/sp_org/%.h5: features/$(DS)/c_org/%.h5 features/$(DS)/id_pairs/%.h5
	mkdir -p $(dir $@)
	./shortpath.py -i $< -o $@ --field org_logTFIDF -p $(word 2,$^)

features/$(DS)/c_title/%.h5: data/$(DS)/title/%.csv data/title_idf.csv
	mkdir -p $(dir $@)
	./c_org.py $< -o $@ --field title --idf $(word 2,$^)
features/$(DS)/sp_title/%.h5: features/$(DS)/c_title/%.h5 features/$(DS)/id_pairs/%.h5
	mkdir -p $(dir $@)
	./shortpath.py -i $< -o $@ --field title_logTFIDF -p $(word 2,$^)

features/$(DS)/c_venue/%.h5: data/$(DS)/venue/%.csv data/venue_idf.csv
	mkdir -p $(dir $@)
	./c_org.py $< -o $@ --field venue --idf $(word 2,$^)
features/$(DS)/sp_venue/%.h5: features/$(DS)/c_venue/%.h5 features/$(DS)/id_pairs/%.h5
	mkdir -p $(dir $@)
	./shortpath.py -i $< -o $@ --field venue_logTFIDF -p $(word 2,$^)

features/$(DS)/c_keywords/%.h5: data/$(DS)/keywords/%.csv data/keywords_idf.csv
	mkdir -p $(dir $@)
	./c_org.py $< -o $@ --field keywords --idf $(word 2,$^)
features/$(DS)/sp_keywords/%.h5: features/$(DS)/c_keywords/%.h5 features/$(DS)/id_pairs/%.h5
	mkdir -p $(dir $@)
	./shortpath.py -i $< -o $@ --field keywords_logTFIDF -p $(word 2,$^)

features/$(DS)/diff_year/%.h5: data/$(DS)/item/%.csv
	mkdir -p $(dir $@)
	./diff_year.py $^ -o $@ --field year

features/$(DS)/id_pairs/%.h5: data/$(DS)/keywords/%.csv
	mkdir -p $(dir $@)
	./id_pairs.py $^ -o $@ --field keywords

features/$(DS)/valid_index/%.h5: data/$(DS)/keywords/%.csv
	mkdir -p $(dir $@)
	./valid_index.py $^ -o $@ --field keywords

features/$(DS)/label/%.h5: data/$(DS)/item/%.csv
	mkdir -p $(dir $@)
	./label.py $^ -o $@ --ref data/assignment_$(DS).json

result/validate_val/kruskal/%.json: output/validate_val/%.h5 features/validate/id_pairs/%.h5
	mkdir -p $(dir $@)
	./MT_Kruskal.R $< -o $@ --id $(word 2,$^)

result/validate_val/likelihood/%.json: output/validate_val/%.h5 result/validate_val/kruskal/%.json
	mkdir -p $(dir $@)
	./likelihood.R $< -o $@ --id features/validate/id_pairs/$*.h5 --kruskal $(word 2,$^)

data/pubs_validate.json: data/pubs_validate0.json data/assignment_validate.json
	./lfilter.py

validate_val_names:=$(shell jq -r '.val[]' < data/validate/split_1fold.json)
result/validate_val.json: $(validate_val_names:%=result/validate_val/likelihood/%.json)
	./merge_final_assignment.R $^ -o $@

define merge-tpl
features/$(DS)/$(1).h5: $$($(DS)_names:%=features/$(DS)/$(1)/%.h5)
	./merge.py $$^ -o $$@ --field $(1)
endef

paired_features:=c_keywords c_org shortpath diff_year id_pairs valid_index c_title
paired_features+=c_venue doc2vec_singlet_native doc2vec_doublet_native label
paired_features+=doc2vec_triplet_native

$(foreach k,$(paired_features),$(eval $(call merge-tpl,$(k))))

# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
