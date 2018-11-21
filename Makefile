assignment_validate.zip: assignment_validate.json
	ln -sf $^ result.json
	zip -9 $@ result.json
# assignment_validate.json: pubs_validate.json
#	./baseline.py $^ -o $@
assignment_validate.json: pubs_validate.json
	./venue_bag.py $^ -o $@

org_bag.zip: org_bag.json
	ln -sf $^ result.json
	zip -9 $@ result.json
org_bag.json: data/tabular/author
	./org_bag.py $^ -o $@


# Delete partial files when the processes are killed.
.DELETE_ON_ERROR:
# Keep intermediate files around
.SECONDARY:
