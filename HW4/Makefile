.PHONY: pull

all: Train_rev1.csv


pull:
	git lfs install
	git lfs fetch
	git lfs checkout -- .
	git pull

Train_rev1.zip: pull


Train_rev1.csv: Train_rev1.zip
	unzip -fq $<


