.PHONY: pull data python-deps


TEST_CELLS_DATASET=BBBC018_v1_images-fixed/test/00733-actin.DIB.bmp

deps: $(TEST_CELLS_DATASET) ternaus_v2.pt losses.py python-deps


python-deps:
	pip install -r requirements.txt
	pip install .

pull:
	git pull
	git submodule update

ternaus_v2.pt:
	# Source: https://github.com/ternaus/TernausNetV2
	# Download URL: https://drive.google.com/file/d/1k95VGNZG74Vvu-X-MSpbaHjMDvNEepIi/view?usp=sharing
	./gdrive_download.sh 1k95VGNZG74Vvu-X-MSpbaHjMDvNEepIi $@

losses.py:
	# Got diceloss
	wget 'https://raw.githubusercontent.com/kevinzakka/pytorch-goodies/eb56b4b7374b31e15f31d61c2f64382b1540d72a/losses.py' -O $@

cells.zip:
	wget https://www.dropbox.com/s/jy34yowcf85ydba/data.zip?dl=0 -O cells.zip

$(TEST_CELLS_DATASET): cells.zip
	unzip -fq cells.zip
	touch $@
