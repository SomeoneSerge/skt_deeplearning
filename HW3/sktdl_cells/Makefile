.PHONY: pull data python-deps unzip-cells


TEST_CELLS_DATASET=BBBC018_v1_images-fixed/test/00733-actin.DIB.bmp

deps: $(TEST_CELLS_DATASET) losses.py python-deps


python-deps:
	pip install -r requirements.txt
	pip install .

pull:
	git pull
	git submodule update

losses.py:
	# Got diceloss
	wget 'https://raw.githubusercontent.com/kevinzakka/pytorch-goodies/eb56b4b7374b31e15f31d61c2f64382b1540d72a/losses.py' -O $@

cells.zip:
	wget https://www.dropbox.com/s/jy34yowcf85ydba/data.zip?dl=0 -O cells.zip


unzip-cells: cells.zip
	unzip -fo cells.zip


$(TEST_CELLS_DATASET): unzip-cells
	touch $@
