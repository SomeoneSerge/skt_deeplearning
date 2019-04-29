.PHONY: pull data python-deps


all: pull BBBC018_v1_images-fixed python-deps


python-deps:
	pip install -U -r requirements.txt
	pip install -U .

pull:
	git pull
	git submodule update

# losses.py:
# 	# Got diceloss
# 	wget 'https://raw.githubusercontent.com/kevinzakka/pytorch-goodies/eb56b4b7374b31e15f31d61c2f64382b1540d72a/losses.py' -O $@

cells.zip:
	wget -c https://www.dropbox.com/s/jy34yowcf85ydba/data.zip?dl=0 -O cells.zip

BBBC018_v1_images-fixed: cells.zip
	unzip -fo cells.zip
	touch $@
