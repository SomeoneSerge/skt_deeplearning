.PHONY: pull data install_self


all: pull BBBC018_v1_images-fixed .requirements


.requirements: requirements.txt
	pip install -U -r requirements.txt
	cp requirements.txt .requirements

install_self:
	pip install -U .

pull:
	git pull
	git submodule update

# losses.py:
# 	# Got diceloss
# 	wget 'https://raw.githubusercontent.com/kevinzakka/pytorch-goodies/eb56b4b7374b31e15f31d61c2f64382b1540d72a/losses.py' -O $@

cells.zip:
	wget -c https://www.dropbox.com/s/jy34yowcf85ydba/data.zip?dl=0 -O cells.zip


.cells.dataset: cells.zip
	unzip cells.zip 2>&1 > $@
