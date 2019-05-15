.PHONY: pull data install_self


all: pull .cells.dataset .requirements install_self


.requirements: requirements.txt
	pip install -U -r requirements.txt
	cp requirements.txt .requirements

install_self:
	python setup.py develop

pull:
	git pull
	git submodule update

# losses.py:
# 	# Got diceloss
# 	wget 'https://raw.githubusercontent.com/kevinzakka/pytorch-goodies/eb56b4b7374b31e15f31d61c2f64382b1540d72a/losses.py' -O $@

cells.zip:
	curl https://www.dropbox.com/s/jy34yowcf85ydba/data.zip?dl=0 -o cells.zip


.cells.dataset: cells.zip
	unzip cells.zip 2>&1 > $@

sktdl_utils/weight_init.py:
	curl https://gist.githubusercontent.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5/raw/d837f5c96a727d1b3292a3eedf06c19aca6c1a8a/weight_init.py -o $@
