.PHONY: pull


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

