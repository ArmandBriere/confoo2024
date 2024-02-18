defautl: build

build:
	marp --pdf --allow-local-files README.md

serve:
	marp -s .
