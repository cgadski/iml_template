.PHONY: clean fmt upload

fmt:
	black *.py
	black util/

upload: build/ex_26_09.html
	scp $^ root@cgad.ski:/www/iml/

clean:
	rm -rf build/

# Use this target if want a plaintext .py file to be your main source.
%.ipynb: %.py
	jupytext $^ -o $@

build/%.html: %.ipynb
	jupyter nbconvert --output-dir=build $^ --to html --execute
