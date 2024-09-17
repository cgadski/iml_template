.PHONY: clean fmt

fmt:
	black *.py
	black util/

clean:
	rm -rf build/

# Use this target if want a plaintext .py file to be your main source.
%.ipynb: %.py
	jupytext $^ -o $@

build/%.html: %.ipynb
	jupyter nbconvert --output-dir=build $^ --to html --execute

