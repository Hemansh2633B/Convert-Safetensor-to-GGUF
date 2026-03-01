# simple makefile to build converters and run the example pipeline

.PHONY: all build-c build-cpp build-python convert run clean

all: build-c build-cpp build-python
	@echo "use 'make run' to execute example workflow"

build-c:
	@echo "building C converter"
	./c/build.sh

build-cpp:
	@echo "building C++ converter"
	./cpp/build.sh

build-python:
	@echo "(python dependencies handled at runtime)"

convert: build-c build-cpp
	@echo "converting qwen model with all tools"
	./scripts/convert_all.sh

run: convert

clean:
	rm -f c/safetensor_to_gguf cpp/safetensor_to_gguf qwen_model*.*
