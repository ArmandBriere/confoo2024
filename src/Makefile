default:
	@echo "Please specify a target to build"

.PHONY: virtualenv activate download convert

PYTHON_BENCHMARK_RESULTS = python_data.txt
GOLANG_BENCHMARK_RESULTS = golang_data.txt

# Create virtualenv
virtualenv:
	@echo "Create virtualenv"
	python3 -m venv env

# Run Python benchmark
benchmark-python:
	@echo "Benchmark python"
	@rm -f ${PYTHON_BENCHMARK_RESULTS}
	for run in {1..100}; do python main.py --img mouton.jpg >> ${PYTHON_BENCHMARK_RESULTS}; done
	@awk '{s+=$$1}END{print "Average inference time in Python:", s/NR}' ${PYTHON_BENCHMARK_RESULTS}

# Run Golang benchmark
benchmark-golang:
	@echo "Benchmark golang"
	@rm -f ${GOLANG_BENCHMARK_RESULTS}
	for run in {1..100}; do ./main >> ${GOLANG_BENCHMARK_RESULTS}; done
	@awk '{s+=$$1}END{print "Average inference time in Golang:", s/NR}' ${GOLANG_BENCHMARK_RESULTS}