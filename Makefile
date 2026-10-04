.PHONY: install run test bench quick docker docker-bench clean
install:      ; pip install -r requirements-dev.txt
run:          ; python -m kairos serve
test:         ; python -m pytest -q tests
bench:        ; python -m bench.run_bench
quick:        ; python -m bench.run_bench --quick
docker:       ; docker compose up --build
docker-bench: ; docker compose run --rm bench
clean:        ; rm -rf .kairos_cache traces __pycache__ */__pycache__
