# AURA — geliştirme kısayolları (cross-platform: macOS/Linux. Windows'ta setup.ps1 / run.ps1 / dev.ps1 kullanın.)
.PHONY: setup run doctor train eval metrics test lint format clean help

PY := .venv/bin/python
ifeq ($(OS),Windows_NT)
	PY := .venv/Scripts/python
endif

help:           ## Bu yardımı göster
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:          ## Tek-komut kurulum (bootstrap.py)
	python3 bootstrap.py --dev

run:            ## Tüm servisleri kaldır (inference :8080, qod :8081, nv :8082)
	./run.sh

doctor:         ## Ortam/sağlık kontrolü (bağımlılık, cihaz, ağırlık, config, profil)
	$(PY) tools/doctor.py

train:          ## Eğitim CLI yardımı
	$(PY) -m train --help

eval:           ## Değerlendirme — örnek video + QoD A/B
	$(PY) -m aura.eval --source data/samples/ornek.mp4 --ground-truth data/samples/ornek_gt.json --qod-comparison

metrics:        ## FTR §4 metrik raporu (test_video özetlerinden P/R/F1; --summaries DIR)
	$(PY) -m aura.eval --metrics-report --summaries eval_results/ab

test:           ## Unit testler (integration skip)
	$(PY) -m pytest -m "not integration"

lint:           ## ruff lint
	$(PY) -m ruff check .

format:         ## black format
	$(PY) -m black .

clean:          ## Geçici dosyaları temizle
	rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__ **/__pycache__ eval_results/*.json

video-test:     ## Gerçek video testi: make video-test VIDEO=/yol/video.mp4 → annotated mp4 + JSON kanıt
	$(PY) tools/test_video.py --source $(VIDEO) --device auto
