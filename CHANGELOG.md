# Changelog

All notable changes to the CENTROID project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Ongoing optimizations and feature expansions.

## [0.1.0] - 2026-09-08

### Added
- Modular 8-stage local text ingestion, tokenization, semantic clustering, and reporting pipeline.
- Stage 00 state checkpoint manager (`00_state.py`) with SHA-256 caching and partial resumption.
- Stage 01 static web scraper (`01_fetch.py`) with noise stripping and response size caps.
- Stage 02 document extractor (`02_extract_pdf.py`) supporting PDF, DOCX, and TXT files.
- Stage 03 9-rule word scanner and document-level TF-IDF computation (`03_scan_words.py`).
- Stage 04 autonomous DuckDuckGo keyword expansion (`04_web_expand.py`).
- Stage 05 100-dimensional Word2Vec training and KMeans centroid clustering (`05_cluster.py`) with deterministic MD5 word seeding.
- Stage 06 analytical Markdown report compiler (`06_report.py`).
- Stage 07 standalone interactive HTML dashboard generator (`07_dashboard.py`) with XSS-safe serialization.
- Shared utilities foundation (`pipeline/common.py`).
- Automated unit and regression test suite (`tests/test_centroid.py`).
- End-to-end integration validation suite (`tests/validate_pipeline.py`).
