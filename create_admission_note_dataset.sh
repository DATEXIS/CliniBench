#!/bin/bash

# postgresql://username:password@psql_ip:psql_port/mimic-database-name
psql_connection_string="postgresql://postgres:postgres@localhost:5432/mimiciv"

python src/dataset/create_dataset.py --save-dir=./data/mimic-iv --split=hosp --icd-version=9 --psql-connection-string=$psql_connection_string
python src/dataset/create_dataset.py --save-dir=./data/mimic-iv --split=icu --icd-version=9 --psql-connection-string=$psql_connection_string
python src/dataset/create_dataset.py --save-dir=./data/mimic-iv --split=hosp --icd-version=10 --psql-connection-string=$psql_connection_string
python src/dataset/create_dataset.py --save-dir=./data/mimic-iv --split=icu --icd-version=10 --psql-connection-string=$psql_connection_string

python src/dataset/create_splits.py --save_dir=./data/preprocessed --path_to_admission_notes=./data/mimic-iv --path_to_split_ids=./data/splits --few_shot_note_save_path=data/fewshot-candidates/notes.parquet