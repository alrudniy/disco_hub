export DH_DATA_ROOT=/home/alex/discovery_hub/data
nohup python 01_5_filter_openalex.py --require-abstract > openalex_filter.log 2>&1 &
tail -f openalex_filter.log