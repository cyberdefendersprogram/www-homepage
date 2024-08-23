# Datagen
Data generation from google spreadsheets and csv files.

Updating to 1.1.2

# Usage
``` bash
datagen % poetry run python yaml_datagen.py         
Please provide either --csv or --gsheet switch.
```

# TODO
1. Enable datagen to be run from a description file
    1.1 read yaml config file
    1.2 parse config file
    1.3 generate data based on config
  e.g. Given a CSV file, Table DDL, Transform DDL, Output format
