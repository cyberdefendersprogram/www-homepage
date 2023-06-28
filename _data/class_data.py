import argparse
import gspread
import yaml, os, csv
from oauth2client.service_account import ServiceAccountCredentials

def convert_google_spreadsheet_to_yaml(spreadsheet_id = '1yN3bmLLB_KuESmNqgiDEJgchy83cVZjVzdySJJCL9-s', sheet_name='roster'):
    """ Given a spreadsheet_id and sheet_name, convert it to a yaml """
    # Replace 'credentials.json' with the path to your credentials JSON file
    HOME=os.environ.get("HOME")
    credentials = ServiceAccountCredentials.from_json_keyfile_name(HOME + "/.google/gdrive/cyberdefenders/gspread.json", ['https://www.googleapis.com/auth/spreadsheets'])
    gc = gspread.authorize(credentials)

    # Replace 'your_spreadsheet_id' with the actual ID of your Google Spreadsheet
    spreadsheet = gc.open_by_key(spreadsheet_id)

    # Replace 'Sheet1' with the name of the sheet you want to read
    sheet = spreadsheet.worksheet(sheet_name)

    # Get all values from the sheet
    data = sheet.get_all_values()
    # Extract the first row as keys
    keys = data[0]
    # Convert the remaining rows to dictionaries using the keys
    rows = [dict(zip(keys, row)) for row in data[1:]]
    # Convert the data to YAML format
    yaml_content = yaml.dump(rows, default_flow_style=False)
    return yaml_content

def convert_csv_to_yaml(csv_file="", yaml_file=""):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description='Convert CSV or Google Spreadsheet to YAML')
    parser.add_argument('--csv', action='store_true', help='Convert from CSV file')
    parser.add_argument('--spreadsheet', action='store_true', help='Convert from Google Spreadsheet')
    parser.add_argument('--input', help='Input CSV file or Google Spreadsheet ID')
    parser.add_argument('--sheet', help='Sheet name (only for Google Spreadsheet)')
    parser.add_argument('--output', help='Output YAML file')
    args = parser.parse_args()

    if args.csv and not args.spreadsheet:
        convert_csv_to_yaml(args.input, args.output)
    elif args.spreadsheet and not args.csv:
        convert_google_spreadsheet_to_yaml(args.input, args.sheet, args.output)
    else:
        print("Please provide either --csv or --spreadsheet switch.")

if __name__ == '__main__':
    main()

