import gspread
from oauth2client.service_account import ServiceAccountCredentials
import yaml, os

# Replace 'credentials.json' with the path to your credentials JSON file
HOME=os.environ.get("HOME")
print(HOME)
credentials = ServiceAccountCredentials.from_json_keyfile_name(HOME + "/.google/gdrive/cyberdefenders/gspread.json", ['https://www.googleapis.com/auth/spreadsheets'])
gc = gspread.authorize(credentials)

# Replace 'your_spreadsheet_id' with the actual ID of your Google Spreadsheet
spreadsheet_id = '1yN3bmLLB_KuESmNqgiDEJgchy83cVZjVzdySJJCL9-s'
spreadsheet = gc.open_by_key(spreadsheet_id)

# Replace 'Sheet1' with the name of the sheet you want to read
sheet_name = 'roster'
sheet = spreadsheet.worksheet(sheet_name)

# Get all values from the sheet
data = sheet.get_all_values()
# Extract the first row as keys
keys = data[0]
# Convert the remaining rows to dictionaries using the keys
rows = [dict(zip(keys, row)) for row in data[1:]]
# Convert the data to YAML format
yaml_content = yaml.dump(rows, default_flow_style=False)
print(yaml_content)

# Create the final YAML front matter string
yaml_front_matter = '---\n' + '\n'.join(yaml_content) + '\n---\n'
