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


front_matter = []

# Iterate over each row of data
for row in data:
    print(row)
    # Selectively add data to the front matter based on your conditions
    if row[0] == 'Title':
        front_matter.append(f'title: {row[1]}')
    elif row[0] == 'Author':
        front_matter.append(f'author: {row[1]}')
    elif row[0] == 'Date':
        front_matter.append(f'date: {row[1]}')
    # Add additional conditions for other fields as needed

# Create the final YAML front matter string
yaml_front_matter = '---\n' + '\n'.join(front_matter) + '\n---\n'


