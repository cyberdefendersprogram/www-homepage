require 'yaml'
require 'net/http'
require 'uri'

files_array = YAML.load(File.read("_files.yml"))
for file in files_array do
    spreadsheet_url = "https://docs.google.com/spreadsheets/d/#{file["id"]}/export?format=csv&id=#{file["id"]}&gid=0"
    uri = URI.parse(spreadsheet_url)
    response = Net::HTTP.get_response(uri)
    file_name = file["name"]
    print("Writing file #{file["name"]} \n")
    File.write(file_name, response.body, mode: "w")
end
