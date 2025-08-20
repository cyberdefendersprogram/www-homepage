# Cyber Defenders Program Homepage

This is a Jekyll-based website for the Cyber Defenders Program, a cybersecurity education initiative that offers courses, workshops, hackathons, and other educational programs.

## Project Structure
- **Jekyll site**: Static site generator with Ruby/Bundler setup
- **Sass/CSS**: Uses Bulma CSS framework with custom Sass compilation
- **Data management**: Python scripts for syncing Google Sheets data to YAML files
- **GitHub Pages**: Deployed via GitHub Pages hosting

## Development Commands
- `npm start` - Start development server with live reload and CSS watching
- `npm run serve` - Jekyll serve with live reload only  
- `npm run css-build` - Compile Sass to CSS
- `npm run css-watch` - Watch and compile Sass changes
- `npm run deploy` - Build CSS for production

## Data Management
The site uses automated data sync from Google Sheets:
- `npm run gs-data` - Sync Growth Sector Python Academy schedule
- `npm run m-cis52-data` - Sync Merritt CIS52 course schedule  
- `npm run m-cis55-data` - Sync Merritt CIS55 course schedule
- `npm run m-cis60-data` - Sync Merritt CIS60 course schedule

## Content Structure
- `/pages/` - Static pages and program descriptions
- `/_data/` - YAML data files for schedules, projects, jobs
- `/_includes/` - Reusable HTML components
- `/_layouts/` - Page templates
- `/assets/` - Images, CSS, JavaScript, PDFs

## Key Features
- Course/workshop management with dynamic schedules
- Student project showcases and hackathon results
- Job board and career resources
- Educational content and cybersecurity resources
- Donation functionality

## Dependencies
- Ruby with Bundler for Jekyll
- Node.js with npm for build tools
- Python with Poetry for data management scripts
- Sass for CSS preprocessing
- Bulma CSS framework