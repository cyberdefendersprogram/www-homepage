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

## Git Commit Guidelines
- Keep commit messages concise and descriptive
- Do not mention Claude Code or AI assistance in commit messages
- Follow existing commit message style (simple, lowercase descriptions)

## Workflows

### post-job
Add a new job listing to `_data/jobs.yml`:
1. Get the job URL from the user
2. Use WebFetch to extract: title, company, description, and key requirements
3. Ask who recommended the job (by field)
4. Add entry to `_data/jobs.yml` with this format:
   ```yaml
   - title: "Job Title"
     link: <job-url>
     description: "Brief description including requirements and application deadline if available"
     company: Company Name
     tags:
       - relevant-tag
     date: <today's date YYYY-MM-DD>
     by: <recommender name>
   ```
5. Use today's date for the `date` field, not the application deadline
6. Include application deadline in the description if available