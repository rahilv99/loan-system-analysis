# Bank Statement Insights
https://www.loom.com/share/75ad6812a5b647d3b0d5ed6138c80e48?sid=e4c88389-bcca-4328-91d1-3c8e147755f8
## Overview
A full-stack application for analyzing bank statements using machine learning and data visualization.

## Tech Stack
- Backend: Django (Python)
- Frontend: Next.js with React and Tailwind CSS
- Data Visualization: Chart.js
- ML Insights: Custom Python data analysis

## Setup Instructions

### Backend Setup
1. Navigate to `backend` directory
2. Create virtual environment:
   ```
   python -m venv backend_venv
   backend_venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run migrations:
   ```
   python manage.py migrate
   ```
5. Start server:
   ```
   python manage.py runserver
   ```

### Frontend Setup
1. Navigate to `frontend` directory
2. Install dependencies:
   ```
   npm install
   ```
3. Run development server:
   ```
   npm run dev
   ```

## Features
- PDF Bank Statement Upload
- Expense Category Analysis
- Monthly Spending Trends
- Income vs Expense Visualization

## License
MIT License
