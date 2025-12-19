```# Create and navigate to backend directory
mkdir bb84-backend && cd bb84-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn pydantic

# Create main.py with the FastAPI backend code

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000```


## Project Structure
```├── bb84-backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile          # Backend container```
