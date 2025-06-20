📄 pdfreader

A lightweight web-based PDF viewer and reader built with Python, Flask, and Docker.

Features
	•	Upload and display PDF files directly in your browser.
	•	Navigate pages with a simple web UI.
	•	Supports full PDF-1.7 spec including text, images, fonts, and annotations  ￼.
	•	Lazy-loading for fast performance and low memory use.

📦 Tech Stack
	•	Back-end: Python + Flask
	•	PDF Parsing: pdfreader (Pythonic API, spec-compliant)  ￼
	•	Containerization: Docker, docker-compose
	•	Front-end: HTML, CSS, JavaScript
You can open the app doing 
cd pdfreader
pip install -r requirements.txt
python app.py
Access at http://localhost:5000.

🎛️ How It Works
	•	Navigate to the upload form and add a .pdf file.
	•	The file is parsed on the server using pdfreader.
	•	Pages are rendered dynamically and sent to the client.
	•	Navigate through thumbnails and actual pages for a smooth experience.
