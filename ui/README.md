# RAG Debugger UI

Simple web interface for the RAG Debugger backend.

## Features

- **Document Upload**: Drag and drop or click to upload PDF/TXT files
- **Question Interface**: Ask questions about uploaded documents
- **Visual Results**: 
  - Answer with sentence-level highlighting
  - Evidence alignment showing which chunks support each sentence
  - Unsupported sentences flagged with warnings
  - Retrieved chunks with scores

## Usage

1. Make sure the backend server is running on `http://127.0.0.1:8000`
2. Open `index.html` in your web browser
3. Upload documents using the upload area
4. Ask questions and view the results

## File Structure

- `index.html` - Main HTML structure
- `style.css` - Styling and layout
- `script.js` - JavaScript functionality and API calls

## Browser Compatibility

Works in all modern browsers (Chrome, Firefox, Safari, Edge).

## API Endpoints Used

- `POST /upload` - Upload documents
- `POST /ask` - Ask questions and get answers
