// API Base URL
const API_BASE = (window.location && window.location.protocol.startsWith('http') && window.location.origin)
    ? window.location.origin
    : 'http://127.0.0.1:8000';

// File handling
let selectedFiles = [];

document.getElementById('fileInput').addEventListener('change', function(e) {
    selectedFiles = Array.from(e.target.files);
    updateFileList();
    document.getElementById('uploadBtn').disabled = selectedFiles.length === 0;
});

// Drag and drop
const uploadArea = document.getElementById('uploadArea');
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = '#f0f2ff';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.background = '';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = '';
    const files = Array.from(e.dataTransfer.files).filter(f => 
        f.name.endsWith('.pdf') || f.name.endsWith('.txt')
    );
    selectedFiles = files;
    updateFileList();
    document.getElementById('uploadBtn').disabled = selectedFiles.length === 0;
});

function updateFileList() {
    const fileList = document.getElementById('fileList');
    if (selectedFiles.length === 0) {
        fileList.innerHTML = '';
        return;
    }
    
    fileList.innerHTML = selectedFiles.map(file => `
        <div class="file-item">
            <span>📄 ${file.name}</span>
            <span style="color: #666; font-size: 0.9em;">${(file.size / 1024).toFixed(2)} KB</span>
        </div>
    `).join('');
}

// Upload files
async function uploadFiles() {
    if (selectedFiles.length === 0) return;
    
    const uploadBtn = document.getElementById('uploadBtn');
    const statusDiv = document.getElementById('uploadStatus');
    
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<span class="loading"></span>Uploading...';
    statusDiv.className = 'status-message info';
    statusDiv.textContent = 'Uploading files...';
    statusDiv.style.display = 'block';
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });
    
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const data = await response.json();
        statusDiv.className = 'status-message success';
        statusDiv.textContent = `✓ Successfully uploaded ${data.document_ids.length} document(s)`;
        
        // Enable question input
        document.getElementById('askBtn').disabled = false;
        document.getElementById('questionInput').disabled = false;
        
        // Clear selected files
        selectedFiles = [];
        document.getElementById('fileInput').value = '';
        updateFileList();
        
    } catch (error) {
        statusDiv.className = 'status-message error';
        statusDiv.textContent = `✗ Error: ${error.message}`;
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload Documents';
    }
}

// Ask question
async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();
    
    if (!question) {
        showStatus('questionStatus', 'Please enter a question', 'error');
        return;
    }
    
    const askBtn = document.getElementById('askBtn');
    const statusDiv = document.getElementById('questionStatus');
    const resultsSection = document.getElementById('resultsSection');
    
    askBtn.disabled = true;
    askBtn.innerHTML = '<span class="loading"></span>Processing...';
    statusDiv.className = 'status-message info';
    statusDiv.textContent = 'Generating answer...';
    statusDiv.style.display = 'block';
    resultsSection.style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to get answer');
        }
        
        const data = await response.json();
        displayResults(data);
        
        statusDiv.className = 'status-message success';
        statusDiv.textContent = '✓ Answer generated successfully';
        
    } catch (error) {
        statusDiv.className = 'status-message error';
        statusDiv.textContent = `✗ Error: ${error.message}`;
        resultsSection.style.display = 'none';
    } finally {
        askBtn.disabled = false;
        askBtn.textContent = 'Ask Question';
    }
}

// Display results
function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Display answer
    const answerText = document.getElementById('answerText');
    answerText.innerHTML = formatAnswer(data.answer, data.evidence_alignment, data.unsupported_sentences);
    
    // Display evidence alignment
    const evidenceDiv = document.getElementById('evidenceAlignment');
    if (data.evidence_alignment && data.evidence_alignment.length > 0) {
        evidenceDiv.innerHTML = data.evidence_alignment.map((item, idx) => {
            const isUnsupported = item.supporting_chunks.length === 0;
            return `
                <div class="evidence-item ${isUnsupported ? 'unsupported' : ''}">
                    <div class="sentence-text">${escapeHtml(item.answer_sentence)}</div>
                    <div class="supporting-chunks">
                        ${isUnsupported 
                            ? '<strong>⚠️ No supporting chunks found</strong>' 
                            : `<strong>Supported by chunks:</strong> ${item.supporting_chunks.join(', ')}`
                        }
                    </div>
                </div>
            `;
        }).join('');
    } else {
        evidenceDiv.innerHTML = '<p>No evidence alignment data available.</p>';
    }
    
    // Display unsupported sentences
    const unsupportedBlock = document.getElementById('unsupportedBlock');
    const unsupportedDiv = document.getElementById('unsupportedSentences');
    if (data.unsupported_sentences && data.unsupported_sentences.length > 0) {
        unsupportedBlock.style.display = 'block';
        unsupportedDiv.innerHTML = data.unsupported_sentences.map(sentence => `
            <div class="unsupported-item">${escapeHtml(sentence)}</div>
        `).join('');
    } else {
        unsupportedBlock.style.display = 'none';
    }
    
    // Display retrieved chunks
    const chunksDiv = document.getElementById('retrievedChunks');
    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
        chunksDiv.innerHTML = data.retrieved_chunks.map((chunk, idx) => `
            <div class="chunk-item">
                <div class="chunk-header">
                    <span class="chunk-id">Chunk ${idx + 1}: ${chunk.chunk_id}</span>
                    <span class="chunk-score">Score: ${chunk.score.toFixed(4)}</span>
                </div>
                <div class="chunk-text">${escapeHtml(chunk.text)}</div>
            </div>
        `).join('');
    } else {
        chunksDiv.innerHTML = '<p>No chunks retrieved.</p>';
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Format answer with highlighting
function formatAnswer(answer, evidenceAlignment, unsupportedSentences) {
    if (!evidenceAlignment || evidenceAlignment.length === 0) {
        return `<p>${escapeHtml(answer)}</p>`;
    }
    
    // Create a map of unsupported sentences for quick lookup
    const unsupportedSet = new Set(unsupportedSentences || []);
    
    // Split answer into sentences and format
    const sentences = answer.split(/([.!?]+\s+)/);
    let formatted = '';
    
    for (let i = 0; i < sentences.length; i += 2) {
        if (i + 1 < sentences.length) {
            const sentence = sentences[i] + sentences[i + 1];
            const isUnsupported = unsupportedSet.has(sentence.trim());
            formatted += `<div class="sentence ${isUnsupported ? 'unsupported' : ''}">${escapeHtml(sentence)}</div>`;
        } else if (sentences[i].trim()) {
            const sentence = sentences[i];
            const isUnsupported = unsupportedSet.has(sentence.trim());
            formatted += `<div class="sentence ${isUnsupported ? 'unsupported' : ''}">${escapeHtml(sentence)}</div>`;
        }
    }
    
    return formatted || `<p>${escapeHtml(answer)}</p>`;
}

// Utility functions
function showStatus(elementId, message, type) {
    const statusDiv = document.getElementById(elementId);
    statusDiv.className = `status-message ${type}`;
    statusDiv.textContent = message;
    statusDiv.style.display = 'block';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Allow Enter key to submit question (Ctrl+Enter)
document.getElementById('questionInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.ctrlKey) {
        askQuestion();
    }
});
