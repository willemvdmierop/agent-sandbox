# VME Document Question Answering System

An AI-powered document question answering system specifically designed for Belgian property management (VME - Vereniging van Mede-Eigenaars). The system processes property management documents and provides accurate, legally-grounded answers to questions about property management, co-ownership, and related matters.

## Features

- Recursive document processing with support for nested folder structures
- Intelligent document chunking and vector storage
- Language detection and response in the user's preferred language
- Comprehensive answer generation with legal citations and sources
- Structured responses following Belgian property management best practices

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd vme-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'  # On Windows: set OPENAI_API_KEY=your-api-key
```

## Usage

1. Place your VME documents (PDFs) in a folder
2. Run the agent:
```bash
python vme_agent.py
```

3. Ask questions about your VME documents in your preferred language

## Project Structure

- `vme_agent.py`: Main agent implementation
- `Processdocument.py`: Document processing utilities
- `vectorstores/`: Directory for storing document embeddings (gitignored)
- `requirements.txt`: Project dependencies

## Response Format

The system provides structured responses including:
- Clear decision (Allowed/Not Allowed/Requires AV Vote)
- Action plan with steps and deadlines
- Cost allocation and reimbursement method
- Analysis of authority and permissions
- Source citations with exact quotes
- Legal disclaimer

## License

[Your chosen license]

## Disclaimer

This system provides practical guidance based on Belgian property management documents and regulations. For binding legal advice, always consult a notary or lawyer. 