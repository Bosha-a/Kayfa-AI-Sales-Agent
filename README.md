# Kayfa Sales Agent 

> A Streamlit-based sales assistant for Kayfa that combines authenticated chat, CRM lead capture, and role-based access on top of MongoDB, Qdrant, Groq, and local knowledge files.

## Overview

Kayfa Agent is built for Kayfa's internal sales and support workflow. Users log in with a role, then access the chat experience, the CRM dashboard, or both depending on their permissions. The chat flow uses Kayfa's content library and vector search stack, while the CRM view surfaces captured leads and conversation summaries for follow-up.

## Features

- Role-based access control for `admin`, `sales`, and `user` accounts.
- Authenticated Streamlit chat UI with Arabic and English layout handling.
- CRM dashboard for viewing and filtering captured leads.
- MongoDB-backed storage for users, chat sessions, messages, and CRM tickets.
- Retrieval stack built with Qdrant, sentence-transformers, and fastembed.
- Branded Kayfa UI with custom styling and localized content.

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Auth | bcrypt, MongoDB |
| Chat / Agent | pydantic-ai, Groq |
| Vector Search | Qdrant, sentence-transformers, fastembed |
| Database | MongoDB |
| Data | Markdown knowledge base in `data/text` and JSON content in `data/json` |

## Getting Started

### Prerequisites

- Python 3.10 or newer
- MongoDB instance
- Qdrant instance
- Groq API key

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root with the required service credentials:

```env
MONGODB_URI=your-mongodb-connection-string
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key
GROQ_API_KEY=your-groq-api-key
```

### Run the app

```bash
streamlit run app/main.py
```

## Project Structure

```text
app/
  auth.py      # login form and role permissions
  chat.py      # authenticated chat experience
  crm.py       # CRM lead dashboard
  login.py     # legacy login/bootstrap script
  main.py      # main app shell and routing
data/
  json/        # structured data
  text/        # knowledge-base documents 
images/        # Kayfa branding assets
main.ipynb     # notebook workspace entry point
```

## Notes

- `admin` has access to both chat and CRM.
- `sales` has CRM access only.
- `user` has chat access only.
- The app expects the Kayfa branding images referenced by the code to be present in `images/`.

## License

See [LICENSE](LICENSE) for details.
