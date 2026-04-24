## Voice RAG with Gemini API

This project is a small web app that lets you "talk to your documents."

You upload one or more PDF files, ask a question in plain English, and the app:

- finds the most relevant parts of the uploaded PDFs
- writes an answer based on those document sections
- reads the answer out loud as audio

It is built with Streamlit for the interface, Qdrant for document search, FastEmbed for embeddings, and the Google GenAI SDK for Gemini text and speech generation.

### What This Project Does

If someone gives the app a PDF manual, report, or set of notes, the app can help them ask questions like:

- "What does this document say about authentication?"
- "Summarize the main setup steps"
- "Which page mentions rate limits?"

Instead of searching the whole PDF manually, the app pulls out the most relevant document snippets first, then asks Gemini to answer using those snippets. This makes the answer more grounded in the uploaded files instead of being a generic response.

The final result is:

- a text answer on screen
- a spoken audio version of that answer
- a short list of document sources used

### For People New to These Technologies

Here is what each major piece is doing in simple terms:

- `Streamlit`: creates the browser-based UI so you can upload files and ask questions without building a full frontend
- `PDF processing`: reads text out of the uploaded PDF so the app can work with it
- `Embeddings`: converts text into number-based representations so similar meaning can be searched efficiently
- `FastEmbed`: the library used to create those embeddings locally in the app
- `Qdrant`: a vector database that stores the document chunks and helps find the most relevant ones for a question
- `Gemini`: the AI model that writes the final answer and also generates the spoken audio
- `RAG (Retrieval-Augmented Generation)`: a pattern where the app retrieves useful document content first, then generates an answer from that content

If you are new to RAG, the easiest way to think about it is:

1. The app reads your document.
2. The app breaks it into smaller pieces.
3. The app stores those pieces in a searchable format.
4. When you ask a question, it finds the best matching pieces.
5. Gemini answers using those pieces as context.

### Why This Is Useful

This kind of app is helpful when you want an AI answer that stays tied to your own documents, such as:

- product documentation
- company policies
- research PDFs
- training manuals
- class notes

It is especially useful when reading the whole document would take too long or when you want both a written and spoken response.

### Features

- Upload PDF documents through a simple web app
- Extract and split PDF text into searchable chunks
- Store document embeddings in Qdrant
- Retrieve relevant chunks for a user question
- Generate a Gemini text answer grounded in retrieved content
- Generate a Gemini TTS audio response
- Download the audio as a WAV file
- Choose from multiple Gemini voices

### How It Works

1. You upload a PDF in the Streamlit app.
2. The app extracts text from the PDF.
3. That text is split into smaller chunks.
4. Each chunk is converted into an embedding.
5. The embeddings are stored in Qdrant.
6. When you ask a question, the app embeds your question too.
7. Qdrant finds the document chunks most similar to your question.
8. Gemini uses those retrieved chunks to generate an answer.
9. Gemini also converts the answer into speech and returns a WAV audio file.

### Project Flow at a Glance

`PDF -> text extraction -> chunking -> embeddings -> Qdrant search -> Gemini answer -> Gemini speech`

### Models Used

- Text model: `gemini-2.5-flash`
- Speech model: `gemini-2.5-flash-preview-tts`

Gemini TTS is a preview capability, so availability, pricing, and limits can vary by project.

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add your credentials to `.env`:

```bash
GEMINI_API_KEY='your-gemini-api-key'
QDRANT_URL='your-qdrant-url'
QDRANT_API_KEY='your-qdrant-api-key'
```

The app reads `GEMINI_API_KEY` first and also accepts `GOOGLE_API_KEY` as a fallback.

3. Run the app:

```bash
streamlit run rag_voice.py
```

4. Open the Streamlit page in your browser, upload a PDF, and ask a question.

### Example Use Case

Imagine you upload a 40-page API guide. Instead of manually scanning it, you can ask:

- "How do I authenticate requests?"
- "What are the required environment variables?"
- "Summarize the setup steps for a beginner"

The app will search the PDF, produce an answer based on the most relevant sections, and read that answer aloud.
