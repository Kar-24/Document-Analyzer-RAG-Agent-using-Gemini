from typing import Dict, List, Optional, Tuple
import os
import tempfile
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from fastembed import TextEmbedding
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

load_dotenv(override=True)

COLLECTION_NAME = "voice-rag-agent"
TEXT_MODEL = "gemini-2.5-flash"
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TTS_SAMPLE_RATE = 24000
GEMINI_VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
]


@dataclass
class DocumentChunk:
    """Minimal document container for embedded content."""

    page_content: str
    metadata: Dict


def _clean_env_value(value: Optional[str]) -> str:
    """Normalize values loaded from widgets or .env files."""
    if not value:
        return ""
    return value.strip().strip('"').strip("'")


def load_credentials_from_env() -> Dict[str, str]:
    """Reload credentials from .env and process environment."""
    load_dotenv(override=True)
    gemini_key = _clean_env_value(
        os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    )
    return {
        "qdrant_url": _clean_env_value(os.getenv("QDRANT_URL", "")),
        "qdrant_api_key": _clean_env_value(os.getenv("QDRANT_API_KEY", "")),
        "gemini_api_key": gemini_key,
    }


def init_session_state() -> None:
    """Initialize Streamlit session state with default values."""
    env_credentials = load_credentials_from_env()
    defaults = {
        "initialized": False,
        "qdrant_url": env_credentials["qdrant_url"],
        "qdrant_api_key": env_credentials["qdrant_api_key"],
        "gemini_api_key": env_credentials["gemini_api_key"],
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "selected_voice": "Kore",
        "processed_documents": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def setup_sidebar() -> None:
    """Configure sidebar with API settings and voice options."""
    with st.sidebar:
        st.title("Configuration")
        st.markdown("---")

        if st.button("Reload .env"):
            env_credentials = load_credentials_from_env()
            st.session_state.qdrant_url = env_credentials["qdrant_url"]
            st.session_state.qdrant_api_key = env_credentials["qdrant_api_key"]
            st.session_state.gemini_api_key = env_credentials["gemini_api_key"]
            st.rerun()

        st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url,
            type="password",
            key="qdrant_url",
        )
        st.text_input(
            "Qdrant API Key",
            value=st.session_state.qdrant_api_key,
            type="password",
            key="qdrant_api_key",
        )
        st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            key="gemini_api_key",
        )

        st.markdown("---")
        st.markdown("### Gemini Voice")
        st.selectbox(
            "Select Voice",
            options=GEMINI_VOICES,
            index=GEMINI_VOICES.index(st.session_state.selected_voice),
            help="Choose the Gemini TTS voice for the audio response",
            key="selected_voice",
        )


def setup_qdrant() -> Tuple[QdrantClient, TextEmbedding]:
    """Initialize Qdrant client and embedding model."""
    qdrant_url = _clean_env_value(
        st.session_state.get("qdrant_url") or os.getenv("QDRANT_URL", "")
    )
    qdrant_api_key = _clean_env_value(
        st.session_state.get("qdrant_api_key") or os.getenv("QDRANT_API_KEY", "")
    )

    if not all([qdrant_url, qdrant_api_key]):
        raise ValueError("Qdrant credentials not provided")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embedding_model = TextEmbedding()
    test_embedding = list(embedding_model.embed(["test"]))[0]
    embedding_dim = len(test_embedding)

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
    except Exception as exc:
        if "already exists" not in str(exc):
            raise

    return client, embedding_model


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split extracted text into overlapping chunks."""
    cleaned_text = " ".join(text.split())
    if not cleaned_text:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def process_pdf(file) -> List[DocumentChunk]:
    """Extract PDF text and convert it to chunks without LangChain splitters."""
    tmp_file_path: Optional[str] = None
    try:
        from pypdf import PdfReader

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        reader = PdfReader(tmp_file_path)
        timestamp = datetime.now().isoformat()
        documents: List[DocumentChunk] = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            for chunk_index, chunk in enumerate(split_text(page_text), start=1):
                documents.append(
                    DocumentChunk(
                        page_content=chunk,
                        metadata={
                            "source_type": "pdf",
                            "file_name": file.name,
                            "page": page_number,
                            "chunk": chunk_index,
                            "timestamp": timestamp,
                        },
                    )
                )

        if not documents:
            raise ValueError("No extractable text found in the PDF")

        return documents
    except Exception as exc:
        st.error(f"PDF processing error: {exc}")
        return []
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def store_embeddings(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    documents: List[DocumentChunk],
    collection_name: str,
) -> None:
    """Store document embeddings in Qdrant."""
    for doc in documents:
        embedding = list(embedding_model.embed([doc.page_content]))[0]
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={"content": doc.page_content, **doc.metadata},
                )
            ],
        )


def mask_api_key(api_key: str) -> str:
    """Return a masked representation of the active API key."""
    if not api_key:
        return "[empty]"
    if len(api_key) <= 12:
        return "[set]"
    return f"{api_key[:6]}...{api_key[-4:]}"


def get_gemini_client(api_key: str) -> genai.Client:
    """Create a Gemini client with the provided API key."""
    if not api_key:
        raise ValueError("Gemini API key not provided")
    return genai.Client(api_key=api_key)


def build_tts_prompt(text_response: str) -> str:
    """Construct a prompt for Gemini native TTS."""
    return (
        "Read the following answer exactly as written in a professional, clear, "
        "friendly voice with a steady pace.\n\n"
        f"{text_response}"
    )


def save_wave_file(
    filename: str, pcm_data: bytes, channels: int = 1, rate: int = TTS_SAMPLE_RATE, sample_width: int = 2
) -> None:
    """Save raw PCM audio data to a WAV file."""
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm_data)


def generate_text_response(client: genai.Client, context: str) -> str:
    """Generate a text answer from Gemini."""
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=context,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are a helpful documentation assistant. Answer clearly and concisely, "
                "use the provided context only, mention source file names when relevant, "
                "and format the response so it is easy to read aloud."
            ),
            max_output_tokens=700,
        ),
    )
    text_response = (response.text or "").strip()
    if not text_response:
        raise ValueError("Gemini returned an empty text response.")
    return text_response


def generate_audio_response(client: genai.Client, text_response: str, voice: str) -> str:
    """Generate audio from Gemini TTS and save it to a WAV file."""
    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=build_tts_prompt(text_response),
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )

    audio_data = response.candidates[0].content.parts[0].inline_data.data
    if not audio_data:
        raise ValueError("Gemini returned an empty audio response.")

    audio_path = os.path.join(tempfile.gettempdir(), f"response_{uuid.uuid4()}.wav")
    save_wave_file(audio_path, audio_data)
    return audio_path


def format_gemini_error(exc: genai_errors.APIError, api_key: str, stage: str) -> str:
    """Convert Gemini SDK errors into clearer UI messages."""
    masked_key = mask_api_key(api_key)

    if exc.code == 429:
        return (
            f"Gemini rejected the current key {masked_key} during {stage} because the project "
            "or account has hit a rate or quota limit. Check usage limits in Google AI Studio."
        )

    if exc.code in {400, 401, 403}:
        return (
            f"Gemini rejected the current key {masked_key} during {stage}. "
            f"Status: {exc.status or 'unknown'}. Message: {exc.message or exc.details}"
        )

    return (
        f"Gemini API error during {stage} with key {masked_key}: "
        f"{exc.message or exc.details}"
    )


def process_query(
    query: str,
    client: QdrantClient,
    embedding_model: TextEmbedding,
    collection_name: str,
    gemini_api_key: str,
    voice: str,
) -> Dict:
    """Process the user query with Gemini and generate an audio response."""
    try:
        gemini_client = get_gemini_client(gemini_api_key)

        st.info("Step 1: Generating query embedding and searching documents...")
        query_embedding = list(embedding_model.embed([query]))[0]
        st.write(f"Generated embedding of size: {len(query_embedding)}")

        search_response = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=3,
            with_payload=True,
        )

        search_results = search_response.points if hasattr(search_response, "points") else []
        st.write(f"Found {len(search_results)} relevant documents")

        if not search_results:
            raise ValueError("No relevant documents found in the vector database")

        st.info("Step 2: Preparing context from search results...")
        context = "Use the following documentation excerpts to answer the user's question.\n\n"
        for i, result in enumerate(search_results, start=1):
            payload = result.payload
            if not payload:
                continue
            content = payload.get("content", "")
            source = payload.get("file_name", "Unknown Source")
            page = payload.get("page")
            source_label = f"{source} (page {page})" if page else source
            context += f"Source: {source_label}\n{content}\n\n"
            st.write(f"Document {i} from: {source_label}")

        context += f"User question: {query}"

        st.info("Step 3: Preparing Gemini request...")
        st.write(f"Using Gemini key: {mask_api_key(gemini_api_key)}")
        st.write(f"Text model: {TEXT_MODEL}")
        st.write(f"TTS model: {TTS_MODEL}")

        st.info("Step 4: Generating text response...")
        try:
            text_response = generate_text_response(gemini_client, context)
        except genai_errors.APIError as exc:
            raise RuntimeError(format_gemini_error(exc, gemini_api_key, "text generation")) from exc
        st.write(f"Generated text response of length: {len(text_response)}")

        st.info("Step 5: Generating audio response...")
        try:
            audio_path = generate_audio_response(gemini_client, text_response, voice)
        except genai_errors.APIError as exc:
            raise RuntimeError(format_gemini_error(exc, gemini_api_key, "speech generation")) from exc
        st.write(f"Saved WAV file to: {audio_path}")

        st.success("Query processing complete.")
        return {
            "status": "success",
            "text_response": text_response,
            "audio_path": audio_path,
            "sources": [
                r.payload.get("file_name", "Unknown Source")
                for r in search_results
                if r.payload
            ],
        }

    except Exception as exc:
        st.error(f"Error during query processing: {exc}")
        return {
            "status": "error",
            "error": str(exc),
            "query": query,
        }


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Voice RAG Agent with Gemini",
        page_icon="microphone",
        layout="wide",
    )

    init_session_state()
    setup_sidebar()

    st.title("Voice RAG Agent with Gemini")
    st.info(
        "Upload PDF documentation, retrieve relevant chunks from Qdrant, and get a Gemini-powered "
        "answer with downloadable speech."
    )

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        file_name = uploaded_file.name
        if file_name not in st.session_state.processed_documents:
            with st.spinner("Processing PDF..."):
                try:
                    if not st.session_state.client:
                        client, embedding_model = setup_qdrant()
                        st.session_state.client = client
                        st.session_state.embedding_model = embedding_model

                    documents = process_pdf(uploaded_file)
                    if documents:
                        store_embeddings(
                            st.session_state.client,
                            st.session_state.embedding_model,
                            documents,
                            COLLECTION_NAME,
                        )
                        st.session_state.processed_documents.append(file_name)
                        st.session_state.setup_complete = True
                        st.success(f"Added PDF: {file_name}")
                except Exception as exc:
                    st.error(f"Error processing document: {exc}")

    if st.session_state.processed_documents:
        st.sidebar.header("Processed Documents")
        for doc in st.session_state.processed_documents:
            st.sidebar.text(doc)

    query = st.text_input(
        "What would you like to know about the documentation?",
        placeholder="e.g., How do I authenticate API requests?",
        disabled=not st.session_state.setup_complete,
        key="query_input",
    )

    if query and st.session_state.setup_complete:
        with st.status("Processing your query...", expanded=True) as status:
            try:
                result = process_query(
                    query,
                    st.session_state.client,
                    st.session_state.embedding_model,
                    COLLECTION_NAME,
                    st.session_state.gemini_api_key,
                    st.session_state.selected_voice,
                )

                if result["status"] == "success":
                    status.update(label="Query processed.", state="complete")

                    st.markdown("### Response")
                    st.write(result["text_response"])

                    if "audio_path" in result:
                        st.markdown(
                            f"### Audio Response (Voice: {st.session_state.selected_voice})"
                        )
                        st.audio(result["audio_path"], format="audio/wav", start_time=0)

                        with open(result["audio_path"], "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.download_button(
                                label="Download Audio Response",
                                data=audio_bytes,
                                file_name=(
                                    f"voice_response_{st.session_state.selected_voice}.wav"
                                ),
                                mime="audio/wav",
                            )

                    st.markdown("### Sources")
                    for source in result["sources"]:
                        st.markdown(f"- {source}")
                else:
                    status.update(label="Error processing query", state="error")
                    st.error(f"Error: {result.get('error', 'Unknown error occurred')}")

            except Exception as exc:
                status.update(label="Error processing query", state="error")
                st.error(f"Error processing query: {exc}")

    elif not st.session_state.setup_complete:
        st.info("Please configure the system and upload documents first.")

    if not st.session_state.gemini_api_key:
        st.warning("Gemini API key is empty. Add `GEMINI_API_KEY` to `.env` or paste it in the sidebar.")


if __name__ == "__main__":
    main()
