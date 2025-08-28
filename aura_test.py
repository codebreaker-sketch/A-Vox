import os
import requests
import time
import google.generativeai as genai
import streamlit as st
import re
from io import BytesIO
from difflib import SequenceMatcher
import sqlite3

# Try to import pydub for audio clipping
PYDUB_AVAILABLE = True
try:
    from pydub import AudioSegment
except Exception:
    PYDUB_AVAILABLE = False

# -------------------------------
# ASSEMBLYAI CONFIGURATION
# -------------------------------
ASSEMBLYAI_API_KEY = "b968b0d0ad9d4c88a87316567c6ca1db"
ASSEMBLYAI_API_URL = "https://api.assemblyai.com/v2"
headers = {
    "authorization": ASSEMBLYAI_API_KEY,
    "content-type": "application/json"
}

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("vox.db")
    cursor = conn.cursor()
    
    # Create table with only the desired columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            summary TEXT,
            dialogue TEXT,
            trendy_content TEXT,
            key_moments TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Check existing columns
    cursor.execute("PRAGMA table_info(transcripts)")
    columns = [info[1] for info in cursor.fetchall()]
    
    # Add new columns if they don't exist
    if 'trendy_content' not in columns:
        cursor.execute("ALTER TABLE transcripts ADD COLUMN trendy_content TEXT")
    if 'key_moments' not in columns:
        cursor.execute("ALTER TABLE transcripts ADD COLUMN key_moments TEXT")
    
    # Set default values for new columns if NULL
    cursor.execute("UPDATE transcripts SET trendy_content = 'No trendy content identified.' WHERE trendy_content IS NULL")
    cursor.execute("UPDATE transcripts SET key_moments = 'No key moments identified.' WHERE key_moments IS NULL")
    
    conn.commit()
    conn.close()

def save_to_db(file_name, summary, dialogue, trendy_content, key_moments):
    # Ensure trendy_content and key_moments are properly formatted as markdown lists
    def format_as_markdown_list(text):
        if text.strip() == "No trendy content identified." or text.strip() == "No key moments identified.":
            return text
        # Split by newlines and clean up
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Ensure each line starts with a bullet point
        cleaned_lines = [re.sub(r'^(\-|\*|•|\d+\.)\s*', '', line).strip() for line in lines]
        formatted_lines = [f"* {line}" for line in cleaned_lines]
        return "\n".join(formatted_lines)
    
    formatted_trendy_content = format_as_markdown_list(trendy_content)
    formatted_key_moments = format_as_markdown_list(key_moments)
    
    conn = sqlite3.connect("vox.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transcripts (file_name, summary, dialogue, trendy_content, key_moments)
        VALUES (?, ?, ?, ?, ?)
    """, (file_name, summary, dialogue, formatted_trendy_content, formatted_key_moments))
    conn.commit()
    conn.close()

init_db()

def fetch_all_transcripts():
    conn = sqlite3.connect("vox.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, file_name, summary, dialogue, trendy_content, key_moments, created_at FROM transcripts ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def export_transcripts_to_text():
    """Export all transcripts to a clean text format."""
    transcripts = fetch_all_transcripts()
    output = []
    for tid, fname, summary, dialogue, trendy_content, key_moments, created in transcripts:
        output.append(f"Transcript ID: {tid}")
        output.append(f"File Name: {fname}")
        output.append(f"Created At: {created}")
        output.append("\n## Summary")
        output.append(summary)
        output.append("\n## Dialogue")
        output.append(dialogue)
        output.append("\n## Trendy Content")
        output.append(trendy_content)
        output.append("\n## Key Moments")
        output.append(key_moments)
        output.append("\n" + "="*50 + "\n")
    return "\n".join(output)

def upload_audio(audio_data, filename):
    """Upload audio bytes to AssemblyAI and get public URL."""
    response = requests.post(
        f"{ASSEMBLYAI_API_URL}/upload",
        headers={"authorization": ASSEMBLYAI_API_KEY},
        data=audio_data
    )
    response.raise_for_status()
    return response.json()["upload_url"]

def transcribe_audio_assemblyai(audio_url):
    """Transcribe audio with automatic language detection using AssemblyAI API."""
    payload = {
        "audio_url": audio_url,
        "language_detection": True,
    }
    response = requests.post(
        f"{ASSEMBLYAI_API_URL}/transcript",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.json()["id"]

def get_transcription_result(transcript_id):
    """Retrieve transcription result from AssemblyAI."""
    while True:
        response = requests.get(
            f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] == "completed":
            return data
        elif data["status"] == "failed":
            raise Exception("Transcription failed.")
        time.sleep(5)

def diarize_audio_assemblyai(audio_url):
    """Perform speaker diarization using AssemblyAI API."""
    payload = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "language_detection": True
    }
    response = requests.post(
        f"{ASSEMBLYAI_API_URL}/transcript",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.json()["id"]

def get_diarization_result(transcript_id):
    """Retrieve speaker-wise diarization result from AssemblyAI."""
    while True:
        response = requests.get(
            f"{ASSEMBLYAI_API_URL}/transcript/{transcript_id}",
            headers=headers
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] == "completed":
            return data.get("utterances", [])
        elif data["status"] == "failed":
            raise Exception("Diarization failed.")
        time.sleep(5)

# -------------------------------
# GEMINI CONFIGURATION
# -------------------------------
def summarize_text_gemini(text):
    """Summarize text using Google Gemini API with structured output."""
    genai.configure(api_key="AIzaSyAULn49Ly7XqebdH1C7iri1po2ShQFMFO8")
    model = genai.GenerativeModel("gemini-2.5-flash")
    # prompt = (
    #     "Summarize the following transcript in three distinct sections: "
    #     "1. *Summary*: Provide a concise overview of the main topics discussed, in 3-5 sentences. "
    #     "2. *Trendy Content*: Identify the most exciting, viral, or shareable moments suitable for social media or highlight reels, "
    #     "such as dramatic plays, crowd reactions, or standout performances. Each moment must include the exact timestamp range "
    #     "(e.g., [A 2.24s - 15.60s]) from the transcript to pinpoint the segment for audio clipping. "
    #     "Format as a bullet list with a brief description of each moment. Limit to 3-5 moments for brevity. "
    #     "3. *Key Moments*: List critical points or decisions made during the conversation, such as wickets, catches, or match outcomes, "
    #     "formatted as a bullet list. "
    #     "Use clear headings for each section (## Summary, ## Trendy Content, ## Key Moments). "
    #     "If a section is not applicable, state 'No relevant content identified.'\n\n"
    #     f"Transcript:\n{text}"
    # )
    prompt = (
        "Summarize the following transcript in three distinct sections: "
        "1. **Summary**: Provide a concise overview of the main topics discussed. "
        "2. **Trendy Content**: Highlight any viral or engaging moments suitable for sharing. "
        "   - Include exact quoted text from the transcript for each moment. "
        "   - Provide an approximate timestamp for each moment (e.g., 'around 10s'). "
        "3. **Key Moments**: List critical points or decisions made during the conversation. "
        "   - Include exact quoted text from the transcript for each point. "
        "   - Provide an approximate timestamp for each point (e.g., 'around 20s'). "
        "Use clear headings (## Summary, ## Trendy Content, ## Key Moments). "
        "Format Trendy Content and Key Moments as bullet points with quoted text in quotation marks. "
        "If a section is not applicable, state 'No relevant content identified.'\n\n"
        f"Transcript:\n{text}"
    )
    response = model.generate_content(prompt)
    return response.text

def chat_with_gemini(question, context, history=None):
    """Generate a conversational response with memory using Gemini."""
    genai.configure(api_key="AIzaSyCL2IqaNvldvJ-960RM-BrIQEr5npq8dkA")
    model = genai.GenerativeModel("gemini-2.0-flash")

    history_text = ""
    if history:
        for turn in history:
            history_text += f"User: {turn['question']['content']}\nAssistant: {turn['answer']['content']}\n"

    prompt = (
        "You are a helpful chatbot answering questions based on the following audio transcription and summary. "
        "Maintain context across the conversation (multi-turn). If the user asks unrelated things, "
        "politely inform them it's outside the scope.\n\n"
        f"*Transcription*:\n{context['transcription']}\n\n"
        f"*Summary*:\n{context['summary']}\n\n"
        f"*Conversation History*:\n{history_text}\n"
        f"*User Question*: {question}"
    )

    response = model.generate_content(prompt)
    return response.text

# -------------------------------
# HELPERS FOR TRENDY TIMESTAMPS & CLIPS
# -------------------------------
def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _token_set(s: str) -> set:
    return set(re.findall(r"[a-z0-9']+", s))

def _similarity_score(query_norm: str, window_norm: str) -> float:
    q_tokens = _token_set(query_norm)
    w_tokens = _token_set(window_norm)
    if not q_tokens or not w_tokens:
        jaccard = 0.0
    else:
        inter = len(q_tokens & w_tokens)
        union = len(q_tokens | w_tokens)
        jaccard = inter / max(1, union)
    ratio = SequenceMatcher(None, query_norm, window_norm).ratio()
    return 0.65 * jaccard + 0.35 * ratio

def _parse_bracket_timestamp(item_text):
    """
    Look for a timestamp inside brackets like:
      [A 13:55 - 14:25]
      [13:55 - 14:25]
      (13:55 - 14:25)
      0:13:55 - 0:14:25
    Returns (start_ms, end_ms) or None.
    """
    m = re.search(r"\[?\(?[A-Za-z]?\s*([0-9]{1,2}(?::[0-9]{2}){1,2})\s*[-–—]\s*([0-9]{1,2}(?::[0-9]{2}){1,2})\s*\)?\]?", item_text)
    if not m:
        return None
    def to_ms(t):
        parts = [int(x) for x in t.split(":")]
        if len(parts) == 3:
            h, mm, ss = parts
        elif len(parts) == 2:
            h = 0
            mm, ss = parts
        else:
            return None
        return ((h * 3600) + (mm * 60) + ss) * 1000
    try:
        start_ms = to_ms(m.group(1))
        end_ms = to_ms(m.group(2))
        if start_ms is None or end_ms is None:
            return None
        if end_ms <= start_ms:
            return None
        return int(start_ms), int(end_ms)
    except Exception:
        return None

def find_trendy_span_ms(diarization_data, item_text, mode="loose", allow_bracket_fallback=True):
    """
    Attempts to find a clip span for item_text using several strategies in order:
      1) Parse bracketed timestamps in the item_text (if provided and allow_bracket_fallback==True)
      2) Exact/loose transcript matching (sliding windows)
      3) As a last resort (only in 'loose' mode) return the best single utterance expanded to min length
    """
    if not diarization_data or not item_text or not item_text.strip():
        return None

    if allow_bracket_fallback:
        br = _parse_bracket_timestamp(item_text)
        if br:
            return br

    item_norm = _normalize_text(item_text)
    if not item_norm:
        return None

    MAX_WINDOW_UTTS = 8
    MIN_SCORE = 0.28 if mode == "loose" else 0.45
    MIN_CLIP_MS = 2000
    MAX_CLIP_MS = 25000

    n = len(diarization_data)
    norm_texts = [_normalize_text(seg.get("text", "")) for seg in diarization_data]

    best = None
    for i in range(n):
        if not norm_texts[i]:
            continue
        s = diarization_data[i].get("start", 0) or 0
        end_time = s
        concat = []
        for j in range(i, min(n, i + MAX_WINDOW_UTTS)):
            seg = diarization_data[j]
            seg_norm = norm_texts[j]
            if seg_norm:
                concat.append(seg_norm)
            end_time = seg.get("end", end_time) or end_time
            duration = end_time - s
            if duration > 30_000:
                break
            window_text = " ".join(concat)
            score = _similarity_score(item_norm, window_text)
            if best is None or score > best[0]:
                best = (score, int(s), int(end_time))

    if best and best[0] >= MIN_SCORE:
        score, s, e = best
        dur = e - s
        if dur < MIN_CLIP_MS:
            e = s + MIN_CLIP_MS
        if e - s > MAX_CLIP_MS:
            mid = s + dur // 2
            s = max(0, mid - MAX_CLIP_MS // 2)
            e = s + MAX_CLIP_MS
        return int(s), int(e)

    fragments = [f.strip() for f in re.split(r"[.,;:!?]\s*", item_text) if f.strip()]
    for frag in fragments:
        frag_norm = _normalize_text(frag)
        if not frag_norm:
            continue
        best_frag = None
        for i in range(n):
            if not norm_texts[i]:
                continue
            s = diarization_data[i].get("start", 0) or 0
            end_time = diarization_data[i].get("end", s) or s
            window_text = norm_texts[i]
            sc = _similarity_score(frag_norm, window_text)
            if best_frag is None or sc > best_frag[0]:
                best_frag = (sc, int(s), int(end_time))
        if best_frag and best_frag[0] >= (MIN_SCORE * 0.9):
            sc, s, e = best_frag
            if e - s < MIN_CLIP_MS:
                e = s + MIN_CLIP_MS
            return int(s), int(e)

    if mode == "loose":
        best_single = None
        for i, seg in enumerate(diarization_data):
            seg_norm = norm_texts[i]
            if not seg_norm:
                continue
            sc = _similarity_score(item_norm, seg_norm)
            s = seg.get("start", 0) or 0
            e = seg.get("end", s) or s
            if best_single is None or sc > best_single[0]:
                best_single = (sc, int(s), int(e))
        if best_single and best_single[0] > 0:
            sc, s, e = best_single
            if sc >= 0.12:
                if e - s < MIN_CLIP_MS:
                    e = s + MIN_CLIP_MS
                return int(s), int(e)

    return None

def ms_to_mmss(ms: int) -> str:
    s = max(0, int(ms // 1000))
    return f"{s//60:02d}:{s%60:02d}"

def make_clip_bytes(audio_bytes: bytes, start_ms: int, end_ms: int, out_format="mp3") -> bytes:
    if not PYDUB_AVAILABLE:
        return b""
    audio = AudioSegment.from_file(BytesIO(audio_bytes))
    total = len(audio)
    s = max(0, min(start_ms, total))
    e = max(0, min(end_ms, total))
    if e <= s:
        e = min(total, s + 2000)
    PAD_MS = 150
    s = max(0, s - PAD_MS)
    e = min(total, e + PAD_MS)
    clip = audio[s:e]
    try:
        clip = clip.fade_in(50).fade_out(50)
    except Exception:
        pass
    buf = BytesIO()
    clip.export(buf, format=out_format)
    return buf.getvalue()

# -------------------------------
# STREAMLIT APP
# -------------------------------
st.set_page_config(
    page_title="AURA VOX - Audio AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown("## ⚙️ Matching Options")
    mode = st.radio("Matching mode", options=["loose", "strict"], index=0, help="loose = tolerant matching + fallbacks; strict = high precision")
    allow_bracket = st.checkbox("Allow bracket-timestamp fallback (use provided [A 13:55 - 14:25] if present)", value=True)
    st.divider()

with st.container():
    header_col1, header_col2 = st.columns([9,1])
    hero_html = """
    <div class="hero-box">
      <div class="hero-title">✨ AURA VOX</div>
      <div class="hero-sub">🎙️ Audio Transcription, Diarization & AI Insights</div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("Upload an audio file (mp3, wav, m4a, ogg) and click Process Audio.")
    if not PYDUB_AVAILABLE:
        st.warning("Audio clipping requires pydub + ffmpeg. Please install them to enable clip downloads.")

left_col, right_col = st.columns([1,2], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an audio file to upload", type=["mp3", "wav", "m4a", "ogg"])
    if uploaded_file is not None:
        st.success(f"File ready: {uploaded_file.name}")
        audio_data = uploaded_file.read()
    else:
        audio_data = None

    if uploaded_file is not None:
        if st.button("Process Audio", key="process_btn"):
            status_placeholder = st.empty()
            status_placeholder.info("Uploading audio to AssemblyAI...")
            try:
                audio_url = upload_audio(audio_data, uploaded_file.name)
                status_placeholder.success("Audio uploaded successfully.")
            except Exception as e:
                status_placeholder.error(f"Upload failed: {e}")
                st.stop()

            status_placeholder.info("Transcribing audio...")
            try:
                transcript_id = transcribe_audio_assemblyai(audio_url)
                transcript_data = get_transcription_result(transcript_id)
                status_placeholder.success("Transcription completed.")
            except Exception as e:
                status_placeholder.error(f"Transcription failed: {e}")
                st.stop()

            status_placeholder.info("Performing speaker diarization...")
            try:
                diarization_id = diarize_audio_assemblyai(audio_url)
                diarization_data = get_diarization_result(diarization_id)
                status_placeholder.success("Diarization completed.")
            except Exception as e:
                status_placeholder.error(f"Diarization failed: {e}")
                st.stop()

            aligned_dialogue = []
            for segment in diarization_data:
                speaker = segment.get("speaker", "Unknown")
                start_time = segment.get("start", 0) / 1000
                end_time = segment.get("end", 0) / 1000
                text = segment.get("text", "")
                aligned_dialogue.append(f"[{speaker} {start_time:.2f}s - {end_time:.2f}s] {text}")

            status_placeholder.info("Generating summary with Gemini...")
            try:
                dialogue_text = "\n".join(aligned_dialogue)
                summary = summarize_text_gemini(dialogue_text)
                
                # Extract trendy content and key moments
                trendy_match = re.search(r'##\s*Trendy Content\s*(.*?)(##\s*Key Moments|$)', summary, re.DOTALL | re.IGNORECASE)
                trendy_content = trendy_match.group(1).strip() if trendy_match else "No trendy content identified."
                
                key_moments_match = re.search(r'##\s*Key Moments\s*(.*)', summary, re.DOTALL | re.IGNORECASE)
                key_moments = key_moments_match.group(1).strip() if key_moments_match else "No key moments identified."
                
                # Extract summary (before Trendy Content)
                summary_section = re.split(r'##\s*Trendy Content', summary, 1)[0].strip()
                
                status_placeholder.success("Summary generated.")
            except Exception as e:
                status_placeholder.error(f"Summary generation failed: {e}")
                st.stop()

            st.session_state['aligned_dialogue'] = aligned_dialogue
            st.session_state['summary'] = summary_section
            st.session_state['trendy_content'] = trendy_content
            st.session_state['key_moments'] = key_moments
            st.session_state['diarization_data'] = diarization_data
            st.session_state['audio_bytes'] = audio_data

            try:
                save_to_db(
                    uploaded_file.name,
                    st.session_state['summary'],
                    "\n".join(st.session_state['aligned_dialogue']),
                    st.session_state['trendy_content'],
                    st.session_state['key_moments']
                )
                st.success("✅ Data saved to database.")
            except Exception as e:
                st.error(f"Database save failed: {e}")
                
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dialogue", "Summary", "Trendy Content", "Key Moments", "Chat"])

    with tab1:
        st.header("🗣️ Speaker-Labeled Dialogue")
        if 'aligned_dialogue' in st.session_state:
            for line in st.session_state['aligned_dialogue']:
                parts = line.split("] ", 1)
                if len(parts) == 2:
                    timestamp_label = parts[0][1:]
                    text = parts[1]
                    st.markdown(f"<div class=\"dialogue-box\"><div class=\"speaker-label\">{timestamp_label}</div><div style=\"margin-top:8px\">{text}</div></div>", unsafe_allow_html=True)
            st.download_button(label="Download Dialogue", data="\n".join(st.session_state['aligned_dialogue']), file_name="aligned_dialogue.txt", mime="text/plain")
        else:
            st.info("Process an audio file to view dialogue.")

    with tab2:
        st.header("📝 Summary")
        if 'summary' in st.session_state:
            st.markdown(st.session_state['summary'])
            st.download_button(label="Download Summary", data=st.session_state['summary'], file_name="summary.txt", mime="text/plain")
        else:
            st.info("Process an audio file to view summary.")

    with tab3:
        st.header("🔥 Trendy Content")
        if 'trendy_content' in st.session_state:
            trendy_content = st.session_state['trendy_content']
            if trendy_content and trendy_content != "No trendy content identified.":
                raw_lines = [ln.strip() for ln in trendy_content.split("\n")]
                trendy_items = []
                for ln in raw_lines:
                    if not ln:
                        continue
                    cleaned = re.sub(r"^(\-|\*|•|\d+\.)\s*", "", ln).strip()
                    if cleaned:
                        trendy_items.append(cleaned)

                if not trendy_items:
                    st.info("No trendy content identified.")
                else:
                    diarization_data = st.session_state.get('diarization_data', [])
                    audio_bytes = st.session_state.get('audio_bytes', None)

                    for idx, item in enumerate(trendy_items, start=1):
                        st.markdown(f"### ✨ Trendy Clip {idx}")
                        st.markdown(f"**Text:** {item}")

                        span = find_trendy_span_ms(diarization_data, item, mode=mode, allow_bracket_fallback=allow_bracket)
                        if span:
                            start_ms, end_ms = span
                            st.markdown(f"**Timestamps:** `{ms_to_mmss(start_ms)} – {ms_to_mmss(end_ms)}`")

                            if PYDUB_AVAILABLE and audio_bytes:
                                try:
                                    clip_bytes = make_clip_bytes(audio_bytes, start_ms, end_ms, out_format="mp3")
                                    if clip_bytes:
                                        st.audio(BytesIO(clip_bytes), format="audio/mp3")
                                        st.download_button(label=f"⬇️ Download Clip {idx}", data=clip_bytes, file_name=f"trendy_clip_{idx}_{ms_to_mmss(start_ms)}-{ms_to_mmss(end_ms)}.mp3", mime="audio/mp3", key=f"dl_trendy_{idx}")
                                    else:
                                        st.warning("Clip could not be generated (check ffmpeg/pydub).")
                                except Exception as e:
                                    st.warning(f"Clip error: {e}")
                            else:
                                if not PYDUB_AVAILABLE:
                                    st.info("Install pydub + ffmpeg to enable clip download and preview.")
                                elif not audio_bytes:
                                    st.info("Original audio bytes not found for clipping.")
                        else:
                            st.warning("Could not confidently locate timestamps for this item in the transcript.")
                        st.divider()
            else:
                st.info("No trendy content identified.")
        else:
            st.info("Process an audio file to view trendy content.")

    with tab4:
        st.header("⭐ Key Moments")
        if 'key_moments' in st.session_state:
            key_moments = st.session_state['key_moments']
            if key_moments and key_moments != "No key moments identified.":
                st.markdown(key_moments)
            else:
                st.info("No key moments identified.")
        else:
            st.info("Process an audio file to view key moments.")

    with tab5:
        st.header("💬 Chat About the Audio")
        if 'aligned_dialogue' in st.session_state and 'summary' in st.session_state:
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []

            user_input = st.chat_input("Ask a question about the audio content...")
            if user_input:
                context = {
                    "transcription": "\n".join(st.session_state['aligned_dialogue']),
                    "summary": st.session_state['summary']
                }
                try:
                    with st.spinner("Generating chat response..."):
                        response = chat_with_gemini(user_input, context, st.session_state['chat_history'])
                    st.session_state['chat_history'].append({
                        "question": {"role": "user", "content": user_input},
                        "answer": {"role": "assistant", "content": response}
                    })
                except Exception as e:
                    st.error(f"Chat response failed: {e}")

            if st.button("Clear Chat History", key="clear_chat_btn"):
                st.session_state['chat_history'] = []

            for pair in st.session_state['chat_history']:
                with st.chat_message(pair["question"]["role"]):
                    st.markdown(pair["question"]["content"])
                with st.chat_message(pair["answer"]["role"]):
                    st.markdown(pair["answer"]["content"])
        else:
            st.info("Process an audio file to enable the chat feature.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("Made with ❤️ — AssemblyAI + Gemini. Keep API keys private in production.")

with st.expander("📂 View Saved Transcripts"):
    transcripts = fetch_all_transcripts()
    if transcripts:
        for tid, fname, summary, dialogue, trendy_content, key_moments, created in transcripts:
            st.markdown(f"**ID {tid}** — {fname} ({created})")
            st.text_area("Summary", summary, height=100, key=f"summary_{tid}")
            st.text_area("Dialogue", dialogue, height=150, key=f"dialogue_{tid}")
            st.text_area("Trendy Content", trendy_content, height=100, key=f"trendy_{tid}")
            st.text_area("Key Moments", key_moments, height=100, key=f"moments_{tid}")
            st.divider()
        # Add export button for clean text output
        export_data = export_transcripts_to_text()
        st.download_button(
            label="⬇️ Export All Transcripts",
            data=export_data,
            file_name="transcripts_export.txt",
            mime="text/plain"
        )
    else:
        st.info("No transcripts saved yet.")
