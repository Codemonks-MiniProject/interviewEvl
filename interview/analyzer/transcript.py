import whisper
import spacy
import language_tool_python

# Load spaCy model and language tool (load once at module level for performance)
nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool('en-US')

TECHNICAL_KEYWORDS = [
    'database', 'API', 'algorithm', 'recursion', 'OOP',
    'Java', 'Python', 'SQL', 'data structure', 'inheritance'
]

def transcribe_audio(audio_path):
    # Using a more accurate model ("small" instead of "base")
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    transcript = result.get('text', '')
    # Validate transcript length (if too short, flag as poor quality)
    if len(transcript.split()) < 5:
        transcript = "Transcript too short. Please ensure clear speech during recording."
    return transcript

def analyze_technical_content(transcript):
    doc = nlp(transcript)

    # Keyword Score: count how many technical keywords appear relative to total keywords expected
    keyword_hits = sum(1 for kw in TECHNICAL_KEYWORDS if kw.lower() in transcript.lower())
    # For instance, if all keywords appear, score is 100%
    keyword_score = min(keyword_hits / len(TECHNICAL_KEYWORDS), 1.0) * 100

    # Entity Richness: count unique entities; assume 10+ unique entities is ideal.
    unique_entities = set(ent.text.lower() for ent in doc.ents if len(ent.text) > 2)
    entity_score = min(len(unique_entities) / 10, 1.0) * 100

    # Grammar Quality: use language_tool to count errors, but only penalize above a minimum threshold
    matches = tool.check(transcript)
    error_count = len(matches)
    # Only penalize if errors are greater than, say, 5; otherwise, give a near perfect score.
    if error_count <= 5:
        grammar_score = 100
    else:
        # For each error above 5, deduct 5 points
        grammar_score = max(100 - ((error_count - 5) * 5), 0)

    # Final technical score as a weighted average: 50% keywords, 20% entity richness, 30% grammar quality.
    technical_score = round(
        (keyword_score * 0.5) + (entity_score * 0.2) + (grammar_score * 0.3),
        2
    )
    return technical_score