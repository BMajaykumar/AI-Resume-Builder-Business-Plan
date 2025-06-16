# analyzer.py
def analyze_transcript(text):
    import re
    filler_words = ["um", "uh", "like", "you know", "so"]
    filler_count = sum(text.lower().count(word) for word in filler_words)
    word_count = len(text.split())
    verbosity = "High" if word_count > 300 else "Moderate"
    clarity_score = round(10 - (filler_count * 0.3), 2)

    return {
        "clarity_score": clarity_score,
        "filler_word_count": filler_count,
        "verbosity": verbosity
    }
