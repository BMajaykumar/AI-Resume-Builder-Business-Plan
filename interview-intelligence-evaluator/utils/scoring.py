# utils/scoring.py
def keyword_match_score(transcript, keywords):
    found = [kw for kw in keywords if kw.lower() in transcript.lower()]
    missing = [kw for kw in keywords if kw.lower() not in transcript.lower()]
    return {
        "matched_keywords": found,
        "missing_keywords": missing,
        "match_score": round(len(found) / len(keywords) * 10, 2)
    }
