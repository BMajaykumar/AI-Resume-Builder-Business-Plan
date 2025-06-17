# generator.py
def generate_report(analysis, rag_result):
    return {
        "clarity_score": analysis["clarity_score"],
        "filler_words_used": analysis["filler_word_count"],
        "verbosity_index": analysis["verbosity"],
        "rag_summary": rag_result["rag_response"],
        "recommendation": "Proceed to next round with notes" if analysis["clarity_score"] > 6 else "Needs communication improvement"
    }