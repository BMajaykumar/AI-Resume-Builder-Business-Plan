def generate_report(topic, questions, answers):
    report = f"# Web Research Report on: {topic}\n\n"
    report += "## Introduction\n"
    report += f"This report explores the topic **{topic}** using structured research questions and web-sourced answers.\n\n"

    for i, question in enumerate(questions):
        report += f"## {i}. {question}\n"
        report += answers[question] + "\n\n"

    report += "## Conclusion\n"
    report += "This research provides a well-rounded understanding of the topic using both reasoning and action via web search."
    return report
