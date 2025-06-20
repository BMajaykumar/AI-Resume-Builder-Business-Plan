import google.generativeai as genai
import os

class AppFeatureChain:
    def __init__(self):
        # Load Gemini API key
        genai.configure(api_key=os.getenv("AIzaSyDvuEhNgS4kHx9RIbz6rUPF8eNRgjWTAyk"))

        # Init Gemini model
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def fetch_app_features(self, description):
        prompt = f"""
        Given the following mobile app description:

        "{description}"

        Identify and list 5 to 7 core features that this app should support.
        Examples: GPS, Camera, Microphone, Bluetooth, Fitness Tracker, etc.

        Output as a plain list, one feature per line, no extra text.
        """
        response = self.model.generate_content(prompt)
        return [line.strip() for line in response.text.split("\n") if line.strip()]

    def generate_feature_table(self, features_dict):
        header = "| **Feature** | **Selected** | **Comments / Notes** |\n"
        divider = "|-------------|--------------|------------------------|\n"
        rows = ""
        for feature, response in features_dict.items():
            selected = "✅" if response.lower() == "yes" else "⭕️"
            comment = "Selected" if selected == "✅" else "Not selected"
            rows += f"| {feature} | {selected} | {comment} |\n"
        return header + divider + rows
