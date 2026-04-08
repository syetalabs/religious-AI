import requests

# API URL (your backend)
API_URL = "http://localhost:8000/ask"


def ask_question(question):
    """Send a question to the chatbot API"""
    response = requests.post(
        API_URL,
        json={
            "question": question,
            "religion": "Buddhism",
            "language": "English"
        }
    )

    return response.json()


def run_test():
    """Run a simple test"""
    question = "What is Buddhism?"

    print("Question:", question)

    try:
        result = ask_question(question)

        print("\nAnswer:")
        print(result.get("answer", "No answer"))

        print("\nSources:")
        print(result.get("sources", []))

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    run_test()