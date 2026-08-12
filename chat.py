"""
OpenMath — Interactive Chat Mode

Loads the model once at startup and provides a continuous conversation loop
for real-time math Q&A. Type 'quit' or 'exit' to close gracefully.
"""

import inference


def main():
    print("OpenMath Interactive Chat Mode")
    print("Loading model... (this may take a moment)\n")
    inference.load_model()
    print("Model loaded. Ready for questions!\n")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat mode.")
            break

        cmd = question.lower()
        if cmd in ("quit", "exit"):
            print("Goodbye!")
            break

        if not question:
            continue

        print("\nOpenMath: ", end="", flush=True)
        try:
            solution = inference.generate_solution(problem=question)
            print(solution)
        except Exception as e:
            print(f"Error during generation: {e}")
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    main()
