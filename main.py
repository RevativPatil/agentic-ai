from .answer import load_pdfs, generate_response, generate_summary # type: ignore

load_pdfs()

queries = [
    "What are the technical skills of Dishank?",
    "Tell me about Bonish.",
    "Tell me about Revati.",
    "Tell me education details."
]

for q in queries:
    print(f"Q: {q}")
    print(f"A: {generate_response(q)}\n{'-'*50}\n")

print("📄 Full Document Summary:\n")
print(generate_summary())
