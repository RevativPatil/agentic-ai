from answer import load_pdfs, generate_response, generate_summary

print("Loading PDFs into FAISS and embedding...")
load_pdfs()

queries = [
    "What are the technical skills of Dishank?",
    "Tell me about Bonish.",
    "Tell me about Revati.",
    "Tell me education details."
]

for q in queries:
    print(f"\nQ: {q}")
    print(f"A:\n{generate_response(q)}\n{'-'*50}")

print("\n📄 Full Document Summary:\n")
print(generate_summary())
