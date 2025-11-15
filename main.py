from utils.pdf_loader import load_pdf, chunk_text
from utils.rag_engine_gemini import RAGEngine

def main():
    # Ask user for PDF path dynamically
    pdf_path = input("Enter the PDF file path: ").strip()

    # Load and chunk PDF
    print("Loading PDF...")
    text = load_pdf(pdf_path)
    print("PDF loaded successfully.")

    print("Chunking text...")
    chunks = chunk_text(text, chunk_size=800, overlap=150)
    print(f"Created {len(chunks)} chunks.")

    # Build RAG index
    rag = RAGEngine()
    print("Building index...")
    rag.build_index(chunks)
    print("Index built successfully.")

    # Ask question dynamically
    question = input("Enter your question: ")

    # Query model
    print("Querying RAG engine...")
    answer = rag.query(question)

    print("\nAnswer:", answer)


if __name__ == "__main__":
    main()
