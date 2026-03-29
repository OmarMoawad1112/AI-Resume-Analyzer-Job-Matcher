from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    """
    Extracts text content from a PDF file using LangChain's PyPDFLoader.
    
    Args:
        file_path (str): The local path or URL to the PDF document.
        
    Returns:
        str: A single string containing the concatenated text of all pages.
    """
    # 1. Initialize the loader: This creates an object that understands 
    # how to communicate with the PDF file structure.
    loader = PyPDFLoader(file_path)
    
    # 2. Load the document: The .load() method parses the PDF and returns 
    # a list of 'Document' objects (one for each page).
    documents = loader.load()
    
    # 3. Accumulate text: We iterate through the list of pages.
    # Each 'page' object has a 'page_content' attribute (the raw text)
    # and a 'metadata' attribute (e.g., page number, source).
    full_text = ""
    for document in documents:
        # We append the text of each page, adding a newline for separation.
        full_text += document.page_content + "\n"
        
    return full_text



# Why this matters
# PDF is:
    # messy
    # split into pages

# We:
    # extract all text
    # combine into one string
    # This becomes the input to your RAG system