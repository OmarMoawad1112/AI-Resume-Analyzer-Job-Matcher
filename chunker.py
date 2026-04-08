# Import the RecursiveCharacterTextSplitter from LangChain
# This tool is used to split long text into smaller chunks in a smart way
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text):
    """
    This function takes a long text as input
    and returns a list of smaller text chunks.
    """

    # Create a text splitter object
    splitter = RecursiveCharacterTextSplitter(
        
        # Maximum size of each chunk (in characters)
        # Example: 500 means each chunk will have up to 500 characters
        chunk_size=500,
        
        # Number of overlapping characters between consecutive chunks
        # This helps preserve context between chunks
        # Example: last 100 characters of chunk1 will appear in chunk2
        chunk_overlap=50
    )

    # Split the input text into smaller chunks
    # The splitter will:
    # 1. Try splitting by paragraphs
    # 2. If too big → split by lines
    # 3. If still too big → split by words
    # 4. If needed → split by characters
    chunks = splitter.split_text(text)

    # Return the list of chunks
    return chunks