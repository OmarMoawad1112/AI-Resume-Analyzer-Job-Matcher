from sentence_transformers import SentenceTransformer

# 1. Load the Model: 'all-MiniLM-L6-v2' is a high-speed, 
# lightweight model perfect for real-time resume parsing.
# It maps sentences & paragraphs to a 384-dimensional dense vector space.
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks):
    """
    Converts text chunks into numerical vectors (embeddings).
    
    Args:
        chunks (list[str]): A list of text segments from the resume.
        
    Returns:
        numpy.ndarray: A matrix where each row is a vector representing 
                       the semantic meaning of a chunk.
    """
    # 2. Vectorization: The .encode() method performs the mathematical 
    # transformation. It handles tokenization and forward passes through 
    # the Transformer layers automatically.
    embeddings = model.encode(chunks)
    
    return embeddings



# Since you are studying Natural Language Processing and building an AI-Resume Analyzer, understanding the Transformer is essential. It is the "engine" inside the model you just used (all-MiniLM-L6-v2) and the foundation of GPT, BERT, and Claude.

# Before Transformers (using older models like RNNs or LSTMs), computers read text like a human: one word at a time, from left to right. Transformers changed this by looking at the entire sentence all at once.

# The 3 Core Pillars of a Transformer
    # 1. Self-Attention (The "Context" Mechanism)
        # This is the most important part. It allows the model to understand which words in a sentence relate to each other, regardless of how far apart they are.
        # Example: "The bank of the river was muddy, so I went to the bank to withdraw money."
        # The Transformer uses "Attention" to link the first "bank" to "river" and the second "bank" to "money," giving them different mathematical meanings (embeddings).


    # 2. Positional Encoding
        # Because the Transformer processes all words simultaneously, 
        # it technically "forgets" the order of words. To fix this, 
        # it adds a unique mathematical signal (a "time stamp") to each word 
        # so it knows that "The dog bit the man" is different from "The man bit the dog."


    # 3. Parallelization
        # Unlike older models that had to wait for the previous word to finish, 
        # Transformers can process massive amounts of data in parallel. 
        # This is why we can train them on the entire internet—and why 
        # they perform so well on modern GPUs.


# The Architecture: Encoder vs. Decoder
    # The original Transformer has two main halves:
        # The Encoder: 
            # Reads and understands the input text. 
            # Models like BERT (and the SentenceTransformer you're using) 
            # are "Encoder-only." They are great for extracting meaning 
           

        # The Decoder: 
            # Predicts and generates the next word. 
            # Models like GPT are "Decoder-only." 
            # They are built for writing text.

# Why this matters for your Resume Analyzer
# When you run model.encode(chunks), the Transformer inside:
    # Breaks your resume text into Tokens.
    # Applies Self-Attention to see how "Python" relates to "5 years experience."
    # Outputs a Vector (the embedding) that captures that specific context.