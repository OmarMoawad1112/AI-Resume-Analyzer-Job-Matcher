from loader import load_pdf
from chunker import chunk_text
from embeddings import get_embeddings



# text = load_pdf("cv.pdf")
# chunks = chunk_text(text)

# print("Number of chunks:", len(chunks))
# for chunk in chunks:
#     print(chunk)
#     print('=======================\n')



chunks = ["I know Python", "I love football"]
vectors = get_embeddings(chunks)

print(vectors.shape)
print(type(vectors))
print(vectors)