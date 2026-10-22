from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader=PyPDFLoader
)

docs = loader.load()

# every document loader fn has a load and a lazy load fn to load docs to the memory
# load: it does eager load ie..if a pdf has 400 pages it will load all 400 pages to memory 
# returns a list of documents

# lazy load: instead of giving a list of docs it gives a generator of documents. 
# using this u can load one page at a time to memory. After performing operations to that document 
# it is removed from the memory