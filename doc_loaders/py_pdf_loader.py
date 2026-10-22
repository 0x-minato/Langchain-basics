from langchain_community.document_loaders import PyPdfLoader

loader = PyPdfLoader('any-pdf')

docs = loader.load()

