from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("/Users/willemvandemierop/Desktop/sandbox_owen/Syndicusovereenkomst_michiel.pdf") ## you can make this more generic . 
documents = loader.load() 

# print(documents)

import pprint

pprint.pp(documents[0].metadata)