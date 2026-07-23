import pdfplumber
import math
import chromadb
from sentence_transformers import SentenceTransformer
def extractor(file): 
    page_text = []
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages,):
            text = page.extract_text() or ""
            page_text.append({"index":i+1,"text" :text})
    return page_text

def retrievel_chunk(text, chunk_size = 500, overlap = 100,seperators = None):
    seperators = ["\n\n","\n","."," "] or seperators
    
    def splits(text,sep):
        chunk,currents = [],""
        if len(text) <= chunk_size or not sep:
            return [text]

        seps,rest_seps = sep[0],sep[1:]
        parts = text.split(seps)

        
        for part in parts:
            candidate = currents + seps + part if currents else part
            if len(candidate)<= chunk_size:
                currents = candidate
            else:
                if currents:
                    chunk.append(currents)
                currents = part
            if currents:
                chunk.append(currents)
            final = []
            for c in chunk:
                if len(c) > chunk_size:
                    final.extend(splits(c,rest_seps))
                else:
                    final.append(c)
                return final
        
    raw_chunk = splits(text,seperators)
    overlapped = []
    for i,c in enumerate(raw_chunk):
        if i>0:
            c = raw_chunk[i-1][-overlap:]+c
        overlapped.append(c)
        return overlapped 
