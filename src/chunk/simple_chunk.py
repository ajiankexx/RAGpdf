from chonkie import SentenceChunker

simple_chunker = SentenceChunker(
    tokenizer_or_token_counter="gpt2",                
    chunk_size=512,                                   
    chunk_overlap=10,                                  
    min_sentences_per_chunk=1,                        
    min_characters_per_sentence=12,                   
    approximate=True,                                 
    delim=[".", "?", "!", "\n"],                      
    include_delim="prev",                             
    return_type="texts"                              
)